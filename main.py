#!/usr/bin/env python3
"""
Multi-Line Object Counter - Main Entry Point

A comprehensive computer vision application for counting objects crossing
predefined lines and zones using YOLO detection and tracking.
"""

import sys
import gc
import logging
import traceback
from datetime import datetime
from pathlib import Path
from tkinter import messagebox
from typing import Optional, Set, Union
import cv2
# Import our modules
from utils.logger import stop_logging, setup_logging
from utils.crash_reporting import (
    close_crash_reporting,
    install_crash_reporting,
    report_exception,
)
from config_manager import AppConfig, DeploymentRequest
from deployment_manager import ManifestError, load_deployment
from gui.gui_setup import InteractiveGUI
from gui.startup_window import StartupWindow
from core.detection_engine import DetectionEngine
from core.video_processing.camera_runner import CameraRunner
from core.video_processing.batch_runner import BatchRunner
from scripts.scheduled_runner import ScheduleSupervisor
from utils.profiling import run_with_profiling, run_with_memory_profiling
from utils.system_checks import check_dependencies
from utils.cli_parser import parse_arguments

__version__ = "1.0.1"
__author__ = "TH"

class Application:
    """Main application class that orchestrates all components"""

    def __init__(self):
        self.config: Optional[AppConfig] = None
        self.detection_engine: Optional[DetectionEngine] = None
        self.video_processor: Optional[
            Union[CameraRunner, BatchRunner, ScheduleSupervisor]
        ] = None
        self.logger = logging.getLogger(__name__)

    # In main.py, replace the initialize method in the Application class:

    def initialize(
        self,
        config_file: Optional[str] = None,
        skip_gui: bool = False,
        headless: bool = False,
        source_name: Optional[str] = None,
        stop_at: Optional[datetime] = None,
        window_index: int = 0,
        window_count: int = 1,
    ) -> bool:
        """
        Initialize the application with user configuration or from file

        Args:
            config_file: Optional path to configuration file to load
            skip_gui: If True, skip GUI setup when config_file is provided

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info(f"Starting Multi-Line Object Counter v{__version__}")

            config_manager = StartupWindow(discover_sources=not (config_file and skip_gui))

            # Try to load config from file if provided
            if config_file:
                if not Path(config_file).exists():
                    self.logger.error(f"Configuration file not found: {config_file}")
                    return False

                self.logger.info(f"Loading configuration from file: {config_file}")
                self.config = config_manager.load_config(config_file)

                if self.config is None:
                    self.logger.error("Failed to load configuration file")
                    return False

                validation_errors = config_manager.validate_config(self.config)
                if validation_errors:
                    for error in validation_errors:
                        self.logger.error(f"Invalid configuration: {error}")
                    return False

                if skip_gui and not (self.config.lines_config or self.config.zones_config):
                    self.logger.error(
                        "Non-interactive mode requires at least one saved counting line or zone"
                    )
                    return False

                self.config.runtime_headless = headless
                self.config.runtime_stop_at = stop_at
                self.config.runtime_window_index = max(0, int(window_index))
                self.config.runtime_window_count = max(1, int(window_count))
                if source_name:
                    self.config.source_name = source_name.strip()

                self.logger.info("Configuration loaded from file successfully")

                # Initialize detection engine
                self.logger.info("Initializing detection engine...")
                self.detection_engine = DetectionEngine(
                    model_path=self.config.model_path,
                    confidence_threshold=self.config.confidence_threshold,
                    device=self.config.device
                )

                # Warm up the model
                self.detection_engine.warmup()
                self.logger.info("Detection engine initialized and warmed up")

                # Skip GUI setup if requested and we have lines configured
                if skip_gui:
                    self.logger.info("Skipping GUI setup (using saved configuration)")
                else:
                    # Run GUI setup even with loaded config (for modifications)
                    self.logger.info("Running interactive setup...")
                    gui_setup = InteractiveGUI(
                        config=self.config,
                        class_names=self.detection_engine.class_names
                    )

                    lines_config, zones_config, exclusion_zones = gui_setup.run_setup()

                    if not lines_config and not zones_config:
                        self.logger.error("No counting lines or zones configured. Exiting.")
                        return False

                    # Update config with GUI results
                    self.config.lines_config = lines_config
                    self.config.zones_config = zones_config
                    self.config.exclusion_zones = exclusion_zones

                    # Save the updated configuration
                    config_path = Path(self.config.output_folder) / "config.json"
                    if config_manager.save_config(self.config, config_path):
                        self.logger.info(f"Configuration saved to: {config_path}")
                        self.logger.info("You can reuse this config with: --config config.json --no-gui")

            else:
                # No config file - go through normal GUI setup
                self.logger.info("Step 1: Getting initial configuration...")
                startup_selection = config_manager.get_initial_config(
                    new_camera_callback=lambda reserved_outputs, parent: self._create_source_config(
                        config_manager, parent, reserved_outputs
                    )
                )

                if startup_selection is None:
                    self.logger.info("Configuration canceled by user")
                    return False

                if isinstance(startup_selection, DeploymentRequest):
                    try:
                        deployment = load_deployment(Path(startup_selection.manifest_path))
                    except ManifestError as exc:
                        self.logger.error(f"Invalid deployment: {exc}")
                        return False
                    self.video_processor = ScheduleSupervisor(
                        deployment,
                        manifest_path=Path(startup_selection.manifest_path),
                    )
                    self.logger.info(
                        "Deployment loaded: %d camera(s)", len(deployment.jobs)
                    )
                    return True

                self.config = startup_selection

                self.logger.info("Configuration loaded")

                # Initialize detection engine
                self.logger.info("Step 2: Initializing detection engine...")
                self.detection_engine = DetectionEngine(
                    model_path=self.config.model_path,
                    confidence_threshold=self.config.confidence_threshold,
                    device=self.config.device
                )

                self.detection_engine.warmup()
                self.logger.info("Detection engine initialized and warmed up")

                # Check if user loaded a complete config and wants to skip GUI setup
                skip_interactive = getattr(self.config, '_skip_gui_setup', False)
                
                if skip_interactive and (self.config.lines_config or self.config.zones_config):
                    self.logger.info("Using loaded configuration (skipping interactive setup)")
                else:
                    # Interactive GUI setup
                    self.logger.info("Step 3: Starting interactive setup...")
                    gui_setup = InteractiveGUI(
                        config=self.config,
                        class_names=self.detection_engine.class_names
                    )

                    lines_config, zones_config, exclusion_zones = gui_setup.run_setup()

                    if not lines_config and not zones_config:
                        self.logger.error("No counting lines or zones configured. Exiting.")
                        return False

                    # Update config
                    self.config.lines_config = lines_config
                    self.config.zones_config = zones_config
                    self.config.exclusion_zones = exclusion_zones

                    # Save configuration for future use
                    config_path = Path(self.config.output_folder) / "config.json"
                    if config_manager.save_config(self.config, config_path):
                        self.logger.info(f"Configuration saved to: {config_path}")
                        self.logger.info("You can reuse this config with: --config {config_path} --no-gui")

            # Common initialization for both paths
            # Configure exclusion zones in detection engine
            if (
                self.config.exclusion_zones
                and self.config.input_source
                and not self.config.is_camera
            ):
                cap = cv2.VideoCapture(str(self.config.input_source))
                ret, frame = cap.read()
                if ret:
                    self.detection_engine.set_exclusion_zones(self.config.exclusion_zones, frame.shape)
                cap.release()

            # Build allowed class set
            selected = set()
            for L in self.config.lines_config:
                selected.update(int(c) for c in getattr(L, "classes", []) or [])
            for Z in self.config.zones_config:
                selected.update(int(c) for c in getattr(Z, "classes", []) or [])

            if selected:
                self.config.allowed_classes = selected
                self.detection_engine.update_allowed_classes(selected)
                self.logger.info(f"Restricting to classes: {sorted(selected)}")

            self.logger.info(f"Setup completed:")
            self.logger.info(f"  Lines configured: {len(self.config.lines_config)}")
            self.logger.info(f"  Zones configured: {len(self.config.zones_config)}")
            self.logger.info(f"  Exclusion zones: {len(self.config.exclusion_zones)}")

            # Initialize the correct video runner based on config
            self.logger.info("Step 4: Initializing video runner...")
            if self.config.is_camera:
                self.logger.info("Setting up CameraRunner for live feed.")
                self.video_processor = CameraRunner(
                    config=self.config,
                    detection_engine=self.detection_engine
                )
            else:
                self.logger.info("Setting up BatchRunner for file/folder processing.")
                self.video_processor = BatchRunner(
                    config=self.config,
                    detection_engine=self.detection_engine
                )

            self.logger.info("Application initialization completed successfully")
            return True

        except KeyboardInterrupt:
            self.logger.info("Initialization interrupted by user")
            return False
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.logger.error(traceback.format_exc())
            report_exception("Application initialization failed", e)
            return False

    def _create_source_config(
        self,
        config_manager: StartupWindow,
        parent,
        reserved_output_folders: Optional[Set[str]] = None,
    ) -> Optional[str]:
        """Configure, draw, and persist one source before returning to deployment setup."""
        config = config_manager.get_new_source_config(parent, reserved_output_folders)
        if config is None:
            return None

        config_path = Path(config.output_folder) / "config.json"
        if (
            getattr(config, "_skip_gui_setup", False)
            and (config.lines_config or config.zones_config)
        ):
            if config_manager.save_config(config, config_path):
                return str(config_path.resolve())
            return None

        preview_frame = config_manager.get_source_preview(parent, config)
        if preview_frame is None:
            self.logger.warning("New source setup stopped because preview validation failed")
            return None

        engine = None
        try:
            self.logger.info("Initializing model for new camera setup")
            engine = DetectionEngine(
                model_path=config.model_path,
                confidence_threshold=config.confidence_threshold,
                device=config.device,
            )
            engine.warmup()

            gui_setup = InteractiveGUI(
                config=config,
                class_names=engine.class_names,
                preview_frame=preview_frame,
                dialog_parent=parent,
            )
            lines_config, zones_config, exclusion_zones = gui_setup.run_setup()
            if not lines_config and not zones_config:
                self.logger.warning("New camera setup canceled without counting geometry")
                messagebox.showinfo(
                    "Source Not Added",
                    "The source was not added because no counting line or zone was confirmed.",
                    parent=parent,
                )
                return None

            config.lines_config = lines_config
            config.zones_config = zones_config
            config.exclusion_zones = exclusion_zones
            if not config_manager.save_config(config, config_path):
                self.logger.error("Could not save new camera config: %s", config_path)
                messagebox.showerror(
                    "Source Not Saved",
                    f"The source configuration could not be saved to:\n{config_path}",
                    parent=parent,
                )
                return None

            self.logger.info("New camera config saved: %s", config_path)
            return str(config_path.resolve())
        except Exception as exc:
            self.logger.error("New camera setup failed: %s", exc)
            self.logger.debug(traceback.format_exc())
            messagebox.showerror(
                "Source Setup Failed",
                f"The source could not be added.\n\n{exc}",
                parent=parent,
            )
            return None
        finally:
            if engine is not None:
                del engine
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def run(self) -> int:
        """
        Run the main processing loop

        Returns:
            Exit code (0 for success, 1 for error)
        """
        if not self.video_processor:
            self.logger.error("Application not properly initialized")
            return 1

        try:
            self.logger.info("Starting video processing...")

            # Run the main processing loop
            results = self.video_processor.run()

            # Note: Final results are NOT exported here because the master_event_log.xlsx
            # is already incrementally updated during processing. Creating additional
            # all_events_*.json/.csv/.xlsx files would be redundant.
            if results:
                self.logger.info("Processing complete. All events already saved to master_event_log.xlsx")

            self.logger.info("Processing completed successfully")
            return 0

        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user")
            return 0
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            self.logger.error(traceback.format_exc())
            report_exception("Camera/video processing failed", e)
            return 1

    def cleanup(self):
        """Clean up resources"""
        try:
            if self.video_processor:
                self.video_processor.cleanup()
            self.logger.info("Cleanup completed")
        except Exception as e:
            self.logger.warning(f"Cleanup error: {e}")

def print_banner():
    """Print application banner"""
    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║              Multi-Line Object Counter v{__version__}                ║
║                                                              ║
║  A computer vision application for counting objects crossing ║
║  predefined lines and zones using YOLO detection.            ║
║                                                              ║
║  Author: {__author__:<49}   ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main(
    config_file: str = None,
    skip_gui: bool = False,
    debug: bool = False,
    log_file: str = None,
    source_name: str = None,
    stop_at: datetime = None,
    headless: bool = False,
    window_index: int = 0,
    window_count: int = 1,
    crash_report_dir: str = None,
) -> int:
    """
    Main entry point

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # DELETED: args = parse_arguments() <-- We don't need this inside the function anymore!

    # Print banner
    try:
        print_banner()
    except NameError:
        pass # Just in case print_banner isn't imported in your current snippet

    # Setup logging using the variables passed into the function
    log_level = logging.DEBUG if debug else logging.INFO
    log_file_path = log_file or "logs/app.log"
    setup_logging(level=log_level, log_file=log_file_path)

    logger = logging.getLogger(__name__)
    try:
        report_source = source_name or (
            Path(config_file).stem if config_file else "application"
        )
        crash_reporter = install_crash_reporting(
            log_file=log_file_path,
            source_name=report_source,
            report_dir=crash_report_dir,
        )
        logger.info("Crash reporting armed: %s", crash_reporter.report_path)
    except OSError as exc:
        logger.warning("Crash reporting could not be initialized: %s", exc)

    try:
        # Check dependencies
        logger.info("Checking dependencies...")
        missing_deps = check_dependencies()
        if missing_deps:
            logger.error(f"Missing dependencies: {', '.join(missing_deps)}")
            print(f"\nError: Missing required dependencies: {', '.join(missing_deps)}")
            print("Please install them using: pip install -r requirements.txt")
            return 1

        # Create application instance
        app = Application()

        # DELETED: config_file = args.config
        # DELETED: skip_gui = args.no_gui
        # We don't need these because they are already passed into the function!

        headless = bool(headless)
        skip_gui = bool(skip_gui or headless)

        # Validate non-interactive usage
        if skip_gui and not config_file:
            logger.error("--no-gui requires --config to be specified")
            print("\nError: --no-gui requires a configuration file to be specified with --config")
            return 1

        # Initialize application with config options
        if not app.initialize(
            config_file=config_file,
            skip_gui=skip_gui,
            headless=headless,
            source_name=source_name,
            stop_at=stop_at,
            window_index=window_index,
            window_count=window_count,
        ):
            logger.info("Application initialization failed or was canceled")
            return 1 if skip_gui else 0

        # Run main processing
        exit_code = app.run()

        # Cleanup
        app.cleanup()

        return exit_code

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\nApplication interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        report_exception("Unhandled application error", e)
        print(f"\nUnexpected error: {e}")
        return 1
    finally:
        # Ensure logging queue is flushed before exit
        try:
            # FIXED: Updated the import path to match your new folder structure
            stop_logging()
        except ImportError:
            pass
        close_crash_reporting()


if __name__ == "__main__":
    args = parse_arguments()

    # Securely grab debug and log_file flags in case they haven't been added to cli_parser yet
    is_debug = getattr(args, 'debug', False)
    log_path = getattr(args, 'log_file', None)

    if args.profile:
        sys.exit(run_with_profiling(
            main,
            config_file=args.config,
            skip_gui=args.no_gui,
            headless=args.headless,
            debug=is_debug,
            log_file=log_path,
            source_name=args.source_name,
            stop_at=args.stop_at,
            window_index=args.window_index,
            window_count=args.window_count,
            crash_report_dir=args.crash_report_dir,
        ))

    elif args.memory_profile:
        sys.exit(run_with_memory_profiling(
            main,
            config_file=args.config,
            skip_gui=args.no_gui,
            headless=args.headless,
            debug=is_debug,
            log_file=log_path,
            source_name=args.source_name,
            stop_at=args.stop_at,
            window_index=args.window_index,
            window_count=args.window_count,
            crash_report_dir=args.crash_report_dir,
        ))

    else:
        sys.exit(main(
            config_file=args.config,
            skip_gui=args.no_gui,
            headless=args.headless,
            debug=is_debug,
            log_file=log_path,
            source_name=args.source_name,
            stop_at=args.stop_at,
            window_index=args.window_index,
            window_count=args.window_count,
            crash_report_dir=args.crash_report_dir,
        ))
