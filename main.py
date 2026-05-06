#!/usr/bin/env python3
"""
Multi-Line Object Counter - Main Entry Point

A comprehensive computer vision application for counting objects crossing
predefined lines and zones using YOLO detection and tracking.
"""

import sys
import logging
import traceback
from pathlib import Path
from typing import Optional, Union
import cv2
# Import our modules
from utils.logger import stop_logging, setup_logging
from config_manager import AppConfig
from gui.gui_setup import InteractiveGUI
from gui.startup_window import StartupWindow
from core.detection_engine import DetectionEngine
from core.video_processing.camera_runner import CameraRunner
from core.video_processing.batch_runner import BatchRunner
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
        self.video_processor: Optional[Union[CameraRunner, BatchRunner]] = None
        self.logger = logging.getLogger(__name__)

    # In main.py, replace the initialize method in the Application class:

    def initialize(self, config_file: Optional[str] = None, skip_gui: bool = False) -> bool:
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

            config_manager = StartupWindow()

            # Try to load config from file if provided
            if config_file and Path(config_file).exists():
                self.logger.info(f"Loading configuration from file: {config_file}")
                self.config = config_manager.load_config(config_file)

                if self.config is None:
                    self.logger.error("Failed to load configuration file")
                    return False

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
                if skip_gui and self.config.lines_config:
                    self.logger.info("Skipping GUI setup (using saved configuration)")
                else:
                    # Run GUI setup even with loaded config (for modifications)
                    self.logger.info("Running interactive setup...")
                    gui_setup = InteractiveGUI(
                        config=self.config,
                        class_names=self.detection_engine.class_names
                    )

                    lines_config, zones_config, exclusion_zones = gui_setup.run_setup()

                    if not lines_config:
                        self.logger.error("No counting lines configured. Exiting.")
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
                self.config = config_manager.get_initial_config()

                if self.config is None:
                    self.logger.info("Configuration canceled by user")
                    return False

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
                
                if skip_interactive and self.config.lines_config:
                    self.logger.info("Using loaded configuration (skipping interactive setup)")
                else:
                    # Interactive GUI setup
                    self.logger.info("Step 3: Starting interactive setup...")
                    gui_setup = InteractiveGUI(
                        config=self.config,
                        class_names=self.detection_engine.class_names
                    )

                    lines_config, zones_config, exclusion_zones = gui_setup.run_setup()

                    if not lines_config:
                        self.logger.error("No counting lines configured. Exiting.")
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
            if self.config.exclusion_zones and self.config.input_source:
                if self.config.is_camera:
                    cap = cv2.VideoCapture(self.config.input_source)
                else:
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
            return False

    def run(self) -> int:
        """
        Run the main processing loop

        Returns:
            Exit code (0 for success, 1 for error)
        """
        if not self.config or not self.video_processor:
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

def main(config_file: str = None, skip_gui: bool = False, debug: bool = False, log_file: str = None) -> int:
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

        # Validate no-gui usage
        if skip_gui and not config_file:
            logger.error("--no-gui requires --config to be specified")
            print("\nError: --no-gui requires a configuration file to be specified with --config")
            return 1

        # Initialize application with config options
        if not app.initialize(config_file=config_file, skip_gui=skip_gui):
            logger.info("Application initialization failed or was canceled")
            return 0

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
        print(f"\nUnexpected error: {e}")
        return 1
    finally:
        # Ensure logging queue is flushed before exit
        try:
            # FIXED: Updated the import path to match your new folder structure
            stop_logging()
        except ImportError:
            pass


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
            debug=is_debug,
            log_file=log_path
        ))

    elif args.memory_profile:
        sys.exit(run_with_memory_profiling(
            main,
            config_file=args.config,
            skip_gui=args.no_gui,
            debug=is_debug,
            log_file=log_path
        ))

    else:
        sys.exit(main(
            config_file=args.config,
            skip_gui=args.no_gui,
            debug=is_debug,
            log_file=log_path
        ))
