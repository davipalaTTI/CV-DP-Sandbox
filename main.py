#!/usr/bin/env python3
"""
Multi-Line Object Counter - Main Entry Point

A comprehensive computer vision application for counting objects crossing
predefined lines and zones using YOLO detection and tracking.
"""

import tensorrt
import sys
import logging
import traceback
from pathlib import Path
from typing import Optional
import cv2
# Import our modules
from config_manager import ConfigManager, AppConfig
from gui_setup import InteractiveGUI
from detection_engine import DetectionEngine
from video_processor import VideoProcessor
from utils import setup_logging, check_dependencies

__version__ = "1.0.1"
__author__ = "TH"


class Application:
    """Main application class that orchestrates all components"""

    def __init__(self):
        self.config: Optional[AppConfig] = None
        self.detection_engine: Optional[DetectionEngine] = None
        self.video_processor: Optional[VideoProcessor] = None
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

            config_manager = ConfigManager()

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

            # Initialize video processor
            self.logger.info("Step 4: Initializing video processor...")
            self.video_processor = VideoProcessor(
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


def parse_arguments():
    """Parse command line arguments"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Line Object Counter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Interactive mode
  python main.py --config config.json    # Load from config file
  python main.py --debug                 # Enable debug logging
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Load configuration from file'
    )

    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug logging'
    )

    parser.add_argument(
        '--log-file', '-l',
        type=str,
        help='Log file path (default: logs/app.log)'
    )

    parser.add_argument(
        '--version', '-v',
        action='version',
        version=f'Multi-Line Object Counter v{__version__}'
    )

    parser.add_argument(
        '--no-gui',
        action='store_true',
        help='Skip interactive GUI setup (requires config file)'
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    args = parse_arguments()

    # Print banner
    print_banner()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_file = args.log_file or "logs/app.log"
    setup_logging(level=log_level, log_file=log_file)

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

        # Check for config file and no-gui flag
        config_file = args.config
        skip_gui = args.no_gui

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
        from utils import stop_logging
        stop_logging()


# Development and debugging helpers
def run_with_profiling():
    """Run with performance profiling enabled"""
    import cProfile
    import pstats
    from io import StringIO

    profiler = cProfile.Profile()
    profiler.enable()

    exit_code = main()

    profiler.disable()

    # Print profiling results
    stats_stream = StringIO()
    stats = pstats.Stats(profiler, stream=stats_stream).sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions

    print("\n" + "=" * 80)
    print("PERFORMANCE PROFILING RESULTS")
    print("=" * 80)
    print(stats_stream.getvalue())

    return exit_code


def run_with_memory_profiling():
    """Run with memory profiling enabled"""
    try:
        from memory_profiler import profile

        # Wrap main function with memory profiler
        profiled_main = profile(main)
        return profiled_main()

    except ImportError:
        print("memory_profiler not installed. Install with: pip install memory-profiler")
        return main()


if __name__ == "__main__":
    # Development mode checks
    if len(sys.argv) > 1:
        if sys.argv[1] == "--profile":
            sys.exit(run_with_profiling())
        elif sys.argv[1] == "--memory-profile":
            sys.exit(run_with_memory_profiling())

    # Normal execution
    sys.exit(main())
