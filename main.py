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
from typing import Optional
import cv2
# Import our modules
from config_manager import ConfigManager, AppConfig
from gui_setup import InteractiveGUI
from detection_engine import DetectionEngine
from video_processor import VideoProcessor
from results_export import ResultsExporter
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

    def initialize(self) -> bool:
        """
        Initialize the application with user configuration

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info(f"Starting Multi-Line Object Counter v{__version__}")

            # Step 1: Get initial configuration from user
            self.logger.info("Step 1: Getting initial configuration...")
            config_manager = ConfigManager()
            self.config = config_manager.get_initial_config()

            if self.config is None:
                self.logger.info("Configuration canceled by user")
                return False

            self.logger.info(f"Configuration loaded:")
            self.logger.info(f"  Model: {self.config.model_path}")
            self.logger.info(f"  Input: {self.config.input_source}")
            self.logger.info(f"  Output: {self.config.output_folder}")
            self.logger.info(f"  Camera mode: {self.config.is_camera}")
            self.logger.info(f"  Zones enabled: {self.config.enable_zones}")

            # Step 2: Initialize detection engine
            self.logger.info("Step 2: Initializing detection engine...")
            self.detection_engine = DetectionEngine(
                model_path=self.config.model_path,
                confidence_threshold=self.config.confidence_threshold,
                device=self.config.device
            )

            # Warm up the model
            self.detection_engine.warmup()
            self.logger.info("Detection engine initialized and warmed up")

            # Step 3: Interactive GUI setup for lines and zones
            self.logger.info("Step 3: Starting interactive setup...")
            gui_setup = InteractiveGUI(
                config=self.config,
                class_names=self.detection_engine.class_names
            )

            # Get all three configurations
            lines_config, zones_config, exclusion_zones = gui_setup.run_setup()

            if not lines_config:
                self.logger.error("No counting lines configured. Exiting.")
                return False

            # Update config with GUI results
            self.config.lines_config = lines_config
            self.config.zones_config = zones_config
            self.config.exclusion_zones = exclusion_zones  # Add this

            # NEW: Configure exclusion zones in detection engine
            if exclusion_zones and self.config.input_source:
                # Get frame shape from first frame
                cap = cv2.VideoCapture(self.config.input_source)
                ret, frame = cap.read()
                if ret:
                    self.detection_engine.set_exclusion_zones(exclusion_zones, frame.shape)
                cap.release()

            self.logger.info(f"Setup completed:")
            self.logger.info(f"  Lines configured: {len(lines_config)}")
            self.logger.info(f"  Zones configured: {len(zones_config)}")
            self.logger.info(f"  Exclusion zones: {len(exclusion_zones)}")

            # Build the global allowed class set from lines + zones
            selected = set()
            for L in self.config.lines_config:
                selected.update(int(c) for c in getattr(L, "classes", []) or [])
            for Z in self.config.zones_config:
                selected.update(int(c) for c in getattr(Z, "classes", []) or [])

            # Apply to config + engine (so detection & drawing are filtered)
            if selected:
                self.config.allowed_classes = selected
                self.detection_engine.update_allowed_classes(selected)
                self.logger.info(f"Restricting to classes: {sorted(selected)}")

            self.logger.info(f"Setup completed:")
            self.logger.info(f"  Lines configured: {len(lines_config)}")
            self.logger.info(f"  Zones configured: {len(zones_config)}")
            self.logger.info(f"  Heatmap enabled: {self.config.enable_heatmap}")

            # Step 4: Initialize video processor
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

            # Export final results
            if results:
                self.logger.info("Exporting final results...")
                exporter = ResultsExporter(self.config.output_folder)
                exporter.export_final_summary(results)
                self.logger.info("Results exported successfully")

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

        # Handle config file loading
        if args.config:
            logger.info(f"Loading configuration from {args.config}")
            # TO DO: Implement config file loading in ConfigManager
            if not Path(args.config).exists():
                logger.error(f"Config file not found: {args.config}")
                return 1

        # Initialize application
        if not app.initialize():
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
