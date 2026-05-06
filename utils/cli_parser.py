import argparse

__version__ = "1.0.1"

def parse_arguments():
    """Parse command line arguments"""

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

    # Core Application Arguments
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a specific config.json file to load"
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip the interactive GUI setup and run headlessly"
    )

    # Development & Profiling Arguments (These were the missing ones!)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run the application with CPU performance profiling enabled"
    )
    parser.add_argument(
        "--memory-profile",
        action="store_true",
        help="Run the application with memory profiling enabled"
    )

    return parser.parse_args()