import argparse
from datetime import datetime

__version__ = "1.0.1"

def _parse_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be an ISO local datetime such as 2026-08-05T18:30:00"
        ) from exc


def _parse_retention_days(value: str) -> int:
    try:
        days = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a whole number of days") from exc
    if days < 0:
        raise argparse.ArgumentTypeError("cannot be negative")
    return days


def parse_arguments(argv=None):
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
        "-c", "--config",
        type=str,
        help="Path to a specific config.json file to load"
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip interactive configuration and use saved counting geometry"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable runtime display windows (also skips interactive setup)"
    )
    parser.add_argument(
        "--source-name",
        type=str,
        help="Override the source name written to event exports"
    )
    footage_group = parser.add_mutually_exclusive_group()
    footage_group.add_argument(
        "--save-footage",
        dest="save_footage",
        action="store_true",
        help="Override the source config and save annotated footage",
    )
    footage_group.add_argument(
        "--no-save-footage",
        dest="save_footage",
        action="store_false",
        help="Override the source config and disable all footage recording",
    )
    parser.set_defaults(save_footage=None)
    parser.add_argument(
        "--footage-retention-days",
        type=_parse_retention_days,
        help="Override live-footage retention (0 keeps recordings indefinitely)",
    )
    parser.add_argument(
        "--stop-at",
        type=_parse_datetime,
        help="Stop gracefully at an ISO local datetime (used by the scheduler)"
    )
    parser.add_argument(
        "--window-index",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--window-count",
        type=int,
        default=1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "-l", "--log-file",
        type=str,
        help="Write logs to this file (default: logs/app.log)"
    )
    parser.add_argument(
        "--crash-report-dir",
        type=str,
        help="Directory for native/Python crash reports"
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

    return parser.parse_args(argv)
