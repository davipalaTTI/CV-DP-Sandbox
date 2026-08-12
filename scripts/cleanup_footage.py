import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.footage_retention import cleanup_footage_directory

def cleanup(base_path, days_to_keep = 7):
    """Delete files older than the specified number of days."""
    target_dir = Path(base_path)
    if not target_dir.exists():
        print(f"Directory {base_path} does not exist.")
        return
    result = cleanup_footage_directory(target_dir, days_to_keep)
    print(
        f"Cleanup complete. Deleted {result.deleted_files} file(s), "
        f"freed {result.freed_bytes / (1024 * 1024):.1f} MB, "
        f"errors: {result.errors}."
    )
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleanup footage older than N days.")
    parser.add_argument("--path", required=True, help="Path to the footage folder (e.g. /path/to/footage)")
    parser.add_argument("--days", type=int, default=7, help="Number of days to keep (default: 7).")

    args = parser.parse_args()
    cleanup(args.path, args.days)
