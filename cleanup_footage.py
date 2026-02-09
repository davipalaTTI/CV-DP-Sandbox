import shutil
import argparse
import time
from pathlib import Path
from datetime import datetime, timedelta

def cleanup(base_path, days_to_keep = 7):
    """ Deletes YYYY-MM-DD folders older than the specified number of days. """
    target_dir = Path(base_path)
    if not target_dir.exists():
        print(f"Directory {base_path} does not exist.")
        return
    cutoff_sec = time.time() - (days_to_keep * 86400)
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    print(f"Checking for files older than {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}")

    files_deleted = 0
    for file_path in target_dir.iterdir():
        if file_path.is_file():
            file_mtime = file_path.stat().st_mtime

            if file_mtime < cutoff_sec:
                try:
                    print(f"Deleting: {file_path.name} (Modified: {datetime.fromtimestamp(file_mtime)}")
                    file_path.unlink()
                    files_deleted += 1
                except Exception as e:
                    print(f"Error deleting {file_path.name}: {e}")
    print(f"Cleanup complete. Total Files deleted: {files_deleted}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleanup footage older than N days.")
    parser.add_argument("--path", required=True, help="Path to the footage folder (e.g. /path/to/footage)")
    parser.add_argument("--days", type=int, help="Number of days to keep (Default: 7).")

    args = parser.parse_args()
    cleanup(args.path, args.days)