import shutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta

def cleanup(base_path, days):
    """ Deletes YYYY-MM-DD folders older than the specified number of days. """
    target_dir = Path(base_path)
    if not target_dir.exists():
        print(f"Directory {base_path} does not exist.")
        return

    cutOff = datetime.now() - timedelta(days=days)
    print(f"Checking for folders older than {cutOff.strftime('%Y-%m-%d')} days...")

    for folder in target_dir.iterdir():
        if folder.is_dir():
            try:
                folder_date = datetime.strptime(folder.name, "%Y-%m-%d")
                if folder_date < cutOff:
                    print(f"Deleting folder: {folder} (Date: {folder_date.strftime('%Y-%m-%d')})")
                    shutil.rmtree(folder)
            except ValueError:
                print(f"Skipping non-date folder: {folder}")