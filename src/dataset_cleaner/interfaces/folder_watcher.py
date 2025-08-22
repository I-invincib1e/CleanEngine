#!/usr/bin/env python3
"""
Folder Watcher for Automated Dataset Cleaning
Monitors a folder for new CSV/Excel files and automatically cleans them.
"""

import argparse
import os
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from dataset_cleaner import DatasetCleaner


class DatasetHandler(FileSystemEventHandler):
    def __init__(self, output_dir="cleaned_datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.cleaner = DatasetCleaner()

        # Supported file extensions
        self.supported_extensions = {".csv", ".xlsx", ".xls"}

    def on_created(self, event):
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Check if it's a supported file type
        if file_path.suffix.lower() in self.supported_extensions:
            print(f"🔍 New file detected: {file_path.name}")

            # Wait a moment to ensure file is fully written
            time.sleep(2)

            try:
                self.process_file(file_path)
            except Exception as e:
                print(f"❌ Error processing {file_path.name}: {str(e)}")

    def process_file(self, file_path):
        """Process a single file"""
        print(f"🚀 Starting automatic cleaning of: {file_path.name}")

        # Create organized output folder for this dataset
        dataset_name = file_path.stem
        dataset_output_folder = self.output_dir / f"Cleans-{dataset_name}"
        dataset_output_folder.mkdir(exist_ok=True)

        # Clean the dataset
        cleaned_df = self.cleaner.clean_dataset(str(file_path))

        # Save cleaned dataset in organized folder
        cleaned_filename = f"cleaned_{file_path.name}"
        output_file = dataset_output_folder / cleaned_filename

        if output_file.suffix == ".csv":
            cleaned_df.to_csv(output_file, index=False)
        else:
            cleaned_df.to_excel(output_file, index=False)

        # Generate reports in organized folder
        self.cleaner.generate_report(dataset_output_folder, dataset_name)

        # Perform advanced analysis
        try:
            print("🔬 Performing advanced analysis...")
            analysis_results = self.cleaner.perform_advanced_analysis(
                dataset_output_folder, dataset_name
            )
            if analysis_results:
                print("🎉 Advanced analysis completed!")
        except Exception as e:
            print(f"⚠️ Advanced analysis failed: {str(e)}")

        print(f"✅ Completed! Cleaned file: {output_file}")
        print(f"📂 All files saved in: {dataset_output_folder.absolute()}")
        print("-" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Watch folder for new datasets to clean"
    )
    parser.add_argument("watch_folder", help="Folder to watch for new files")
    parser.add_argument(
        "-o",
        "--output",
        default="cleaned_datasets",
        help="Output directory for cleaned files (default: cleaned_datasets)",
    )

    args = parser.parse_args()

    watch_path = Path(args.watch_folder)
    if not watch_path.exists():
        print(f"❌ Watch folder does not exist: {watch_path}")
        return 1

    print(f"👀 Watching folder: {watch_path.absolute()}")
    print(f"📁 Output directory: {Path(args.output).absolute()}")
    print("🔄 Waiting for new CSV/Excel files...")
    print("Press Ctrl+C to stop")

    # Set up file watcher
    event_handler = DatasetHandler(args.output)
    observer = Observer()
    observer.schedule(event_handler, str(watch_path), recursive=False)

    try:
        observer.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping folder watcher...")
        observer.stop()

    observer.join()
    print("👋 Folder watcher stopped")
    return 0


if __name__ == "__main__":
    exit(main())
