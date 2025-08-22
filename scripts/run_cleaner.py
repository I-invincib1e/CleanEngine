#!/usr/bin/env python3
"""
Simple CLI runner for the Dataset Cleaner
Quick way to clean a dataset with default settings.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so 'src.dataset_cleaner' imports resolve
current_dir = Path(__file__).parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.dataset_cleaner.core.cleaner import DatasetCleaner


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_cleaner.py <input_file> [output_file]")
        print("Example: python run_cleaner.py data.csv cleaned_data.csv")
        return 1

    input_file = sys.argv[1]

    # Generate output filename if not provided
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        input_path = Path(input_file)
        output_file = f"cleaned_{input_path.name}"

    try:
        print("🧹 Starting dataset cleaning...")

        # Initialize and run cleaner
        cleaner = DatasetCleaner()

        # Create organized output folder
        input_path = Path(input_file)
        dataset_name = input_path.stem
        output_folder = cleaner.create_output_folder(input_file)

        print(f"📁 Created output folder: {output_folder}")

        # Clean dataset
        cleaned_df = cleaner.clean_dataset(input_file)

        # Determine output file path in organized folder
        if len(sys.argv) > 2:
            # User specified output filename
            output_filename = Path(output_file).name
        else:
            # Default naming
            output_filename = f"cleaned_{input_path.name}"

        output_file_path = output_folder / output_filename

        # Save cleaned dataset
        if output_file_path.suffix == ".csv":
            cleaned_df.to_csv(output_file_path, index=False)
        else:
            cleaned_df.to_excel(output_file_path, index=False)

        print(f"💾 Cleaned dataset saved: {output_file_path}")

        # Generate reports in organized folder
        cleaner.generate_report(output_folder, dataset_name)

        # Perform advanced analysis
        try:
            print("🔬 Performing advanced data analysis...")
            analysis_results = cleaner.perform_advanced_analysis(
                output_folder, dataset_name
            )
            if analysis_results:
                print(
                    "🎉 Advanced analysis completed with insights and visualizations!"
                )
        except Exception as e:
            print(f"⚠️ Advanced analysis failed: {str(e)}")
            print("📊 Basic cleaning reports are still available")

        print("🎉 Cleaning completed successfully!")
        print(f"📂 All files saved in: {output_folder.absolute()}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
