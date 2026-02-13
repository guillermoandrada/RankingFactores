#!/usr/bin/env python
"""
Entry point for importing financial data from Excel/CSV.
Usage: python run_import.py [filepath]
       python run_import.py          # uses default file from config
"""

import sys

from modules.config import DEFAULT_INPUT_FILE
from modules.ingestion import DataImporter


def main() -> None:
    filepath = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_FILE
    importer = DataImporter()

    try:
        importer.import_file(filepath)
    except Exception as e:
        print("\n--- ERROR ---")
        print(e)
        print("-------------")
        input("Press ENTER to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()
