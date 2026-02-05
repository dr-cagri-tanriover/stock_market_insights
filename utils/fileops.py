
import json
import csv
from typing import Any, List, Union, Optional, Dict
from pathlib import Path


def yield_csv_rows(csv_filepath: Union[str, Path]):
    """
    Generator method that yields one row at a time from a CSV file.
    Reads directly from file without loading entire file into memory.
    
    Args:
        csv_filepath: Path to the CSV file (str or Path object)
        
    Yields:
        dict: One row from the CSV file per call as a dictionary with column names as keys
        
    Example:
        >>> for row in yield_csv_rows('data.csv'):
        ...     print(row)  # row is a dict like {'Date': '20240101', 'Open': 100, ...}
    """

    csv_path = Path(csv_filepath) if isinstance(csv_filepath, str) else csv_filepath
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Read CSV row by row using csv.DictReader (memory efficient)
    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            yield row


def read_json_content(source_folder: Path | str, json_filename: str) -> None:
    """
    Get the complete JSON metadata.
    
    Returns:
        Dictionary containing json data, or None.
    """

    raw_data_path = Path(source_folder)
    metadata_path = raw_data_path / json_filename
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        raise FileNotFoundError(f"JSON file {json_filename} not found in {source_folder}")
        metadata = None

    return metadata