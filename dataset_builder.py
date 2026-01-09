"""
Dataset builder module for creating structured stock market datasets.

This module provides a class to build datasets from stock ticker data,
organizing 30-day OHLCV frames into CSV files with proper naming conventions
and metadata.
"""

import os
import json
import shutil
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, List, Union, Optional, Dict
from pathlib import Path

from finance import StockDataFetcher


class DatalakeBuilder:
    """
    Class for building structured datasets from stock market data.
    
    Creates a dataset folder with CSV files containing 30-day OHLCV frames
    and a metadata JSON file describing the dataset.
    """
    
    def __init__(
        self,
        tickers: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        output_folder: str = "stock_datalake",
        use_adjusted: bool = True,
        delete_existing_data: bool = False
    ):
        """
        Initialize the DatalakeBuilder.
        
        Args:
            tickers: List of stock ticker symbols (e.g., ['AAPL', 'MSFT'])
            start_date: Start date for data collection (str 'YYYY-MM-DD' or datetime)
            end_date: End date for data collection (str 'YYYY-MM-DD' or datetime)
            output_folder: Name of the folder to save dataset files (default: 'stock_datalake')
            use_adjusted: If True, use adjusted prices (default: True)
        """
        self.tickers = list(set(tickers))  # removing duplicates
        self.start_date = start_date
        self.end_date = end_date
        self.output_folder = output_folder
        self.use_adjusted = use_adjusted
        
        self.data_summary = {}  # used for sanity check after creating the data lake

        # Convert dates to datetime if strings
        if isinstance(start_date, str):
            self.start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            self.end_date = pd.to_datetime(end_date)
        
        # Create output folder if it doesn't exist
        self.output_path = Path(output_folder)
        if self.output_path.exists():
            if delete_existing_data:
                print(f"Deleting existing data lake: {self.output_path}")
                shutil.rmtree(self.output_path)
                self.output_path.mkdir(parents=True, exist_ok=True)
            else:
                print(f"Overwriting data lake without deleting: {self.output_path}")
        else:
            print(f"Creating new data lake: {self.output_path}")
            self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initializing storage for metadata
        str_start_date = self.start_date.strftime('%Y%m%d')
        str_end_date = self.end_date.strftime('%Y%m%d')
        self.metadata = {
            'start_date': str_start_date,
            'end_date': str_end_date,
            'adjusted_prices': use_adjusted,
            'file_info': {f"{ticker}_{str_start_date}_{str_end_date}.csv": 0 for ticker in tickers}
        }
    
    def build_dataset(self) -> None:
        """
        Main method to build the complete dataset.
        
        This method orchestrates the dataset creation process:
        1. Fetches data for each ticker
        2. Extracts 30-day frames
        3. Saves frames as CSV files
        4. Creates metadata JSON file
        """
        
        for ticker in self.tickers:
            print(f"Processing ticker: {ticker}")
            df = self._process_ticker(ticker)
            #all_frames_info.extend(ticker_frames_info)
        
        # Sanity check: let's print the summary of the data we have just created.
        self._data_sanity_check()

        # Create metadata
        self._create_metadata()
        
        print(f"Dataset built successfully in folder: {self.output_folder}")
    
    def _process_ticker(self, ticker: str) -> pd.DataFrame:
        """
        Process a single ticker: fetch data, extract frames, and save CSV files.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of dictionaries containing frame information for metadata
        """
        # TODO: Implement ticker processing
        # 1. Create StockDataFetcher instance for this ticker
        # 2. Fetch data using fetch_data()
        # 3. Get all 30-day frames using get_all_30_day_frames()
        # 4. For each frame:
        #    - Calculate time_index (month number)
        #    - Get frame start date
        #    - Get OHLCV data for the frame
        #    - Save as CSV with proper naming: <ticker>_<time_index>_<start_date>.csv
        # 5. Return list of frame info dictionaries
        
        fetchObj = StockDataFetcher(
            ticker=ticker,
            start_date=self.start_date,
            end_date=self.end_date,
            use_adjusted=self.use_adjusted
        )

        df = fetchObj.fetch_data()

        days, _ = df.shape  # total days (i.e. indices) in df
        
        df.index = df.index.strftime('%Y%m%d')  # converting the datetime indices to string format for compatibility with the csv file name

        # Get the csv file name that was initialized for the current ticker.
        csv_filename = next(filter(lambda key: ticker in key, self.metadata['file_info']))
        # Update the total number of days that were fetched fir the current ticker
        self.metadata['file_info'][csv_filename] = days

        self._save_frame_to_csv(df, csv_filename)

        # Sanity check: let's print the summary of the data we have just created.
        self.data_summary[ticker] = {
            'total_days': days,
            'first_day': df.index[0],
            'last_day': df.index[-1]
        }
    
    def _save_frame_to_csv(
        self,
        frame_data: pd.DataFrame,
        filename: str
    ) -> None:
        """
        Save a  multi day OHLCV frame DataFrame to CSV file.
        
        Args:
            frame_data: DataFrame containing OHLCV data for the frame
            filename: Name of the CSV file to save
        """
        filepath = self.output_path / filename
        frame_data.to_csv(filepath, index=True) 


    def _data_sanity_check(self) -> None:
        """
        Print the summary of the data we have just created.
        """

        temp = []
        # Check to make sure total frames match for all tickers
        for _, summary in self.data_summary.items():
            temp.append(summary['total_days'])

        if len(set(temp)) > 1:
            raise ValueError("Total days do not match for all tickers.")
        else:
            #total_days = list(set(temp))[0]
            print(f"Total days match for all tickers={temp[0]}")

        temp = []
        # Check to make sure first days match for all tickers
        for _, summary in self.data_summary.items():
            temp.append(summary['first_day'])

        if len(set(temp)) > 1:
            raise ValueError("First days do not match for all tickers.")
        else:
            # Get first (and only) element from set - convert to list first
            #first_day = list(set(temp))[0]
            print(f"First days match for all tickers={temp[0]}")

        temp = []
        # Check to make sure last days match for all tickers
        for _, summary in self.data_summary.items():
            temp.append(summary['last_day'])

        if len(set(temp)) > 1:
            raise ValueError("Last days do not match for all tickers.")
        else:
            # Get first (and only) element from set - convert to list first
            #last_day = list(set(temp))[0]
            print(f"Last days match for all tickers={temp[0]}")


    def _create_metadata(self) -> None:
        """
        Create and save metadata JSON file.        
        """

        # Save metadata to JSON file
        metadata_path = self.output_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)
        
        print(f"Metadata saved to: {metadata_path}")
    
    def get_metadata(self) -> Optional[Dict]:
        """
        Get the dataset metadata.
        
        Returns:
            Dictionary containing metadata, or None if dataset hasn't been built yet
        """
        if self.metadata is None:
            metadata_path = self.output_path / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
        return self.metadata

