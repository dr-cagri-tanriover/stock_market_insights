import json
import csv
import random
import shutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, List, Union, Optional, Dict
from pathlib import Path

class FeatureDatasetBuilder:
    def __init__(self, raw_data_folder: str, metadata_file: str, feature_dataset_filename: str, days_per_feature: int):
        self.raw_data_folder = raw_data_folder
        self.metadata_file = metadata_file
        self.feature_dataset_filename = feature_dataset_filename
        self.metadata = None
        self.days_per_feature = days_per_feature

        # Create a subfolder to store all feature dataset related information
        feature_stem = feature_dataset_filename.split('.')[0]  # filename without the extension
        self.feature_metadata_filename = f"{feature_stem}_metadata.json"
        self.feature_dataset_folder_path = Path(feature_stem)
        
        # Remove folder if it exists and create a fresh one
        if self.feature_dataset_folder_path.exists():
            shutil.rmtree(self.feature_dataset_folder_path)  # remove the folder if it exists
        self.feature_dataset_folder_path.mkdir(parents=True, exist_ok=True)  # raise no error if folder exists.


    def get_metadata(self) -> None:
        """
        Get the dataset metadata.
        
        Returns:
            Dictionary containing metadata, or None if dataset hasn't been built yet
        """
        if self.metadata is None:
            raw_data_path = Path(self.raw_data_folder)
            metadata_path = raw_data_path / self.metadata_file
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            else:
                raise FileNotFoundError(f"Metadata file {self.metadata_file} not found in {self.raw_data_folder}")


    def yield_csv_rows(self, csv_filepath: Union[str, Path]):
        """
        Generator method that yields one row at a time from a CSV file.
        Reads directly from file without loading entire file into memory.
        
        Args:
            csv_filepath: Path to the CSV file (str or Path object)
            
        Yields:
            dict: One row from the CSV file per call as a dictionary with column names as keys
            
        Example:
            >>> builder = FeatureDatasetBuilder(...)
            >>> for row in builder.yield_csv_rows('data.csv'):
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
    
    def build_feature_dataset(self):
        self.get_metadata()  # get the details of the raw dataset we will be using to build the feature dataset
       
        self.metadata['features_info'] = {'days_per_feature': self.days_per_feature}  # new information related to the features will be added to this dictionary

        raw_data_path = Path(self.raw_data_folder)
        
        # Open the feature dataset file for writing (headers only)
        with open(self.feature_dataset_folder_path / self.feature_dataset_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ticker', 'end_date', 'slope', 'zcr', 'volatility', 'trend_strength', 'gt'])  # column names in csv

        for each_ticker_file in self.metadata['file_info']:
            # Starting to process a stock ticker in the raw dataset

            current_ticker = each_ticker_file.split('_')[0]  # stock ticker retrieved from csv file name
            print(f"Processing {current_ticker}...")
            
            # Construct full path to the CSV file
            csv_filepath = raw_data_path / each_ticker_file
            
            days_counter = 0
            samples_counter = 0

            # Number of samples that can be generated from the raw data for this ticker
            expected_number_of_samples = int(self.metadata['file_info'][each_ticker_file]) - self.days_per_feature + 1

            if expected_number_of_samples <= 0:
                raise ValueError(f"Not enough data to generate {self.days_per_feature}-day samples for {current_ticker}")
            
            
            # self.days_per_feature x 2 empty data frame creation for date and price columns
            sample_group = pd.DataFrame({
                "date": pd.Series([""] * self.days_per_feature, dtype=str),
                "price": pd.Series([np.nan] * self.days_per_feature, dtype=float)
                }
                )

            rows_to_fill = self.days_per_feature
            for each_row in self.yield_csv_rows(csv_filepath):
                # Process each row here
                days_counter += 1

                # Shift rows in sample_group as needed
                sample_group = sample_group.shift(-1)  # shift rows upward by one row

                # Update most recent row in sample_group with the current row data read from file
                sample_group.iloc[-1] = [each_row['Date'], float(each_row['Close'])]
                rows_to_fill -= 1  # there is one less row to fill now

                if rows_to_fill == 0:
                    # Sliding window now has a full set of data, so we can compute features
                    # Call feature generation function to compute features for the current sample_group and update features_line with the computed features
                    # Each sample will have the following information:
                    #features_line = {'ticker': None, 'end_date': None, 'slope': None, 'zcr': None, 'volatility': None, 'trend_strength': None, 'gt': None}
                    features_line = {
                                        'ticker': current_ticker,  # already know the stock ticker being processed
                                        'end_date': sample_group.iloc[-1]['date'] # already known from samples_group
                    }

                    computed_features = self.compute_features(sample_group)
                    
                    features_line.update(computed_features)  # extended the features_line to include the computed features as well

                    # Write the new feature_line to the feature dataset file
                    # Open the feature dataset file for APPENDING
                    with open(self.feature_dataset_folder_path / self.feature_dataset_filename, 'a', newline='') as f:
                        writer = csv.writer(f)
                        # Write values from features_line dictionary in the same order as column headers
                        writer.writerow([
                            features_line['ticker'],
                            features_line['end_date'],
                            features_line['slope'],
                            features_line['zcr'],
                            features_line['volatility'],
                            features_line['trend_strength'],
                            features_line['gt']
                        ])


                    rows_to_fill = 1  # reset rows_to_fill to 1 for the next sample group (using the sliding window approach)
                    samples_counter += 1  # generated a new sample with multiple features.       

                #else: #continue reading rows and filling the sample_group with the next row data from the file

            # Check the metadata row count to ensure we have processed the correct number of rows
            if days_counter != self.metadata['file_info'][each_ticker_file]:
                raise ValueError(f"Processed {days_counter} rows for {current_ticker} but expected {self.metadata['file_info'][each_ticker_file]} rows")

            if samples_counter != expected_number_of_samples:
                raise ValueError(f"Generated {samples_counter} samples for {current_ticker} but expected {self.metadata['features_info'][current_ticker]['total_samples']} samples")

            # metadata update for the current stock ticker
            self.metadata['features_info'][current_ticker] = {'total_samples': samples_counter}  # metadata log

        # Full raw dataset is now processed
        # Write the updated metadata to the feature metadata file for future reference and debugging
        with open(self.feature_dataset_folder_path / self.feature_metadata_filename, 'w') as f:
            json.dump(self.metadata, f, indent=4)


    def compute_features(self, sample_group: pd.DataFrame) -> Dict:
        """
        Compute features for the given sample_group.
        
        Args:
            sample_group: DataFrame containing the sample data for prices on multiple days
            
        Returns:
            Dictionary containing the followingcomputed features:
            'slope': slope of the linear fit to the prices in the sample_group
            'zcr': zero crossing rate of the prices in the sample_group
            'volatility': volatility of the prices in the sample_group
            'trend_strength': trend strength of the prices in the sample_group
            'gt': ground truth label for the sample_group
        """

        # Compute the slope of the linear fit to the prices in the sample_group
        slope = get_slope_of_log_prices_line(sample_group)
        zcr = get_zcr_of_prices(sample_group)
        volatility = get_volatility_of_returns(sample_group)
        trend_strength = abs(slope) / volatility

        features = {'slope': slope, 'zcr': zcr, 'volatility': volatility, 'trend_strength': trend_strength}
        gt = get_ground_truth_label(features)

        return {
            'slope': slope,
            'zcr': zcr,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'gt': gt
        }


# Functions to compute each feature listed above

def get_slope_of_log_prices_line(sample_group: pd.DataFrame) -> float:
    """
    Compute the slope of the linear fit (line of best fit) to the log of the prices in the sample_group.
    Any $0 price is replaced with a very small value (1e-10) to avoid log(0) error.
    
    Args:
        sample_group: DataFrame containing the sample data for prices on multiple days (given by the number of rows in the dataframe)
        
    Returns:
        Slope of the linear fit (line of best fit) to the log of the prices in the sample_group
    """

    # Replace zero prices with very small value to avoid log(0) error
    prices = sample_group['price'].replace(0, 1e-10)
    
    # Take log of prices
    log_prices = np.log(prices.values)  # accessing the values in the Series object to convert to numpy array
    
    # Create x values (time indices: 0, 1, 2, ..., n-1)
    x = np.arange(len(log_prices))
    
    # Fit line of best fit: y = a + bx using least squares
    # Using numpy polyfit (degree 1 for linear fit)
    # Returns [slope, intercept]
    coefficients = np.polyfit(x, log_prices, deg=1)
    slope = coefficients[0]  # First coefficient is the slope
    
    return float(slope)


def get_zcr_of_prices(sample_group: pd.DataFrame) -> float:
    """
    Compute the zero crossing rate of the price changes over a time window given by the number of rows in the sample_group.
    
    Args:
        sample_group: DataFrame containing the sample data for prices on multiple days
        
    Returns:
        Zero crossing rate of the price changes in the sample_group
    """

    difference_in_prices = np.diff(sample_group['price'].values)  # computing the difference in prices between consecutive days
    sign_of_differences = np.sign(difference_in_prices)  # computing the sign of the difference in prices (outputs 1, 0 or -1)

    # We need to handle 0 occurences in sign_of_differences before counting the zero crossings
    # Find index of first non-zero element
    non_zero_indices = np.nonzero(sign_of_differences)[0]
    if len(non_zero_indices) > 0:
        first_non_zero_index = non_zero_indices[0]
        if first_non_zero_index > 0:
            # need to pad the the signs moving towards index 0 with the sign of the first non-zero element
            sign_of_differences[:first_non_zero_index] = sign_of_differences[first_non_zero_index]
        #else: no need to pad the signs moving towards index 0 with the sign of the first non-zero element
    else:
        # All elements are zero - no price changes
        first_non_zero_index = None
        # simply pad all sign elements to a non-zero value (either 1 or -1)
        sign_of_differences[:] = 1

    # At this point, there may still be zero entries in sign_of_differences. We need to remove them.
    # We simply sustain the previous non-zero sign for the zero entries.
    sign_of_differences = np.array([sign_of_differences[i-1] if sign_of_differences[i] == 0 else sign_of_differences[i] for i in range(len(sign_of_differences))])

    # At this point, there should be no zero entries in sign_of_differences. We can begin counting the zero crossings now.
    total_sign_changes = int(np.sum(sign_of_differences[1:] != sign_of_differences[:-1]))

    zcr = total_sign_changes / (len(sign_of_differences) - 1)  # total number of sign changes divided by the total number of signs available.)
    
    return zcr


def get_volatility_of_returns(sample_group: pd.DataFrame) -> float:
    """
    Compute the volatility of the returns in the sample_group. Daily returns as calculated as logs.

    Args:
        sample_group: DataFrame containing the sample data for prices on multiple days
        
    Returns:
        Volatility of the returns in the sample_group
    """

    log_returns = np.log(sample_group['price'].values[1:] / sample_group['price'].values[:-1])  # ln(price(t)/price(t-1))

    daily_volatility = np.std(log_returns, ddof=1)  # Use n-1 in denominator (ddof=1 for sample std)

    return float(daily_volatility)

def get_ground_truth_label(features: Dict) -> str:
    f"""
    Compute the ground truth label for the given features.
    features.keys()=['slope', 'zcr', 'volatility', 'trend_strength']
    
    Args:
        features: Dictionary containing the features
        
    Returns:
        Ground truth label for the given features
    """

    gt_label = 'OTHER'  # by default
    
    if abs(features['slope']) < 0.001 and features['zcr'] < 0.46 and features['volatility'] <= 0.008:
        gt_label = 'STATIONARY'
    elif abs(features['trend_strength']) < 0.36 and features['zcr'] >= 0.46 and features['volatility'] > 0.008:
        gt_label = 'OSCILLATING'
    elif features['slope'] >= 0.003 and features['trend_strength'] >= 0.36 and features['zcr'] < 0.46 and features['volatility'] < 0.02:
        gt_label = 'TREND_UP'
    elif features['slope'] <= -0.003 and features['trend_strength'] >= 0.36 and features['zcr'] < 0.46 and features['volatility'] < 0.02:
        gt_label = 'TREND_DOWN'
    else:
        gt_label = 'OTHER'
    
    return gt_label
