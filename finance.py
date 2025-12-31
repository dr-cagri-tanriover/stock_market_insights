"""
Finance API module for handling stock market data interactions using yfinance.

This module provides an object-oriented framework for fetching and processing
stock market data, particularly for time-series analysis and trend classification.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Union, Dict
import warnings


class StockDataFetcher:
    """
    Main class for fetching and managing stock market data from yfinance API.
    
    This class handles API interactions, data retrieval, and provides methods
    for accessing OHLCV data, adjusted prices, and time-series frames.

    NOTE: Multilevel columns ARE NOT supported.
    """
    
    def __init__(
        self,
        ticker: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        period: Optional[str] = None,
        use_adjusted: bool = True
    ):
        """
        Initialize the StockDataFetcher with ticker(s) and date parameters.
        
        Args:
            ticker: Single ticker symbol (str) or list of ticker symbols
            start_date: Start date for data retrieval (str 'YYYY-MM-DD' or datetime)
            end_date: End date for data retrieval (str 'YYYY-MM-DD' or datetime)
            period: Alternative to start/end dates. Options: '1d', '5d', '1mo', '3mo',
                   '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
            use_adjusted: If True, use adjusted prices (default: True)
        
        Note:
            Either (start_date, end_date) or period should be provided, not both.
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.period = period
        self.use_adjusted = use_adjusted
        self._data: Optional[pd.DataFrame] = None
        self._ticker_objects: Dict[str, yf.Ticker] = {}
        
        # Validate input
        if start_date and end_date and period:
            raise ValueError("Cannot specify both date range and period. Use one or the other.")
        if not (start_date and end_date) and not period:
            raise ValueError("Must specify either (start_date, end_date) or period.")
    
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch stock data from yfinance API.
        
        Returns:
            DataFrame with OHLCV data. If use_adjusted = True (default), columns: Close, High, Low, Open, Volume
            with each price automatically adjusted. 
            
            and optionally if use_adjusted=False, columns: Adj Close,Close, High, Low, Open, Volume where all prices
            are UNadjusted.
            
        Raises:
            ValueError: If data fetching fails or returns empty data
        """
        try:
            # Download data
            print("Data download in progress...")
            if self.period:
                data = yf.download(
                    tickers=self.ticker,
                    period=self.period,
                    auto_adjust=self.use_adjusted,
                    prepost=False,
                    threads=True,
                    multi_level_index=False  # multi index not supported to be able to access the column names correctly
                )
            else:
                data = yf.download(
                    tickers=self.ticker,
                    start=self.start_date,
                    end=self.end_date,
                    auto_adjust=self.use_adjusted,
                    prepost=False,
                    threads=True,
                    multi_level_index=False  # multi index not supported to be able to access the column names correctly

                )
            
            # Handle multi-index columns (when multiple tickers)
            if isinstance(data.columns, pd.MultiIndex):
                raise ValueError(f"Multi index columns are not supported. Please use a single ticker.")
            
            if data.empty:
                raise ValueError(f"No data retrieved for ticker(s): {self.ticker}")
            
            print("Data download completed")
            self._data = data
            return data
            
        except Exception as e:
            raise ValueError(f"Error fetching data: {str(e)}")
    
    def get_ohlcv_data(
        self,
        use_adjusted: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Get OHLCV (Open, High, Low, Close, Volume) data.
        
        Args:
            use_adjusted: Override instance use_adjusted setting
            
        Returns:
            DataFrame with OHLCV columns
        """
        if self._data is None:
            self.fetch_data()
        
        use_adj = use_adjusted if use_adjusted is not None else self.use_adjusted
        data = self._data.copy()
        
        # Handles single ticker only
        if use_adj and 'Adj Close' in data.columns:
            # Replace Close with Adj Close
            data['Close'] = data['Adj Close']
        return data[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    def get_price_series(
        self,
        price_type: str = 'Adj Close',
    ) -> pd.Series:
        """
        Get a specific price series (Open, High, Low, Close, Adj Close).
        
        Args:
            price_type: Type of price ('Open', 'High', 'Low', 'Close', 'Adj Close')
            
        Returns:
            Series with price data indexed by date
        """
        if self._data is None:
            self.fetch_data()
        
        data = self._data.copy()
        
        if price_type not in data.columns:
            raise ValueError(f"Price type {price_type} not found in data columns: {data.columns.tolist()}")

        return data[price_type]

    
    def get_30_day_frame(
        self,
        start_date: Union[str, datetime],
        price_type: str = 'Adj Close'
    ) -> pd.Series: 
        """
        Extract a 30-day time-series frame starting from a specific date.
        
        Args:
            start_date: Start date for the 30-day frame
            price_type: Type of price to extract ('Open', 'High', 'Low', 'Close', 'Adj Close')
            
        Returns:
            Series with 30 days of price data
        """
        if self._data is None:
            self.fetch_data()
        
        # Convert start_date to datetime if string
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        
        # Get price series
        price_series = self.get_price_series(price_type=price_type)
        
        # Calculate end date (30 days after start)
        end_date = start_date + timedelta(days=30)
        
        # Filter data for the 30-day window
        frame = price_series[(price_series.index >= start_date) & 
                            (price_series.index <= end_date)]
        
        if len(frame) == 0:
            raise ValueError(f"No data available for 30-day frame starting {start_date}")
        
        return frame
    
    def get_all_30_day_frames(
        self,
        price_type: str = 'Adj Close',
        overlap: bool = False
    ) -> List[pd.Series]:
        """
        Extract all possible 30-day frames from the dataset.
        
        Args:
            price_type: Type of price to extract
            overlap: If True, create overlapping frames (every day). 
                    If False, create non-overlapping frames (every 30 days)
            
        Returns:
            List of Series, each containing a 30-day frame
        """
        if self._data is None:
            self.fetch_data()
        
        price_series = self.get_price_series(price_type=price_type)
        
        if len(price_series) < 30:
            raise ValueError("Insufficient data: need at least 30 days of data")
        
        frames = []
        step = 1 if overlap else 30
        
        for i in range(0, len(price_series) - 29, step):
            frame = price_series.iloc[i:i+30]
            if len(frame) == 30:  # Only include complete 30-day frames
                frames.append(frame)
        
        return frames
    
    def get_returns(
        self,
        price_type: str = 'Adj Close',
        log_returns: bool = False
    ) -> pd.Series:
        """
        Calculate returns from price series.
        
        Args:
            ticker: Specific ticker (required if multiple tickers)
            price_type: Type of price to use for returns calculation
            log_returns: If True, calculate log returns. If False, calculate simple returns
            
        Returns:
            Series with returns data
        """
        price_series = self.get_price_series(price_type=price_type)
        
        # Filter out zero prices before calculating returns
        price_series = price_series[price_series > 0]
        
        if log_returns:
            # Log returns: ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})
            log_prices = np.log(price_series)
            returns = log_prices.diff()
        else:
            # Simple returns: (P_t - P_{t-1}) / P_{t-1}
            returns = price_series.pct_change()
        
        return returns.dropna()
    
    def get_ticker_info(self, ticker: str) -> Dict:
        """
        Get additional information about a ticker (company name, sector, etc.).
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Dictionary with ticker information
        """
        if ticker not in self._ticker_objects:
            self._ticker_objects[ticker] = yf.Ticker(ticker)
        
        ticker_obj = self._ticker_objects[ticker]
        info = ticker_obj.info
        
        return {
            'symbol': ticker,
            'longName': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'marketCap': info.get('marketCap', 'N/A'),
            'currency': info.get('currency', 'N/A')
        }
    
    @property
    def data(self) -> pd.DataFrame:
        """Get the raw data DataFrame."""
        if self._data is None:
            self.fetch_data()
        return self._data
    
    @property
    def tickers(self) -> List[str]:
        """Get list of ticker symbols."""
        return self.ticker


class StockDataProcessor:
    """
    Utility class for processing and analyzing stock data.
    
    This class provides methods for calculating metrics like slopes, volatility,
    and other statistical measures needed for trend classification.
    """
    
    @staticmethod
    def calculate_slope(price_series: pd.Series, use_log: bool = False) -> float:
        """
        Calculate the slope of a linear fit to price data.
        
        Args:
            price_series: Series of prices
            use_log: If True, fit line to log of prices
            
        Returns:
            Slope value (b)
        """
        if len(price_series) < 2:
            raise ValueError("Need at least 2 data points to calculate slope")
        
        # Filter out zero prices if using log
        if use_log:
            price_series = price_series[price_series > 0]
            if len(price_series) < 2:
                raise ValueError("Insufficient non-zero prices for log calculation")
            y = np.log(price_series.values)
        else:
            y = price_series.values
        
        # Create x values (days)
        x = np.arange(len(y))
        
        # Fit linear regression: y = a + b*x
        # Using least squares: b = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - (sum(x))^2)
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        
        return slope
    
    @staticmethod
    def calculate_volatility(returns: pd.Series) -> float:
        """
        Calculate volatility as standard deviation of returns.
        
        Args:
            returns: Series of returns
            
        Returns:
            Volatility (standard deviation)
        """
        return returns.std()
    
    @staticmethod
    def calculate_trend_strength(
        price_series: pd.Series,
        use_log: bool = False
    ) -> float:
        """
        Calculate trend strength as |slope| / std.
        
        Args:
            price_series: Series of prices
            use_log: If True, use log prices for both slope and std
            
        Returns:
            Trend strength value
        """
        slope = StockDataProcessor.calculate_slope(price_series, use_log=use_log)
        
        if use_log:
            price_series = price_series[price_series > 0]
            std = np.log(price_series).std()
        else:
            std = price_series.std()
        
        if std == 0:
            return float('inf') if abs(slope) > 0 else 0.0
        
        return abs(slope) / std


###########################################################################################
###########################################################################################
if __name__ == "__main__":
    print("Finance module sandbox started")
    print("Creating StockDataFetcher object")
    sdf = StockDataFetcher(ticker="AAPL",start_date="2025-12-01",end_date="2025-12-21")
    print("Fetching data")
    data = sdf.fetch_data()
    print("Data fetched")
    print(data.head())
    print("Finance module sandbox ended")