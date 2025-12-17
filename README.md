# Generating Stock Market Insights

This is my ML project where I explore various trends of the stock market. I decided to undertake this project to explore real stock market data in the US, and using my existing ML knowledge and skills to generate meaningful insights that would help me understand a bit more about stock price movements and trends. I also aim to expand my ML skills while working on this real world project using real world data, which is my key motivation.

I plan to explore different problems in this repository, and I will cover each problem as part of this document to provide the necessary details for each.

## Problem 1 - Time-series trend classification

### Problem Definition:   
This is a multi-class classification problem where I would like to find out whether 30-day OHLVC data on different stocks will allow me to train ML models that can categorize a given stock as one of the following 5 classes: ***(OHLVC is a standard format used to represent daily price and volume data  for any traded asset. It represents Opening price, Highest price, Lowest price, Closing price, and Volume (or number) of shares traded )***

**1. Trending up *(class name: TREND_UP)*:** This indicates a strong upward slope over the 30-day period without extreme volatility.

**2. Trending up *(class name: TREND_DOWN)*:** This indicates a strong downward slope over the 30-day period without extreme volatility.

**3. Stationary *(class name: STATIONARY)*:** This indicates there is neither a strong upward or downward slope over the 30-day period, and no volatility exists in that same time period.

**4. Oscillating *(class name: OSCILLATING)*:** This indicates there is too much upward and downward variation within the 30-day period. In oter words, the stock price flips direction too frequently.

**5. No distinct trend *(class name: OTHER)*:** Any trend that cannot be classified as one of the above 4 will be categorized as OTHER.

### Calculation of metrics:
In order to bucket each 30-day time-series frame into one of the above 5 classes, we will need to compute a few metrics on each sample. We will also need to define (or compute) a few thresholds as part of this process.

***1. Overall slope (trend direction):*** We will fit a straight line to each 30-day frame and calculate its slope (b). If b > 0, trend will be upward, and if b < 0, it will be downward. Note |b| can be used to indicate the "strength of the trend". Here we need to be careful with solely depending on |b| as an indicator if the trend in question, and we need to consider it with the noise in the data.

Noise in the 30-day data will be given by the standard deviation (std). Depending on the level of noise, a small |b| may or may not be significant in terms of a trend. Therefore, we will use the following strength value as alternative.

<div align="center">
  <img src="readme_images/trend_strength.png" alt="description" height="50px">
</div>

Here, the std is calculated using all samples in the 30-day frame.

The next important question is "how do we decide if a trend strength is significant or not?". In other words, we need to define a "threshold" for the trend strength somehow.

The simplest rule of thumb that we can use is if a trend strength is less than 0.5 it is considered weak, and strong otherwise.

A data-driven method of defining the trend strength threshold is to calculate it for each frame in the entire dataset and define the threshold at 60 percentile of the trend strength that exists in the dataset. Any value above the 60 percentile can be considered as high trend strength.