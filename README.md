# Generating Stock Market Insights

This is my ML project where I explore various trends of the stock market. I decided to undertake this project to explore real stock market data in the US, and to generate meaningful insights that would help me understand a bit more about stock price movements and trends by using my existing ML knowledge and skills. I also aim to expand my ML skills while working on this real world project using real world data, which is my key motivation.

I plan to explore different problems in this repository, and I will cover each problem as part of this document to provide the necessary details for each.

## Problem 1 - Time-series trend classification

### Problem Definition:   
This is a multi-class classification problem where I use 30-day OHLCV data on different stocks. ***(OHLVC is a standard format used to represent daily price and volume data  for any traded asset. It represents Opening price, Highest price, Lowest price, Closing price, and Volume (or number) of shares traded )***

My challenge is to find out whether the OHLCV data on different stocks will allow me to train ML models that can categorize a given stock as one of the following 5 classes: 

**1. Trending up *(class name: TREND_UP)*:** This indicates a strong upward slope over the 30-day period without extreme volatility.

**2. Trending up *(class name: TREND_DOWN)*:** This indicates a strong downward slope over the 30-day period without extreme volatility.

**3. Stationary *(class name: STATIONARY)*:** This indicates there is neither a strong upward or downward slope over the 30-day period, and no volatility exists in that same time period.

**4. Oscillating *(class name: OSCILLATING)*:** This indicates there is too much upward and downward variation within the 30-day period. In other words, the stock price flips direction too frequently.

**5. No distinct trend *(class name: OTHER)*:** Any trend that cannot be classified as one of the above 4 will be categorized as OTHER.

### First things first - What is meant by "price"?
While talking about my technical approach in the rest of this section, you will come across the word "price" often. I already mentioned the OHLCV data we will use has 4 different prices: Opening, Closing, (Daily) High and (Daily) Low. However, I did not mention which one of those 4 different prices I will be using while creating a solution to Problem 1, which I will try to clarify here.

There is no simple answer to which price you should use as it depends on what you would like to calculate and how detailed you would like your analysis to be. To keep things simple (especially for your initial design) you can simply choose one of Open, Close, High or Low prices listed for each day for a stock ticker, and use it as your reference for calculating all of the metrics (which will be explained in detail later) This is the simplest approach and does not require computation of additional features related to price.

Note a stock can go for an unexpected split that introduces price inconsistencies over time, and this can be fixed by using the "Adjusted" versions of the prices. Typically, most finance APIs provide a price and its adjusted version (e.g., Close and Adjusted Close as can be requested using yfinance python library)

Unless stated otherwise, in the rest of this section, "price" will refer to "Adjusted Close" price. If you are interested in exploring additional price metrics to experiment with, you can refer to my <a href="readme_images/price_definitions.pdf">price definitions</a> where I summarize different types of prices and their uses.


### Calculation of metrics:
After clarifying the "price" we will be using, we can now talk about the metrics that will be part of this solution. In order to bucket each 30-day time-series frame into one of the 5 classes I mentioned already, we will need to compute a few metrics to ensure ground truth information for each 30-day frame is generated accurately.

I define 4 independent metrics as follows.

***1. Slope of linear fit:*** We simply fit a straight line on the prices inside the 30-day frame for a given stock, and calculate the slope (b) of that line. We also need to calculate the magnitude of the slope |b|, and repeat this calculation for all 30-day frames in our dataset. *(the reason for this will become clearer while describing the thresholds in a later  section)*

***Note:*** *You may want to use log of prices rather than the raw stock prices while fitting a line for the following reasons:*
- Log prices make trend and volatility measures scale-invariant across different stocks.
- Log prices also allow comparison between different timelines (where absolute price difference in the same stock may be vastly different-e.g., price in 2005 vs. 2025)
- Log prices align naturally with log returns as well (as will be discussed shortly).
- Using logs give more stable statistical behavior for window-based classification.

***CAUTION 1:*** *When using log of prices, keep in mind ln(0) is undefined. Therefore, you need to pre-filter zero dollar prices prior to applying log to them !!*

***CAUTION 2:*** *When using log of prices, slope b will measure the relative drift rather than the absolute price drift of the stock.*

***CAUTION 3:*** *If you calculate b based on log prices, it will approximately be equal to the (fixed) percentage change in price over time (especially true for small b values). This is regardless of the stock price, which allows direct comparison between different stocks in our solution.*

*Using raw prices will make sense in the case of the following:*
- When analyzing one stock ticker only
- Price level itself is meaningful
- Interpretability  in absolute ($) units matters
- You want shapes defined by dollar movement, not relative movement


***2. Trend strength ratio:*** Trend strength ratio indicates whether the upward or the downward trend of the 30-day frame is a strong or a weak one. This metric is calculated as shown below where the numerator is the slope magnitude of the line of best fit and the denominator is the standard deviation of all the samples in the 30-day frame. 

<div align="center">
  <img src="readme_images/trend_strength.png" alt="trend strength" height="50px">
</div>

***Important:*** *If you calculate the |b| based on the log of prices in the 30-day frame rather than the raw prices, the standard deviation also has to be calculated using the log of prices in that same 30-day frame for consistency!*


***3. Volatility from returns:*** Volatility is defined as the standard deviation of returns, and it measures how much the signal fluctuates relative to its average behavior. We first convert raw prices to "returns". Working with returns has the following advantages:  
- It removes scale (e.g., a $10 stock can be compared to a $500 one)
- It stabilizes variance
- It captures relative movement instead of absolute price level

For each 30-day frame, we can calculate 29 simple return values as follows:
<div align="center">
  <img src="readme_images/simple_return.png" alt="simple return" height="50px">
</div>

We then calculate the mean return value and calculate the volatility (v), which is simply the standard deviation of returns within the 30-day frame as follows (note N=29 in a 30-day frame): 

<div align="center">
  <img src="readme_images/volatility.png" alt="volatility" height="50px">
</div>

***Note:*** *Using log returns instead of simple returns as described above may work better statistically. Following is how you calculate a log return:*

<div align="center">
  <img src="readme_images/log_return.png" alt="log return" height="50px">
</div>

*There are three key reasons why log returns will make sense:*

***Reason 1 -*** *Log returns preserve total movement across a window exactly. For example, if a price moves +10% one day and -10% the next day, for a $100 stock, simple returns gives,* 

Total return: +0.10 - 0.10 = 0

*However, the price movement is not flat:*

100 -> 110 -> 99

*If we look at the same change using log returns,*

Total return: ln(110/100) + ln(99/110) = ln(99/100)

*which means we can use them "additively".*

***Reason 2 -*** *Simple returns are NOT symmetrical while log returns are - due to the mathematical property of logarithms: ln(a) = -ln(1/a)*

*Consider a price that doubles first and then halves as follows:*

100 -> 200 : *in simple returns this yields %100*  
200 -> 100 : *in simple returns this yields -%50*

*Even though the final price is still where it first started, the positive and negative returns are numerically NOT symmetrical!*

*Now in the case of the same example, when log returns are used instead:*

100 -> 200 : *in log returns this yields +0.693*  
200 -> 100 : *in log returns this yields -0.693*

*which represents a symmetric change!*

*This is important because:*   
- *Positive and negative moves contribute equally*
- *Variance reflects true fluctuation magnitude*
- *Distribution is closer to symmetric and stable*

***Reason 3 -*** *If your dataset includes stock tickers with very different price levels, using log returns will allow you to compare them.*

***Key takeaway:** Volatility becomes a cleaner measure of "movement intensity" when log returns are used instead of simple returns!*

***4. Zero-crossing rate (ZCR):*** The goal here is to quantify how often the 30-day series price changes direction (i.e., up -> down or down -> up). ZCR is a strong indicator of an oscillating pattern.

Though the name suggests there is a negative to positive transition around zero, what is meant here is a "sign change" of how the price of a stock is moving. First, we calculate the price difference between two consecutive days (dt). If dt < 0, the price has decreased while dt > 0 indicates a price increase. In a 30-day frame, we can have 29 price differences at most.

The "sign change" refers to whether a positive or negative difference between successive differences is maintained over time. For example,

daily prices:           x1, x2, x3, x4, x5, x6
price differences:      x2-x1, x3-x2, x4-x3, x5-x4, x6-x5
sign of differences:    s1 = sign(x2-x1), s2 = sign(x3-x2), s3 = sign(x4-x3), s4=sign(x5-x4), s5=sign(x6-x5)
sign changes        :   s1 ?= s2, s2 ?= s3, s3 ?= s4, s4 ?= s5

Therefore, for 'n' prices, we will calculate 'n-1' differences, and have a maximum of 'n-2' sign changes.

We count the number of sign changes (i.e., zero crossing count) as follows:
<div align="center">
  <img src="readme_images/sign_change_count.png" alt="zero crossing count" height="50px">
</div>

Then we calculate the rate (i.e., ZCR) as follows:
<div align="center">
  <img src="readme_images/zero_crossing_rate.png" alt="zero crossing rate" height="50px">
</div>

z=0 means the price trend never changes direction while z=1 means the price trend direction changes all the time (highly oscillatory)

***Note:*** *If price does not change from one day to the next, the sign() will indicate 0. As a result, calculating a sign difference will not be possible. For such cases, ignoring the transitions involving 0 will make sense. This will be equivalent to dropping day transitions where proce difference is zero.*

### Calculation of thresholds:
Now that we covered the metrics that will be used in defining our ground truth for the classification task, next we need to define (or compute) a few thresholds for those metrics to decide on how to classify each 30-day frame in our dataset.

***1. Trend direction threshold:*** Assuming we have calculated signed slopes (i.e., all b values) for all 30-day frames in our dataset, we can define the following two thresholds:

b(up) = 75th percentile of b in our dataset *(to check strong upward trend)*
b(down) = 25th percentile of b in our dataset *(to check strong downward trend)*

***2. Near-zero slope threshold:*** Assuming we have calculated slope magnitudes (i.e., all |b| values) for all 30-day frames in our dataset, we can define the following threshold:

b(0) = 25th percentile of |b| *(to check the flattest windows, i.e., the stationary candidates)*

***3. Trend strength threshold:*** Assuming we have calculated all trend strength metrics for all 30-day frames in our dataset, we can define the following threshold:

ts(min) = 60th percentile of trend strength *(to separate the strong and weak trend in any frame)*

***4. Oscillation threshold:*** Assuming we have calculated all zero crossing ratios (ZCRs) for all 30-day frames in our dataset, we can define the following threshold:

z(osc) = 75th percentile of ZCRs in our dataset *(to check strong oscillation pattern in a given frame)*

***5. Volatility threshold:*** Assuming we have calculated all volatility metrics for all 30-day frames in our dataset, we can define the two thresholds:

v(low) = 25th percentile of volatility metrics in the dataset. *(to confirm volatility is low)*

v(high) = 75th percentile of volatility metrics in the dataset. *(to confirm volatility is high - useful to ignore seemingly strong trends in the frame!)*

### Labeling rules:
We now have all of our definitions and criteria to create our labeling rules for generating ground truth data for all of our dataset.

We need to apply the following 5 rules in the listed order to successfully create all of the labels for our dataset.

> ***Rule 1*** - STATIONARY  class
> A frame is classified as STATIONARY if it is both flat and quiet.  
> *Criteria:*  
> - |b| <= b(0)  i.e., flat/near flat characteristic  
> - AND v <= v(low)  i.e., low volatility  
> - AND z < z(osc)  i.e., weak/no oscillations  

> ***Rule 2*** - OSCILLATING class
> A frame is classified as OSCILLATING if it flips direction frequently and does not have a strong trend.  
> *Criteria:*    
> - z >= z(osc)  i.e., strong oscillations  
> - AND ts < ts(min) i.e., no strong trend  

> ***Rule 3*** - TREND_UP class
> A frame is classified as TREND_UP if it has a positive slope and trend dominates noise.  
> *Criteria:*    
> - b >= b(up)  i.e., strong positive slope  
> - AND ts > ts(min) i.e., strong trend
> - AND z <= z(osc) i.e., guard against oscillatory frames  
> - AND v <= v(high)  i.e., guard against very noisy "trends"   

> ***Rule 4*** - TREND_DOWN class
> A frame is classified as TREND_DOWN if it has a negative slope and trend dominates noise.  
> *Criteria:*    
> - b <= b(down)  i.e., strong negative slope  
> - AND ts > ts(min) i.e., strong trend
> - AND z <= z(osc) i.e., guard against oscillatory frames  
> - AND v <= v(high)  i.e., guard against very noisy "trends"   

> ***Rule 5*** - OTHER class
> If a frame does not satisfy any of the above 4 rules, it is classified as OTHER. This class should collect the following cases:  
> - moderate slopes *(i.e., not extreme enough for trends)*
> - moderate volatility with low/medium zero crossings
> - mixed behavior *(i.e., trend + oscillations)*
> - transitional regimes  

