
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler

import data_insights as di
import utils.plot_sessions as ps


def run_model_dev_eval(analysis_attributes: dict):
    print(f"Model development and evaluation in progress....")
    print(f"Model development and evaluation report will be saved to {analysis_attributes['reportout_filepath']}")
    print(f"Model development and evaluation report title: {analysis_attributes['pdf_report_title']}")
    print(f"Model development and evaluation report will be saved to {analysis_attributes['reportout_filepath']}")

    dsObj = di.DataInsights(
    dataset_filepath=Path(analysis_attributes['dataset_folderpath'], analysis_attributes['feature_dataset_filename']),
    reportout_filepath=analysis_attributes['reportout_filepath'],
    pdf_report_title=analysis_attributes['pdf_report_title']
    )

    #dsObj.df will have the feature dataset at this point
    print(f"Feature dataset loaded successfully: {len(dsObj.df)} rows, {len(dsObj.df.columns)} columns")

    """
    Isolation Forest Model Development and Evaluation:
    The "Normal" group will be rows with "OSCILLATING", "OTHER" AND "STATIONARY" class labels.
    The "Anomaly" group will be rows with "TREND_UP" AND "TREND_DOWN" class labels.
    """

    train_window = {'lo': 2013, 'hi': 2022}  # Training samples will be picked for these year range in the dataset (inclusive)
    test_window = {'lo': 2023, 'hi': 2025}  # Testing samples will be picked for these year range in the dataset (inclusive)

    # Following are the column names and their categories in this dataset
    column_defs = {'features': ['slope', 'zcr', 'trend_strength', 'volatility'],  # features to train on
                    'target': 'gt',  # aka ground truth
                    'meta': ['end_date', 'ticker']
                }

    normal_classes = ['OSCILLATING', 'OTHER', 'STATIONARY']
    anomaly_classes = ['TREND_UP', 'TREND_DOWN']
    normal_df = dsObj.df[dsObj.df['gt'].isin(normal_classes)]
    anomaly_df = dsObj.df[dsObj.df['gt'].isin(anomaly_classes)]

    # We encode Normal with 1 and Anomaly with 0 for 'gt' next
    normal_df.loc[:, 'gt'] = 1
    anomaly_df.loc[:, 'gt'] = 0

    # We need to create the train and test splits next. Because we are dealing with time-series data, we cannot perform the split randomly.
    # to avoid temporal leakage. Therefore, we will use a time-based split.
    # We are going ot use the [2013-2022] period for training and the [2023-2025] period for testing in Normal split.
    # For Anomaly split, we will use the [2023-2025] period for testing.
    #
    # Also, because we are using Isolation Forest, we need to use part of the Normal split for training. NO SAMPLE IN ANOMALY SPLIT
    # WILL BE USED IN TRAINING THE MODEL, WHICH IS AN IMPORTANT DISTINCTION FROM THE TRADITIONAL ML TRAINING/TESTING.
    # Since our Anomaly dataframe includes few samples, we will use all of it for testing. We will also add the same number of samples to the test set
    # using the Normal dataframe's test split.

    get_year = lambda intTimestamp: datetime.strptime(str(intTimestamp), '%Y%m%d').year  # helper lambda function to get the year from the dataset 'end_date'
    get_date = lambda intTimestamp: datetime.strptime(str(intTimestamp), '%Y%m%d').date()  # helper lambda function to get the date from the dataset 'end_date'

    train_normal_df = normal_df[(normal_df['end_date'].apply(get_year) >= train_window['lo']) & (normal_df['end_date'].apply(get_year) <= train_window['hi'])]
    test_normal_df = normal_df[(normal_df['end_date'].apply(get_year) >= test_window['lo']) & (normal_df['end_date'].apply(get_year) <= test_window['hi'])]
    #anomaly_df will have the 2013 to 2025 content already. So will use it as "test_anomaly_df"  directly.

    # We shuffle the training data at this point. Shuffling the training data is important to break the ticker groups before building
    # trees in Isolation Forest. That way every tree in the forest will see a representative mix of all tickers across multiple years.
    # If we don't shuffle the training data, the model will be biased towards the ticker groups.
    random_seed = 1974  # for repeatability of the results
    train_normal_df = train_normal_df.sample(frac=1, random_state=random_seed)  # keeps 100% of all rows and shuffles them

    train_X = train_normal_df.drop(columns=column_defs['meta'] + [column_defs['target']])  # numeric only feature matrix for training
    train_y = train_normal_df[column_defs['target']]  # target vector for training

    # We will NOT shuffle the test set. However, we need to ensure the the normal and anomaly test sets are chronologically ordered.
    # This will allow us to evaluate how the performance of the model changes over time.
    # Note after the chronological order, different tickers will also be interlaced and will not be grouped anymore (as in the original dataset)
    # However, this is not a problem from an evaluation perspective and is a realistic way of testing the model. Since Isolation Forest looks
    # at each sample and features in a row independent of other samples (i.e., it has no memoery like the LSTMs, for example).
    # We are basically performing a hard test to see if the Isolation Forest can identify anomalies regardless of the stock ticker.
    # Also, for the FAANG dataset used, len(test_normal_df) >= 12 x len(anomaly_df), which is what we want to stress test the model.
    # If we were to make the two set lengths equal, we would be skewing the real-world anomaly scenario of FAANG stocks.
    # This would lead to a catastrophic failure in the real-world!
    test_df = pd.concat([test_normal_df, anomaly_df], ignore_index=True)  # Index in the concatenated dataframe is reset and made continuous
    test_df['end_date'] = test_df['end_date'].apply(get_date) # convert end_time column to date time for sorting
    test_df = test_df.sort_values(by='end_date', ascending=True)  # sort the dataframe by 'end_date' in ascending order

    test_X = test_df.drop(columns=column_defs['meta'] + [column_defs['target']])  # combined and ordered numerics-onlyfeature matrix for testing
    test_y = test_df[column_defs['target']]  # combined and ordered target vector for testing

    print(f"Train set: {len(train_X)} samples")
    print(f"Test normal set: {len(test_X)} samples")
    print(f"Test anomaly set: {len(anomaly_df)} samples")  # All samples in anomaly_df used in test set.

    # Scale training and test data next. We will use the StandardScaler from scikit-learn.
    scaler = StandardScaler()
    scaler.fit(train_X)  # fit ONCE using the historical train data (one ruler to measure it all)
    train_X_scaled = scaler.transform(train_X)  # perform scaling on train_X
    test_X_scaled = scaler.transform(test_X)  # perform scaling on test_X

    # Visualize the train_X_scaled data next.
    plot_attributes = {'save_folder': analysis_attributes['plot_save_folder'], 'filename': 'train_X_scaled.png', 'x_label': 'time_step (day)'
    , 'y_label':'scaled magnitude', 'title': 'Features After Scaling'}

    traces = [
        {'x': list(range(0, len(train_X))), 'y': train_X_scaled[:, train_X.columns.get_loc('slope')], 'label': 'slope', 'color': 'b'},
        {'x': list(range(0, len(train_X))), 'y': train_X_scaled[:, train_X.columns.get_loc('zcr')], 'label': 'zcr', 'color': 'r'},
        {'x': list(range(0, len(train_X))), 'y': train_X_scaled[:, train_X.columns.get_loc('volatility')], 'label': 'volatility', 'color': 'g'},
        {'x': list(range(0, len(train_X))), 'y': train_X_scaled[:, train_X.columns.get_loc('trend_strength')], 'label': 'trend_strength', 'color': 'm'}
    ]

    plot_session = ps.scatterPlot2D(plot_attributes)
    plot_session.plot(traces=traces)

   # Visualize the test_X_scaled data next.
    plot_attributes = {'save_folder': analysis_attributes['plot_save_folder'], 'filename': 'test_X_scaled.png', 'x_label': 'time_step (day)'
    , 'y_label':'scaled magnitude', 'title': 'Features After Scaling'}

    traces = [
        {'x': list(range(0, len(test_X))), 'y': test_X_scaled[:, test_X.columns.get_loc('slope')], 'label': 'slope', 'color': 'b'},
        {'x': list(range(0, len(test_X))), 'y': test_X_scaled[:, test_X.columns.get_loc('zcr')], 'label': 'zcr', 'color': 'r'},
        {'x': list(range(0, len(test_X))), 'y': test_X_scaled[:, test_X.columns.get_loc('volatility')], 'label': 'volatility', 'color': 'g'},
        {'x': list(range(0, len(test_X))), 'y': test_X_scaled[:, test_X.columns.get_loc('trend_strength')], 'label': 'trend_strength', 'color': 'm'}
    ]

    plot_session = ps.scatterPlot2D(plot_attributes)
    plot_session.plot(traces=traces)

    # Next, we will build the Isolation Forest model.

    # Performance metric that matters the most is Precision. We want the model to identify the True Positives with high accuracy without missing
    # any rare occurence of anomalies !

    # CONTAMINATION: During model training:
    # initially set contamination='auto' to use the default method to identify the contamination rate in the training set.
    # Then use decision_function() for testing. Look at the raw scores rather than the -1/1 output.
    # Compare the Test Normal and Test Anomaly score and see how much they overlap. If the overlap is too much, that means the features are not
    # strong enough to distinguish between normal and anomaly. If there is a clear gap, we can set a manual contamination threshold using the
    # number of TREND_UP and TREND_DOWN samples and the number of normal samples in test_set.   
    # Alternatively, we can write a Threshold Tuning script to scan through different contamination levels to se which one maximizes our F1
    # score on the test set!


