
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import roc_curve, average_precision_score


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

    
    #initial_isolation_forest_exploration(analysis_attributes, dsObj)  # used for understanding the model behavior and obtaining early results.

    isolation_forest_optimized_model(analysis_attributes, dsObj)  # after the initial exploration, this is where the model is tuned and optimized.


def isolation_forest_optimized_model(analysis_attributes: dict, dsObj: di.DataInsights):
    """
    Isolation Forest Model Development and Evaluation:
    The "Normal" group will be rows with "OSCILLATING", "OTHER" AND "STATIONARY" class labels.
    The "Anomaly" group will be rows with "TREND_UP" AND "TREND_DOWN" class labels.
    """

    # We need to define train, validation and test splits.
    # Trein set below is whhere our pure normal samples will come from that represents baseline FAANG stock behavior over 10 years.
    train_window = {'lo': 2013, 'hi': 2017}  # Training samples will be picked for these year range in the dataset (inclusive)
    
    # Validation set is selected as the challenging years after Covid with significant market volatility. Grid search will be based on this set
    # for robustness! Time window selected to ensure there is sufficient number of anomalies in it!
    validation_window = {'lo': 2018, 'hi': 2021}  # Validation samples will be picked for these year range in the dataset (inclusive)
    
    # test window selected to have sufficient number of anomalies in it!
    #test_window = {'lo': 2024, 'hi': 2025}  # Testing samples will be picked for these year range in the dataset (inclusive)
    test_window = {'lo': 2024, 'hi': 2025}  # Testing samples will be picked for these year range in the dataset (inclusive)


    # Following are the column names and their categories in this dataset
    column_defs = {'features': ['slope', 'zcr', 'trend_strength', 'volatility'],  # features to train on
                    'target': 'gt_flag',  # aka encoded ground truth
                    'meta': ['end_date', 'ticker', 'gt']  # 'gt' keeps the original string labels of the data
                }

    normal_classes = ['OSCILLATING', 'OTHER', 'STATIONARY']
    anomaly_classes = ['TREND_UP', 'TREND_DOWN']
    normal_df = dsObj.df[dsObj.df['gt'].isin(normal_classes)]
    anomaly_df = dsObj.df[dsObj.df['gt'].isin(anomaly_classes)]

    # We encode Normal with 0 and Anomaly with 1 for 'gt' next
    normal_df.loc[:, 'gt_flag'] = 0  # Inliers code = 0
    anomaly_df.loc[:, 'gt_flag'] = 1  # Outliers code = 1  (anomaly is the positive class of interest)

    # We need to create the train and test splits next. Because we are dealing with time-series data, we cannot perform the split randomly.
    # to avoid temporal leakage. Therefore, we will use a time-based split while avoiding temporal leakage.
    # We will create the following 3 splits:
    # 1 - Train set: [2013-2017]: Will include normal samples only
    # 2 - Validation set: [2018-2021]: Will include a mixture of normal and anomaly samples
    # 3 - Test set: [2024-2025]: Will include a mixture of normal and anomaly samples
    # Note years 2022 and 2023 are not used for training, validation or testing due to their high volatility.

    # Also, because we are using Isolation Forest, we need to use part of the Normal split for training. NO SAMPLE IN ANOMALY SPLIT
    # WILL BE USED IN TRAINING THE MODEL, WHICH IS AN IMPORTANT DISTINCTION FROM THE TRADITIONAL ML TRAINING/TESTING.
    # Since our Anomaly dataframe includes few samples, we will use all of it for testing. We will also add the same number of samples to the test set
    # using the Normal dataframe's test split.

    get_year = lambda intTimestamp: datetime.strptime(str(intTimestamp), '%Y%m%d').year  # helper lambda function to get the year from the dataset 'end_date'
    get_date = lambda intTimestamp: datetime.strptime(str(intTimestamp), '%Y%m%d').date()  # helper lambda function to get the date from the dataset 'end_date'

    train_normal_df = normal_df[(normal_df['end_date'].apply(get_year) >= train_window['lo']) & (normal_df['end_date'].apply(get_year) <= train_window['hi'])]
    val_normal_df = normal_df[(normal_df['end_date'].apply(get_year) >= validation_window['lo']) & (normal_df['end_date'].apply(get_year) <= validation_window['hi'])]
    test_normal_df = normal_df[(normal_df['end_date'].apply(get_year) >= test_window['lo']) & (normal_df['end_date'].apply(get_year) <= test_window['hi'])]

    # We shuffle the training data at this point. Shuffling the training data is important to break the ticker groups before building
    # trees in Isolation Forest. That way every tree in the forest will see a representative mix of all tickers across multiple years.
    # If we don't shuffle the training data, the model will be biased towards the ticker groups.
    random_seed = 1974  # for repeatability of the results
    train_normal_df = train_normal_df.sample(frac=1, random_state=random_seed)  # keeps 100% of all rows and shuffles them

    train_X = train_normal_df.drop(columns=column_defs['meta'] + [column_defs['target']])  # leave numeric only feature matrix for training
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
    # NOTE: we are only using a subset of the complete anomaly samples in the dataset that fall within the test set date window!!
    anomaly_df_in_test_years = anomaly_df[(anomaly_df['end_date'].apply(get_year) >= test_window['lo']) & (anomaly_df['end_date'].apply(get_year) <= test_window['hi'])]
    test_df = pd.concat([test_normal_df, anomaly_df_in_test_years], ignore_index=True)  # Index in the concatenated dataframe is reset and made continuous
    test_df['end_date'] = test_df['end_date'].apply(get_date) # convert end_time column to date time for sorting
    test_df = test_df.sort_values(by='end_date', ascending=True)  # sort the dataframe by 'end_date' in ascending order

    test_X = test_df.drop(columns=column_defs['meta'] + [column_defs['target']])  # combined and ordered numerics-onlyfeature matrix for testing
    test_y = test_df[column_defs['target']]  # combined and ordered target vector for testing
    test_X_meta_data = test_df[column_defs['meta'] + [column_defs['target']]].copy()  # copy metadata of test for evaluation later on

    # Create validation set as the test set
    anomaly_df_in_validation_years = anomaly_df[(anomaly_df['end_date'].apply(get_year) >= validation_window['lo']) & (anomaly_df['end_date'].apply(get_year) <= validation_window['hi'])]
    validation_df = pd.concat([val_normal_df, anomaly_df_in_validation_years], ignore_index=True)  # Index in the concatenated dataframe is reset and made continuous
    validation_df['end_date'] = validation_df['end_date'].apply(get_date) # convert end_time column to date time for sorting
    validation_df = validation_df.sort_values(by='end_date', ascending=True)  # sort the dataframe by 'end_date' in ascending order
    validation_X = validation_df.drop(columns=column_defs['meta'] + [column_defs['target']])  # combined and ordered numerics-onlyfeature matrix for validation
    validation_y = validation_df[column_defs['target']]  # combined and ordered target vector for validation
    #validation_X_meta_data = validation_df[column_defs['meta'] + [column_defs['target']]].copy()  # copy metadata of validation for evaluation later on

    print(f"TRAIN set: total samples{len(train_X)}, anomaly samples: NONE")
    print(f"VALIDATION: total samples: {len(validation_X)}, anomaly samples: {len(anomaly_df_in_validation_years)}")
    print(f"TEST: total samples: {len(test_X)}, anomaly samples: {len(anomaly_df_in_test_years)}")

    # Scale training and test data next. We will use the StandardScaler from scikit-learn.
    scaler = StandardScaler()
    scaler.fit(train_X)  # fit ONCE using the historical train data (one ruler to measure it all)
    train_X_scaled = scaler.transform(train_X)  # perform scaling on train_X
    validation_X_scaled = scaler.transform(validation_X)  # perform scaling on validation_X
    test_X_scaled = scaler.transform(test_X)  # perform scaling on test_X

    """
    # Visualize the train_X_scaled data next. (for verification and documentation)
    plot_attributes = {'save_folder': analysis_attributes['plot_save_folder'], 'filename': 'optimized_train_X_scaled.png', 'x_label': 'time_step (day)'
    , 'y_label':'scaled magnitude', 'title': 'Features After Scaling'}
    plot_scaled_features(plot_attributes, train_X_scaled, train_X.columns)
    plot_attributes['filename'] = 'optimized_validation_X_scaled.png'
    plot_scaled_features(plot_attributes, validation_X_scaled, validation_X.columns)
    plot_attributes['filename'] = 'optimized_test_X_scaled.png'
    plot_scaled_features(plot_attributes, test_X_scaled, test_X.columns)
    """

    # Nest we will perform a Grid search to find the best hyperparameters for the Isolation Forest model.
    X_combined = np.vstack([train_X_scaled, validation_X_scaled])
    y_combined = np.concatenate([train_y, validation_y])

    # We define the test fold next. test_fold values -1 are used for training and 0 for validation by convention in PredefinedSplit()
    test_fold = np.concatenate([-1 * np.ones(len(train_X_scaled)), 0 * np.ones(len(validation_X_scaled))])
    pds = PredefinedSplit(test_fold)  # using a wrapper for scikitlearn's GridSearchCV() to handle the test fold as needed.

    # Define parameter grid for the search
    param_grid = {
        'contamination': ['auto'], # keep this fixed for now!
        'max_features': [0.5, 0.7, 0.8], # keep this fixed for now!
        'random_state': [random_seed], # for repeatability of the results
        'n_estimators': [100, 200, 500], # number of trees to build in the forest
        'max_samples': [512, 1024, 2048], # number of samples to use at random for building each tree in the model
    }

    grid_search = GridSearchCV(
        estimator=IsolationForest(),
        param_grid=param_grid,
        cv=pds, # cross-validation based on the test_fold defined earlier
        scoring=pr_auc_scorer,  # internally calls the average_precision_score() function to calculate the PR AUC score
        n_jobs=-1
        )  # n_jobs=-1 uses all available cores for parallel processing


    grid_search.fit(X_combined, y_combined)
    print(f"Grid search results:")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")

    # We can also see the raking of all options tested:
    results_df = pd.DataFrame(grid_search.cv_results_)
    leaderboard = results_df[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
    # mean_test_score  is the PR-AUC score we are interested in
    leaderboard = leaderboard.sort_values(by='mean_test_score', ascending=False)
    print(f"Leaderboard:")
    print(leaderboard)
    
    # Next, using the best estimator, let's try to estimate the optimal contamination rate. (it was set to 'auto' in the grid search)
    best_estimator = grid_search.best_estimator_

    # Generate raw scores for the validation set using the best estimator
    validation_X_scaled_scores = best_estimator.decision_function(validation_X_scaled)
    # Scores need to be negated prior to feeding to the roc_curve. Positive class needs to have high scores by convention!
    fpr, tpr, thresholds = roc_curve(validation_y, -validation_X_scaled_scores)

    # We use Youden's J statistic to find the optimal threshold.
    # Youden's J statistic is the maximum vertical (NOT ORTHOGONAL) distance between the ROC curve and the diagonal line.
    # J = TPR - FPR
    j_scores = tpr - fpr  # we need to find the index of the maximum J (distance) here
    optimal_threshold = -thresholds[j_scores.argmax()]  # we need to NEGATE the threshold back to the original scale

    print(f"Optimal threshold: {optimal_threshold}")

    # Now that we have the optimal threshold, we can calculate the contamination factor using the entire dataset!
    X_total = np.concat([train_X_scaled, validation_X_scaled])
    total_scores = best_estimator.decision_function(X_total)

    contamination_factor = np.sum(total_scores < optimal_threshold) / len(total_scores)
    print(f"Optimal contamination factor: {contamination_factor}")

    # Now we have all the information needed to create the final model
    final_model = IsolationForest(
        n_estimators=grid_search.best_params_['n_estimators'],
        contamination=contamination_factor,
        max_features=grid_search.best_params_['max_features'],
        max_samples=grid_search.best_params_['max_samples'],
        random_state=random_seed
        )

    final_model.fit(train_X_scaled)
    test_X_scaled_scores = final_model.decision_function(test_X_scaled)
    test_X_meta_data['isf_score'] = test_X_scaled_scores

    # Visualize the normal vs anomaly histograms to see the effectiveness of the trained model on the test set.
    normal_OSCILLATING_scores = test_X_meta_data[test_X_meta_data['gt'] == 'OSCILLATING']['isf_score']
    normal_OTHER_scores = test_X_meta_data[test_X_meta_data['gt'] == 'OTHER']['isf_score']
    normal_STATIONARY_scores = test_X_meta_data[test_X_meta_data['gt'] == 'STATIONARY']['isf_score']
    anomaly_TREND_UP_scores = test_X_meta_data[test_X_meta_data['gt'] == 'TREND_UP']['isf_score']
    anomaly_TREND_DOWN_scores = test_X_meta_data[test_X_meta_data['gt'] == 'TREND_DOWN']['isf_score']

    plot_attributes = {'save_folder': analysis_attributes['plot_save_folder'], 'filename': 'normal_vs_anomaly_CLASSIFIED_best_estimator.png', 'x_label': 'isf_score', 'y_label': 'count', 'title': 'Normal vs Anomaly Classified Histogram'}
    traces = [
        {'data': normal_OSCILLATING_scores, 'label': 'normal OSCILLATING', 'color': 'b', 'bins': 30, 'alpha': 0.4},
        {'data': normal_OTHER_scores, 'label': 'normal OTHER', 'color': 'g', 'bins': 30, 'alpha': 0.3},
        {'data': normal_STATIONARY_scores, 'label': 'normal STATIONARY', 'color': 'yellow', 'bins': 30, 'alpha': 1},
        {'data': anomaly_TREND_UP_scores, 'label': 'anomaly TREND_UP', 'color': 'r', 'bins': 30, 'alpha': 0.2, 'edgecolor':'black'},
        {'data': anomaly_TREND_DOWN_scores, 'label': 'anomaly TREND_DOWN', 'color': 'black', 'bins': 30, 'alpha': 0.5}
    ]

    reference_lines = [
        {'type': 'horizontal', 'value': optimal_threshold, 'color': 'red', 'linewidth': 2, 'linestyle': '--', 'label': f'optimum threshold = {optimal_threshold:.4f}'},
        #{'type': 'vertical', 'value': 100, 'color': 'blue', 'linewidth': 2, 'linestyle': '--', 'label': 'Count 100'}
    ]

    plot_session = ps.histPlot2D(plot_attributes, reference_lines)
    plot_session.plot(traces=traces)


def pr_auc_scorer(estimator, X, y):
    """
    Precision-Recall AUC Scorer
    We negate decision_function() because Isolation Forest assigns low (i.e., negative) scores to anomalies.
    but scorers expect higher values for positive class (which is the anomaly samples in or case!)
    """
    negated_scores = -estimator.decision_function(X)
    return average_precision_score(y, negated_scores)


def initial_isolation_forest_exploration(analysis_attributes: dict, dsObj: di.DataInsights):
    """
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

    """
    Isolation Forest Model Development and Evaluation:
    The "Normal" group will be rows with "OSCILLATING", "OTHER" AND "STATIONARY" class labels.
    The "Anomaly" group will be rows with "TREND_UP" AND "TREND_DOWN" class labels.
    """

    train_window = {'lo': 2013, 'hi': 2022}  # Training samples will be picked for these year range in the dataset (inclusive)
    test_window = {'lo': 2023, 'hi': 2025}  # Testing samples will be picked for these year range in the dataset (inclusive)

    # Following are the column names and their categories in this dataset
    column_defs = {'features': ['slope', 'zcr', 'trend_strength', 'volatility'],  # features to train on
                    'target': 'gt_flag',  # aka encoded ground truth
                    'meta': ['end_date', 'ticker', 'gt']  # 'gt' keeps the original string labels of the data
                }

    normal_classes = ['OSCILLATING', 'OTHER', 'STATIONARY']
    anomaly_classes = ['TREND_UP', 'TREND_DOWN']
    normal_df = dsObj.df[dsObj.df['gt'].isin(normal_classes)]
    anomaly_df = dsObj.df[dsObj.df['gt'].isin(anomaly_classes)]

    # We encode Normal with 0 and Anomaly with 1 for 'gt' next
    normal_df.loc[:, 'gt_flag'] = 0  # Inliers code = 0
    anomaly_df.loc[:, 'gt_flag'] = 1  # Outliers code = 1  (anomaly is the positive class of interest)

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

    # We shuffle the training data at this point. Shuffling the training data is important to break the ticker groups before building
    # trees in Isolation Forest. That way every tree in the forest will see a representative mix of all tickers across multiple years.
    # If we don't shuffle the training data, the model will be biased towards the ticker groups.
    random_seed = 1974  # for repeatability of the results
    train_normal_df = train_normal_df.sample(frac=1, random_state=random_seed)  # keeps 100% of all rows and shuffles them

    train_X = train_normal_df.drop(columns=column_defs['meta'] + [column_defs['target']])  # leave numeric only feature matrix for training
    #train_y = train_normal_df[column_defs['target']]  # target vector for training

    # We will NOT shuffle the test set. However, we need to ensure the the normal and anomaly test sets are chronologically ordered.
    # This will allow us to evaluate how the performance of the model changes over time.
    # Note after the chronological order, different tickers will also be interlaced and will not be grouped anymore (as in the original dataset)
    # However, this is not a problem from an evaluation perspective and is a realistic way of testing the model. Since Isolation Forest looks
    # at each sample and features in a row independent of other samples (i.e., it has no memoery like the LSTMs, for example).
    # We are basically performing a hard test to see if the Isolation Forest can identify anomalies regardless of the stock ticker.
    # Also, for the FAANG dataset used, len(test_normal_df) >= 12 x len(anomaly_df), which is what we want to stress test the model.
    # If we were to make the two set lengths equal, we would be skewing the real-world anomaly scenario of FAANG stocks.
    # This would lead to a catastrophic failure in the real-world!
    # NOTE: we are only using a subset of the complete anomaly samples in the dataset that fall within the test set date window!!
    anomaly_df_in_test_years = anomaly_df[(anomaly_df['end_date'].apply(get_year) >= test_window['lo']) & (anomaly_df['end_date'].apply(get_year) <= test_window['hi'])]
    test_df = pd.concat([test_normal_df, anomaly_df_in_test_years], ignore_index=True)  # Index in the concatenated dataframe is reset and made continuous
    test_df['end_date'] = test_df['end_date'].apply(get_date) # convert end_time column to date time for sorting
    test_df = test_df.sort_values(by='end_date', ascending=True)  # sort the dataframe by 'end_date' in ascending order

    test_X = test_df.drop(columns=column_defs['meta'] + [column_defs['target']])  # combined and ordered numerics-onlyfeature matrix for testing
    #test_y = test_df[column_defs['target']]  # combined and ordered target vector for testing
    test_X_meta_data = test_df[column_defs['meta'] + [column_defs['target']]].copy()  # copy metadata of test for evaluation later on

    print(f"Train set: {len(train_X)} samples")
    print(f"Test normal set: {len(test_X)} samples")
    print(f"Test anomaly set: {len(anomaly_df_in_test_years)} samples")  # Relevant samples in anomaly_df used in test set.

    # Scale training and test data next. We will use the StandardScaler from scikit-learn.
    scaler = StandardScaler()
    scaler.fit(train_X)  # fit ONCE using the historical train data (one ruler to measure it all)
    train_X_scaled = scaler.transform(train_X)  # perform scaling on train_X
    test_X_scaled = scaler.transform(test_X)  # perform scaling on test_X

    """
    # Visualize the train_X_scaled data next. (for verification and documentation)
    plot_attributes = {'save_folder': analysis_attributes['plot_save_folder'], 'filename': 'train_X_scaled.png', 'x_label': 'time_step (day)'
    , 'y_label':'scaled magnitude', 'title': 'Features After Scaling'}
    plot_scaled_features(plot_attributes, train_X_scaled, train_X.columns)
    plot_attributes['filename'] = 'test_X_scaled.png'
    plot_scaled_features(plot_attributes, test_X_scaled, test_X.columns)
    """

    # Next, we will build the Isolation Forest model.
    # IMPORTANT: contamination does not affect the results of the decision_function()
    # but it affects the results of the predict() function, when it is used!
    isf_model = IsolationForest(n_estimators=200, contamination=0.02, max_features=1.0, random_state=random_seed)  # model initialization
    isf_model.fit(train_X_scaled)  # training phase
    test_X_scaled_scores = isf_model.decision_function(test_X_scaled)  # scoring phase

    test_X_meta_data['isf_score'] = test_X_scaled_scores

    # Visualize the normal vs anomaly histograms to see the effectiveness of the trained model on the test set.
    normal_sample_scores = test_X_meta_data[test_X_meta_data['gt_flag'] == 0]['isf_score']
    anomaly_sample_scores = test_X_meta_data[test_X_meta_data['gt_flag'] == 1]['isf_score']

    plot_attributes = {'save_folder': analysis_attributes['plot_save_folder'], 'filename': 'normal_vs_anomaly_isf_scores.png', 'x_label': 'isf_score', 'y_label': 'count', 'title': 'Normal vs Anomaly ISF Scores Histogram'}
    traces = [
        {'data': normal_sample_scores, 'label': 'normal', 'color': 'b', 'bins': 30},
        {'data': anomaly_sample_scores, 'label': 'anomaly', 'color': 'r', 'bins': 30}
    ]

    plot_session = ps.histPlot2D(plot_attributes)
    plot_session.plot(traces=traces)

    # Checks to see if the model is failing globally or due to specific market conditions:
    # 1 - Color code each label in normal samples and plot the histogram to see which label gets mixed with the anomaly samples the most.
    # INSIGHTS:
    # 1.1 - isf_score <= 0.0 is where the anomalies are predicted with high precision and recall. This is the high conviction trading zone.
    # 1.2 - The anomaly samples overlap with the normal samples where their occurence is less likelyi.e., we don't see anomaly samples bleeding into the
    # center of the highest occurence of normal samples (i.e., the isf_score >0.15 region) which is a good thing.
    # 1.3 - A low score of 0.25 indicates the splits occur too quickly in the forest. This occurs when max_samples is too low for the given data size.
    # Increasing max_samples (with the n_estimators) can help improve the separation of anomaly and normal samples)
    # 1.4 - The isf_score range is -0.1 < isf_score <= 0.25, which is a low score. 0.5 is what a typical inlier score is expected. This low score
    # indicates even the most "normal" samples (like OSCILLATING) in our dataset is spread out in 4D space and does not cluster densely as a ball. The model is seeing
    # too much variability in the training set (as FAANG stocks behave inherently noisy and with high variability) 

    normal_OSCILLATING_scores = test_X_meta_data[test_X_meta_data['gt'] == 'OSCILLATING']['isf_score']
    normal_OTHER_scores = test_X_meta_data[test_X_meta_data['gt'] == 'OTHER']['isf_score']
    normal_STATIONARY_scores = test_X_meta_data[test_X_meta_data['gt'] == 'STATIONARY']['isf_score']
    anomaly_TREND_UP_scores = test_X_meta_data[test_X_meta_data['gt'] == 'TREND_UP']['isf_score']
    anomaly_TREND_DOWN_scores = test_X_meta_data[test_X_meta_data['gt'] == 'TREND_DOWN']['isf_score']

    plot_attributes = {'save_folder': analysis_attributes['plot_save_folder'], 'filename': 'normal_vs_anomaly_CLASSIFIED.png', 'x_label': 'isf_score', 'y_label': 'count', 'title': 'Normal vs Anomaly Classified Histogram'}
    traces = [
        {'data': normal_OSCILLATING_scores, 'label': 'normal OSCILLATING', 'color': 'b', 'bins': 30, 'alpha': 0.4},
        {'data': normal_OTHER_scores, 'label': 'normal OTHER', 'color': 'g', 'bins': 30, 'alpha': 0.3},
        {'data': anomaly_TREND_UP_scores, 'label': 'anomaly TREND_UP', 'color': 'r', 'bins': 30, 'alpha': 0.2, 'edgecolor':'black'},
        {'data': anomaly_TREND_DOWN_scores, 'label': 'anomaly TREND_DOWN', 'color': 'black', 'bins': 30, 'alpha': 0.5},
        {'data': normal_STATIONARY_scores, 'label': 'normal STATIONARY', 'color': 'yellow', 'bins': 30, 'alpha': 1}
    ]

    plot_session = ps.histPlot2D(plot_attributes)
    plot_session.plot(traces=traces)

    # 2 - Calculate Mahalanobis distance between anomaly samples in the test setand the normal regime in the training set. Interpret as follows:
    # a - HIGH DISTANCE & HIGH ANOMALY (i.e. very low negative score): The sample is an outlier and the Isolation Forest correctly flagged it.
    # b - HIGH DISTANCE & LOW ANOMALY (i.e., high positive score): Sample is an outlier but the Isolation Forest failed to catch it. The shape of anomaly 
    # is not something a forest can catch.
    # c - LOW DISTANCE & HIGH ANOMALY (i.e., very low negative score): Counter-intuitive point. Sample is not an outlier but the Isolation Forest flagged it as
    # strong anomaly. 
    #
    # 3 - Calculate Precision Recall for grouped results by ticker to see if any one stands out.
    #
    # 4 - Check if a particluar ticker is responsible for the Normal bars leaking into anomaly score zone (i.e., isf_score <= 0.10)
    #
    # 5 - Check to see if model fails during the 2022-2023 transition or the flat market periods in 2024 If 2024 Normal scores are significantly lower,
    # that means the market has structurally shiftted, and our training data is no longer representative)
    #
    # 6 - Check for anomaly samples with high scores (i.e., isf_score > 0.10). Then check the trend_strength and volatility values
    # to anomalies with low scores (i.e., isf_score <= 0.10).


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


def plot_scaled_features(plot_attributes, X_scaled: np.ndarray, columns: list):
    
    traces = [
        {'x': list(range(0, len(X_scaled))), 'y': X_scaled[:, columns.get_loc('slope')], 'label': 'slope', 'color': 'b', 'trace_type': 'scatter'},
        {'x': list(range(0, len(X_scaled))), 'y': X_scaled[:, columns.get_loc('zcr')], 'label': 'zcr', 'color': 'r', 'trace_type': 'scatter'},
        {'x': list(range(0, len(X_scaled))), 'y': X_scaled[:, columns.get_loc('volatility')], 'label': 'volatility', 'color': 'g', 'trace_type': 'scatter'},
        {'x': list(range(0, len(X_scaled))), 'y': X_scaled[:, columns.get_loc('trend_strength')], 'label': 'trend_strength', 'color': 'm', 'trace_type': 'scatter'}
    ]

    plot_session = ps.scatterPlot2D(plot_attributes)
    plot_session.plot(traces=traces)