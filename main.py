from dataset_builder import DatalakeBuilder
from feature_dataset_builder import FeatureDatasetBuilder
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

import data_insights as di
import utils.fileops as fops  # file operations tools


def run_datalake_builder():

    stock_tickers=['META', 'AAPL', 'AMZN', 'NFLX', 'GOOG']  # FAANG stocks. There is built-in protection for duplicates in this list!
    start_date, end_date = '2013-01-01', '2025-12-31'   # 13 years of data per ticker will be collected. (META starts in 2012)
    output_folder='stock_datalake'
    get_adjusted_prices=True  # Gets the adjusted prices as part of OHLCV information.
    remove_existing_datalake=True  # Deletes the  existing data lake if it exists.

    builderObj = DatalakeBuilder(
        tickers=stock_tickers,
        start_date=start_date,
        end_date=end_date,
        output_folder=output_folder,
        use_adjusted=get_adjusted_prices,
        delete_existing_data=remove_existing_datalake
    )

    builderObj.build_dataset()

def run_feature_dataset_construction():
    
    raw_data_folder='stock_datalake'
    metadata_file='metadata.json'

    feature_dataset_filename = 'feature_dataset.csv'  # file will be created in a subfolder called 'feature_datasets'

    days_per_feature = 30   # Total number of days (i.e., prices) that will be used to compute each feature

    featObj = FeatureDatasetBuilder(raw_data_folder, metadata_file, feature_dataset_filename, days_per_feature)

    featObj.build_feature_dataset()

def get_feature_dataset_insights(analysis_attributes: dict):

    #dataset_filepath = Path("feature_dataset/feature_dataset.csv")
    #reportout_filepath = Path("stock_insights_report.pdf")
    #pdf_report_title = "FAANG STOCK DATA INSIGHTS"

    # Note datalake_filepath is not needed for this analysis. Hence not initialized below.
    dsObj = di.DataInsights(
        dataset_filepath=Path(analysis_attributes['dataset_folderpath'], analysis_attributes['feature_dataset_filename']),
        reportout_filepath=analysis_attributes['reportout_filepath'],
        pdf_report_title=analysis_attributes['pdf_report_title']
    )

    # Start basic data insights generation (applicable to to all types of input datasets)
    dsObj.basic_info()
    dsObj.missing_values_analysis()
    dsObj.data_types_summary()
    dsObj.numeric_summary()
    dsObj.numeric_distributions(save_path="plots/numeric_distributions.png", display_plots=False)
    dsObj.categorical_summary()
    dsObj.numerical_statistics_for_categorical_columns()
    dsObj.correlation_analysis()
    # End of common dataset analysis

    # Following methods are specific to the dataset and are not applicable to all types of input datasets.
    # These methods also require knowledge of the dataset. Therefore, completing the common dataset analysis 
    # above first makes sense.
    #features_to_plot = ['trend_strength', 'zcr', 'volatility', 'slope']

    features_to_plot = {'trend_strength': {
                                            'thresholds': [('strength_th', 0.36)],
                                            'use_in_mahalanobis_distance': True,
                                            'k': 0.30 # used in label noise analysis (feature-specific value)
                                            },
                        'zcr': {
                                'thresholds': [('hi_osc_th', 0.46)],
                                'use_in_mahalanobis_distance': True,
                                'k': 0.30 # used in label noise analysis (feature-specific value)
                                },
                        'volatility': {
                                        'thresholds': [('hi_noise_th', 0.02), ('lo_vol_th', 0.008)],
                                        'use_in_mahalanobis_distance': True,
                                        'k': 0.25 # used in label noise analysis (feature-specific value)
                                        },
                        'slope': {
                                    'thresholds': [('upward_th', 0.003), ('downward_th', -0.003), ('flatness_lo', -0.001), ('flatness_hi', 0.001)],
                                    'use_in_mahalanobis_distance': True,  # slope provides sign information and trend_strength uses its magnitude.
                                    'k': 0.25 # used in label noise analysis (feature-specific value)
                                    }
    }

    temporal_stability_params = {'annual_time_step': 1, # determines the 'eras' of the dataset to be analyzed in units of years.
                                 #'sliding_months': 12 # determines the sliding window size for the analysis in units of months. For future use.
                                 }

    dsObj.confusion_risk_analysis(features_to_plot, class_column='gt', save_folder="plots", display_plots=False)

    dsObj.temporal_stability_analysis(temporal_stability_params, class_column='gt', save_folder="plots")

    # Following analysis needs the thresholds inside the feature_to_plot dictionary to work.
    dsObj.label_noise_analysis(features_to_plot, class_column='gt', save_folder="plots")

    dsObj.rule_based_classifier_analysis(features_to_plot, class_column='gt')

    dsObj.stress_test_splits_analysis(feature_dict=features_to_plot, class_column='gt', save_folder="plots")

    dsObj.end_operation()  # Internal dsObj operations are wrapped up inside this function.

    # Press Enter key to exit the program
    print("\nPress Enter key to exit the program...")
    input()


def extract_price_patterns(analysis_attributes: dict, target_patterns: dict):
    # Step 4 - Analyze the raw price patterns of the stocks in the dataset to identify any interesting patterns.
    # This feature is used after checking the report generated by running the 'get_data_insights' step
    # to visualize price movements of selected tickers, in selected time windows for selected classes

    # Lambda functions used:
    lake_file_fetch = lambda ticker, file_dict: [filename for filename in file_dict if filename.startswith(ticker)][0]
    get_fdate = lambda intdate: datetime.strptime(str(intdate), '%Y%m%d')  # end_date in feature dataset converted to datetime format
    get_sdate = lambda strdate: datetime.strptime(strdate, '%Y-%m-%d')  # user specified string date converted to datetime format

    push_df_row = lambda df, date, close: pd.concat([df, pd.DataFrame([[date, close]], columns=['Date', 'Close'])], ignore_index=True)
    pop_df_row = lambda df: df.drop(0, axis=0).reset_index(drop=True)   # FILO buffer operation.

    # Note datalake_filepath is not needed for this analysis. Hence not initialized below.
    dsObj = di.DataInsights(
        datalake_filepath=Path(analysis_attributes['datalake_folderpath']),
        dataset_filepath=Path(analysis_attributes['dataset_folderpath'], analysis_attributes['feature_dataset_filename']),
        reportout_filepath=analysis_attributes['reportout_filepath'],
        pdf_report_title=analysis_attributes['pdf_report_title']
    )

    feature_dataset_metadata = fops.read_json_content(analysis_attributes['dataset_folderpath'], analysis_attributes['feature_dataset_metadata_filename'])

    days_per_window = feature_dataset_metadata['features_info']['days_per_feature']  # number of days used to compute each feature


    for ticker, pattern_info in target_patterns.items():
        start_date = get_sdate(pattern_info['start_date'])   # converted string to datetime format
        end_date = get_sdate(pattern_info['end_date'])   # converted string to datetime format
        target_class = pattern_info['target_class']
        max_samples = pattern_info['max_samples']

        lake_csv_filename = lake_file_fetch(ticker, feature_dataset_metadata['file_info'])  # Fetches the datalake file for the ticker of interest.

        # Get sub-dataframe from feature dataset that only includes the the target_class
        subDf = dsObj.df[(dsObj.df['gt'] == target_class) & (dsObj.df['ticker'] == ticker)]  # ticker and target_class sub dataframe
        subDf = subDf[(subDf['end_date'].apply(get_fdate) >= start_date) & (subDf['end_date'].apply(get_fdate) <= end_date)]   # data in relevant time window
        subDf = subDf.iloc[:max_samples,:]  # first max_samples rows of the sub dataframe will be processed (there may be fewer rows, which is OK.)

        for row in subDf.itertuples():  # faster access to rows than .iterrows()
            # Fetch the next end_date in the sub data frame
            sample_end_date = get_fdate(row.end_date)  # end_date in feature dataset converted to datetime format
        
            priceDf = pd.DataFrame( [[np.nan, np.nan]] * days_per_window, columns=['Date', 'Close'])  # update buffer to accumulate the plot data.
            for csv_row in fops.yield_csv_rows(dsObj.datalake_filepath / lake_csv_filename):

                cur_date = get_fdate(csv_row['Date'])  # date in datalake csv file converted to datetime format
                priceDf = pop_df_row(priceDf)  # drop the oldest row in the buffer. This automatically has the effect of shift by one as well.
                priceDf = push_df_row(priceDf, csv_row['Date'], round(float(csv_row['Close']), 2))  # add new data to the end of the buffer

                if cur_date == sample_end_date:
                    # Accumulated data in priceDf needs to be plotted and saved.
                    dsObj.plot_price_pattern(priceDf, ticker, target_class, sample_end_date, analysis_attributes)
                    break # the for loop to stop the csv scan and move to the next end date to search for.


def main():

    operation_modes = {
        'build_data_lake': "Grab data from Yahoo finance website using the API and store it under a specified folder.",
        'build_feature_dataset': "Create a feature dataset from the data lake by computing features for specified time frame in terms of days.",
        'get_data_insights': "Get insights from the feature dataset by applying a series of analyses and visualizations.",
        'price_pattern_analysis': "Analyze the raw price patterns of the stocks in the dataset to identify any interesting patterns.",
        'train_model': "Train a machine learning model on the feature dataset and evaluate its performance."
    }

    analysis_attributes = {
    'datalake_folderpath': Path("stock_datalake"),
    'dataset_folderpath': Path("feature_dataset"),
    'datalake_metadata_filename': "metadata.json",
    'feature_dataset_metadata_filename': "feature_dataset_metadata.json",
    'feature_dataset_filename': "feature_dataset.csv"
    }


    MODE = 'price_pattern_analysis'  # select key from operation_modes dictionary to run the corresponding function.

    ###############################################################################################################

    if MODE == 'build_data_lake':
        # Step 1 - Build the high level data lake for specified parameters
        run_datalake_builder()
    elif MODE == 'build_feature_dataset':
        # Step 2 - If a data lake has been built previously, you can start from this step to start using it.
        # This step involves creating a dataset (as a subset of the data lake) for model training and testing
        # as well as generating ground truth labels for the dataset.
        run_feature_dataset_construction()
    elif MODE == 'get_data_insights':
        # Step 3 - If a feature_dataset has been built previously, you can start from this step to start using it.
        # This step will run analyses on that feature dataset and provide insights, visualizations and reports
        # to assess the fitness of the dataset for model training and testing in Step 4. Depending on the generated
        # insights additional work on the feature dataset may be required in Step 2 to refine/regenerate the feature dataset.
        analysis_attributes['reportout_filepath'] = Path("stock_insights_report.pdf")  # pdf file name for writing the results of the analysis.
        analysis_attributes['pdf_report_title'] = "FAANG STOCK DATA INSIGHTS"  # Title of insights analysis report
        get_feature_dataset_insights(analysis_attributes)
    elif MODE == 'price_pattern_analysis':
        # Step 4 - Analyze the raw price patterns of the stocks in the dataset to identify any interesting patterns.
        # This feature is used after checking the report generated by running the 'get_data_insights' step
        # to visualize price movements of selected tickers, in selected time windows for selected classes

        target_patterns = {
            'META': {
                'start_date': '2013-01-01',
                'end_date': '2025-12-31',
                'target_class': 'STATIONARY',
                'max_samples': 2        # the first few samples to capture
            },

            'AAPL': {
                'start_date': '2013-01-01',
                'end_date': '2025-12-31',
                'target_class': 'STATIONARY',
                'max_samples': 2        # the first few samples to capture
            },
            'AMZN': {
                'start_date': '2013-01-01',
                'end_date': '2025-12-31',
                'target_class': 'STATIONARY',
                'max_samples': 2        # the first few samples to capture
            },
            'NFLX': {
                'start_date': '2013-01-01',
                'end_date': '2025-12-31',
                'target_class': 'STATIONARY',
                'max_samples': 2        # the first few samples to capture
            },
            'GOOG': {
                'start_date': '2013-01-01',
                'end_date': '2025-12-31',
                'target_class': 'STATIONARY',
                'max_samples': 2        # the first few samples to capture
            }
        }


        analysis_attributes['reportout_filepath'] = Path("price_patterns.pdf")  # pdf file name for writing the results of the analysis.
        analysis_attributes['pdf_report_title'] = "SELECTED PRICE PATTERNS"  # Title of insights analysis report
        analysis_attributes['plot_save_folder'] = Path("plots")  # folder to save the plots
        extract_price_patterns(analysis_attributes, target_patterns)

    elif MODE == 'train_model':
        # Step 4 - Ready to use the features dataset to train and test one or more machine learning models.
        print(f"Model training and testing features not implemented yet....")
    else:
        print(f"Invalid mode: {MODE}. Please select a valid mode from the following: {operation_modes.keys()}")

if __name__ == "__main__":
    main()
