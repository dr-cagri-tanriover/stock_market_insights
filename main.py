from dataset_builder import DatalakeBuilder
from feature_dataset_builder import FeatureDatasetBuilder
from pathlib import Path

import data_insights as di


def run_datalake_builder():

    stock_tickers=['META', 'AAPL', 'AMZN', 'NFLX', 'GOOG']  # FAANG stocks. There is built-in protection for duplicates in this list!
    start_date, end_date = '2014-01-01', '2024-12-31'   # 10 years of data per ticker will be collected.
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

def get_feature_dataset_insights():

    dataset_filepath = "feature_dataset/feature_dataset.csv"
    reportout_filepath = Path("stock_insights_report.pdf")
    pdf_report_title = "FAANG STOCK DATA INSIGHTS"

    dsObj = di.DataInsights(dataset_filepath, reportout_filepath, pdf_report_title)

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
                                            'use_in_mahalanobis_distance': True
                                            },
                        'zcr': {
                                'thresholds': [('hi_osc_th', 0.46)],
                                'use_in_mahalanobis_distance': True
                                },
                        'volatility': {
                                        'thresholds': [('hi_noise_th', 0.02), ('lo_vol_th', 0.008)],
                                        'use_in_mahalanobis_distance': True
                                        },
                        'slope': {
                                    'thresholds': [('upward_th', 0.003), ('downward_th', -0.003), ('flatness_lo', -0.001), ('flatness_hi', 0.001)],
                                    'use_in_mahalanobis_distance': True  # slope provides sign information and trend_strength uses its magnitude.
                                    }
    }


    dsObj.confusion_risk_analysis(features_to_plot, class_column='gt', save_folder="plots", display_plots=False)

    dsObj.end_operation()  # Internal dsObj operations are wrapped up inside this function.


    # Press Enter key to exit the program
    print("\nPress Enter key to exit the program...")
    input()



def main():

    operation_modes = {
        'build_data_lake': "Grab data from Yahoo finance website using the API and store it under a specified folder.",
        'build_feature_dataset': "Create a feature dataset from the data lake by computing features for specified time frame in terms of days.",
        'get_data_insights': "Get insights from the feature dataset by applying a series of analyses and visualizations.",
        'train_model': "Train a machine learning model on the feature dataset and evaluate its performance."
    }

    MODE = 'get_data_insights'  # select key from operation_modes dictionary to run the corresponding function.
    
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
        get_feature_dataset_insights()
    elif MODE == 'train_model':
        # Step 4 - Ready to use the features dataset to train and test one or more machine learning models.
        print(f"Model training and testing features not implemented yet....")
    else:
        print(f"Invalid mode: {MODE}. Please select a valid mode from the following: {operation_modes.keys()}")

if __name__ == "__main__":
    main()
