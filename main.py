from dataset_builder import DatalakeBuilder


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

def main():
    # Step 1 - Build the high level data lake for specified parameters
    run_datalake_builder()

    # Step 2 - If a data lake has been built previously, you can start from this step to start using it.
    # This step involves creating a dataset (as a subset of the data lake) for model training and testing
    # as well as generating ground truth labels for the dataset.
    #run_dataset_construction()


if __name__ == "__main__":
    main()
