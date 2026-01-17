from typing import Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm
from pathlib import Path
from utils import printing as prt
from utils import reportify as rprt

def print_divider(text: str):
    def decorator(func_name: str):
        def wrapper(*args, **kwargs):
            print(f"\n" + "=" *80)
            print(text)
            print(f"=" *80)
            func_name(*args, **kwargs)
        return wrapper
    return decorator    

def print_line(func_name: str):
    def wrapper(*args, **kwargs):
        print(f"\n" + "=" *80)
        func_name(*args, **kwargs)
    return wrapper


class DataInsights:

    @print_divider("INITIALIZING DATA INSIGHTS OBJECT")
    def __init__(self, path: str | Path, reportout_filepath: Path, pdf_report_title: str):
        self.df = pd.DataFrame()
                
        self.reportObj = rprt.reporter(report_filepath=reportout_filepath,
                                        author="Cagri Tanriover",
                                        title=pdf_report_title,
                                        subject="Stock ticker price regime classification")

        self.pdf_report_title = pdf_report_title

        try:
            self.df = pd.read_csv(path)
            print(f"Data loaded successfully: {len(self.df)} rows, {len(self.df.columns)} columns")
        except FileNotFoundError:
            print(f"File not found: {path}")
            raise
        except Exception as e:
            print(f"Error loading data: {e}")
            raise


    def end_operation(self):
        """
            Perform all required final operations internal to this class.
        """
        self.reportObj.generate_report()  # Finalize the report out as required.
        
        # Close all matplotlib figures to free up memory
        plt.close('all')
        print("All plot windows closed.")


    #@print_divider("BASIC DATAFRAME INFORMATION")
    @print_line
    def basic_info(self):
        """
            Display basic information about the dataframe self.df
        """
        self.reportObj.new_page(title=self.pdf_report_title)  # Start a new page in the pdf report created. Also add the report title here.

        self.reportObj.print(rprt.ReportDataType.HEADING_2, "BASIC DATAFRAME INFORMATION")  # Add a page title for the basic information section.

        self.reportObj.print(rprt.ReportDataType.BODY, f"Number of rows: {len(self.df)}")  # Print the paragraph to the console as well as the pdf report
        self.reportObj.print(rprt.ReportDataType.BODY, f"Number of columns: {len(self.df.columns)}")  # Print the paragraph to the console as well as the pdf report
        self.reportObj.print(rprt.ReportDataType.BODY, f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")  # Print the paragraph to the console as well as the pdf report

        self.reportObj.print(rprt.ReportDataType.BODY, "\nColumn Names:")
        for i, col in enumerate(self.df.columns, start=1):
            self.reportObj.print(rprt.ReportDataType.BODY, f"  {i}. {col}")  # Print the paragraph to the console as well as the pdf report


    @print_divider("MISSING VALUES ANALYSIS")
    def missing_values_analysis(self):
        """
            Display missing values in the dataframe self.df, if any.
        """
        
        missing_elements_per_column = self.df.isnull().sum()  # as pandas Series

        missing_record_df = pd.DataFrame({
            'Columns': missing_elements_per_column.index,
            'Missing Elements': missing_elements_per_column.values,
            'Percentage Missing': missing_elements_per_column.values.sum() *100 / self.df.size
        })
    
        # Optionally sort rows in according to "Missing Elements" in ascending order.
        missing_record_df = missing_record_df.loc[missing_record_df['Missing Elements'] > 0].sort_values('Missing Elements', ascending=True)

        self.reportObj.open_new_page(page_title="MISSING VALUES ANALYSIS")  # Add an empty page in the pdf report, and add the page title to the page.

        if len(missing_record_df):
            # There is at least one row with missing elements
            self.reportObj.print(rprt.ReportDataType.BODY, f"Found missing values in dataset!")  # Print the paragraph to the console as well as the pdf report

            print(f"{self.df.to_string(index=False)}")
 
            self.reportObj.print_dataframe_as_table(missing_record_df)
        else:
            # Following will be displayed on the console as well.
            self.reportObj.print(rprt.ReportDataType.BODY, f"There are no missing elements in the dataset !!")  # Print the paragraph to the console as well as the pdf report


    @print_divider("DATA TYPES SUMMARY")
    def data_types_summary(self):
        """
            Provide a summary of data types and their distributions.
        """

        numeric_columns = self.df.select_dtypes(include=["number"]).columns.tolist()
        categorical_columns = self.df.select_dtypes(include=["object"]).columns.tolist()
        datetime_columns = self.df.select_dtypes(include=["datetime", "datetime64"]).columns.tolist()

        self.reportObj.open_new_page(page_title="DATA TYPES SUMMARY")  # Add an empty page in the pdf report, and add the page title to the page.

        if len(numeric_columns) > 0:
            message = f"Found {len(numeric_columns)} numeric columns in dataset:"
            self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report

            for i, col_name in enumerate(numeric_columns):
                #print(f"{i} - {col_name}")
                message = f"{i} - {col_name}"
                self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report
        else:
            print("No numerical data found in dataset")
            message = "No numerical data found in dataset"
            self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report

        message="\n\n"
        self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report

        if len(categorical_columns) > 0: 
            #print(f"\nFound {len(categorical_columns)} categorical columns in dataset:")
            message = f"Found {len(categorical_columns)} categorical columns in dataset:"
            self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report
            for i, col_name in enumerate(categorical_columns):
                #print(f"{i} - {col_name}")
                message = f"{i} - {col_name}"
                self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report
        else:
            message = "No categorical data found in dataset"
            self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report

        message="\n\n"
        self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report

        if len(datetime_columns) > 0:
            message = f"Found {len(datetime_columns)} datetime columns in dataset:"
            self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report
            for i, col_name in enumerate(datetime_columns):
                #print(f"{i} - {col_name}")
                message = f"{i} - {col_name}"
                self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report
        else:
            message = "No datetime data found in dataset"
            self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report


    @print_divider("NUMERIC COLUMNS STATISTICS")
    def numeric_summary(self):
        """
        Generate descriptive statistics for numeric columns.
        """
        
        numeric_data_exists = lambda df: True if  len(df.select_dtypes(include=['number']).columns.tolist()) > 0 else False

        self.reportObj.open_new_page(page_title="NUMERIC COLUMNS STATISTICS")  # Add an empty page in the pdf report, and add the page title to the page.

        if numeric_data_exists(self.df):
            summary = self.df.describe(include='number')  # general statistics in summary data frame

            for each_column in summary.columns:
                # Creating, median, skew and kurtosis row indices in summary dataframe as we compute below.
                summary.loc['median', each_column] = self.df[each_column].median()
                summary.loc['skew', each_column] = self.df[each_column].skew()
                summary.loc['kurtosis', each_column] = self.df[each_column].kurtosis()

            #print(f"{summary.to_string()}")  # display full summary as text
            prt.print_dataframe(summary)
            self.reportObj.print_dataframe_as_table(summary)  # print into pdf report as well
        else:
            message = "No numeric data exists in dataset..."
            self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report

    #@print_divider("NUMERIC COLUMNS DISTRIBUTION PLOTS")
    @print_line
    def numeric_distributions(self, figsize: tuple = (10, 6), bins: int = 30, kde: bool = True, hspace: float = 0.3, save_path: Path | str | None = None, display_plots: bool = False):
        """
        Plot the distribution of each numeric column in the dataset.
        
        Args:
            figsize: Figure size tuple (width, height) for each subplot
            bins: Number of bins for histogram
            kde: Whether to overlay Kernel Density Estimation (KDE) plot
            hspace: Height space between subplots (default: 0.3). Increase for more spacing.
            save_path: Optional filepath to save the plot. Supports .png, .pdf, .jpg, .svg formats.
            display_plots: Whether to display the plots interactively. (do not display by default)
        """
        
        numeric_columns = self.df.select_dtypes(include=["number"]).columns.tolist()
        
        if len(numeric_columns) == 0:
            print("No numerical data found in dataset...")
            return
        
        print(f"Plotting distributions for {len(numeric_columns)} numeric column(s):")
        for i, col_name in enumerate(numeric_columns):
            print(f"  {i+1}. {col_name}")
        
        # Calculate grid dimensions for subplots
        n_cols = len(numeric_columns)
        
        # Create subplots - one for each numeric column
        fig, axes = plt.subplots(n_cols, 1, figsize=(figsize[0], figsize[1] * n_cols))
        
        # Handle single column case (axes won't be iterable)
        if n_cols == 1:
            axes = [axes]
        
        for idx, col in enumerate(numeric_columns):
            ax = axes[idx]
            
            # Remove missing values for plotting
            data = self.df[col].dropna()
            
            if len(data) == 0:
                ax.text(0.5, 0.5, f'No data available for {col}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{col} - Distribution (No Data)')
                continue
            
            # Plot histogram
            ax.hist(data, bins=bins, density=False, alpha=0.7, edgecolor='black', 
                   color='steelblue', label='Histogram')
            
            # Add KDE if requested
            if kde:
                try:
                    from scipy import stats
                    # Create KDE
                    kde_data = stats.gaussian_kde(data)
                    x_range = data.min(), data.max()
                    x_values = np.linspace(x_range[0], x_range[1], 200)
                    kde_values = kde_data(x_values)
                    
                    # Scale KDE to match histogram scale
                    hist_counts, _, _ = ax.hist(data, bins=bins, alpha=0)
                    max_hist = hist_counts.max()
                    max_kde = kde_values.max()
                    if max_kde > 0:
                        scaled_kde = kde_values * (max_hist / max_kde)
                        ax_twin = ax.twinx()
                        ax_twin.plot(x_values, scaled_kde, 'r-', linewidth=2, label='KDE')
                        ax_twin.set_ylabel('Density', color='r')
                        ax_twin.tick_params(axis='y', labelcolor='r')
                except ImportError:
                    print(f"  Warning: scipy not available, skipping KDE for {col}")
                except Exception as e:
                    print(f"  Warning: Could not plot KDE for {col}: {e}")
            
            # Add statistics to plot
            mean_val = data.mean()
            median_val = data.median()
            std_val = data.std()
            
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
            
            ax.set_title(f'{col} - Distribution (n={len(data)})', fontsize=12, fontweight='bold')
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--')
        
        # Adjust spacing between subplots and layout
        plt.tight_layout()
        plt.subplots_adjust(hspace=hspace)
        
        # Save the plot if filepath is provided
        if save_path is not None:
            save_path = Path(save_path)
            # Create directory if it doesn't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)
            # Save the figure
            fig.savefig(str(save_path), dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")
            # Add the saved plot filepath to the report
            self.reportObj.open_new_page(page_title="NUMERIC COLUMNS DISTRIBUTION PLOTS")  # Add an empty page in the pdf report, and add the page title to the page.
            self.reportObj.print_image(save_path)
        
        if display_plots == True:
            # Enable interactive mode for non-blocking display
            # Note: Figure remains open until script terminates - no plt.close() call
            self._enable_interactive_plots()
            print(f"\nDistribution plots displayed for {len(numeric_columns)} numeric column(s).")
        


    @print_divider("CATEGORICAL COLUMNS STATISTICS")
    def categorical_summary(self):
        """
        Analyze categorical columns including unique values and frequencies.
        """
        N_MAX = 10  # maximum number of unique items in a column allowed to be displayed in this method.

        categorical_columns_list = self.df.select_dtypes(include=['object']).columns.to_list()

        self.reportObj.open_new_page(page_title="CATEGORICAL COLUMNS STATISTICS")  # Add an empty page in the pdf report, and add the page title to the page.

        if len(categorical_columns_list):
            # There are categorical columns
            #print(f"High level summary of categorical columns:")
            #prt.print_dataframe(self.df.describe(include='object'))
            message = f"High level summary of categorical columns:\n\n"
            self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report
            self.reportObj.print_dataframe_as_table(self.df.describe(include='object'))  # print into pdf report as well
            self.reportObj.print(rprt.ReportDataType.BODY, "\n\n")  # Print the paragraph to the console as well as the pdf report

            # Next display number of unique items and their occurence frequency (where manageable) for each categorical column.
            for col in categorical_columns_list:
                print(f"\n" + "~"*80)
                #print(f"Categorical feature (column): {col}")
                message = f"Categorical feature (column): {col}\n\n"
                self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report

                n_unique_items = self.df[col].nunique()
                print(f"Number of unique items: {n_unique_items}")

                if n_unique_items <= N_MAX:
                    unique_dict = {item: count for item, count in self.df[col].value_counts().items()}
                    printPd = pd.Series(unique_dict).reset_index()  # Convert series to dataframe
                    printPd.columns = ['Item', 'Count']  # Assign custom column names to enw dataframe              
                    prt.print_dataframe(printPd, show_index=False)  # Not showing enumerated indices as they are not informative
                    self.reportObj.print_dataframe_as_table(printPd)  # print into pdf report as well
                    self.reportObj.print(rprt.ReportDataType.BODY, "\n\n")  # Print the paragraph to the console as well as the pdf report
                else:
                    #print(f"\nNumber of unique items > {N_MAX}. Skipping item listing...")
                    message = f"Number of unique items > {N_MAX}. Skipping item listing..."
                    self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report

                # What is the most frequent item in each categorical column?
                most_frequent_col_items = self.df[col].mode().to_list()
                if len(most_frequent_col_items) <= N_MAX:
                    print(f"\nHighest frequency items:")
                    message = f"Highest frequency items:\n\n"
                    self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report
                    itemDict = {item:self.df[col].value_counts()[item] for item in most_frequent_col_items}
                    printPd = pd.Series(itemDict).reset_index()
                    printPd.columns = ['Item', 'Count']
                    prt.print_dataframe(printPd, show_index=False)
                    self.reportObj.print_dataframe_as_table(printPd)  # print into pdf report as well
                else:
                    #print(f"\nThere are >{N_MAX} items at high frequency. Skipping item listing...")
                    message = f"There are >{N_MAX} items at high frequency. Skipping item listing..."
                    self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report

        else:
            #print(f"No categorical data exists in dataset...")
            message = "No categorical data exists in dataset..."
            self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report


    @print_divider("NUMERICAL STATISTICS FOR CATEGORICAL COLUMNS")
    def numerical_statistics_for_categorical_columns(self):
        """
        Generate descriptive statistics for each categorical column in the dataset.
        """

        categorical_columns = self.df.select_dtypes(include=['object']).columns.to_list()
        if len(categorical_columns) == 0:
            print("No categorical data found in dataset...")
            return

        self.reportObj.open_new_page(page_title="NUMERICAL STATISTICS FOR CATEGORICAL COLUMNS")  # Add an empty page in the pdf report, and add the page title to the page.

        for each_category in categorical_columns:
            ##print(f"Generating descriptive statistics for [{each_category}] categorical column:")
            message = f"Generating descriptive statistics for [{each_category}] categorical column:\n\n"
            self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report

            # Get unique items in each_category column to iterate as needed
            unique_items = self.df[each_category].unique()

            for each_item in unique_items:
                #print(f" Generating statistics for item [{each_item}] in [{each_category}] column")
                message = f" Generating statistics for item [{each_item}] in [{each_category}] column\n\n"
                self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report

                _subdf = self.df[self.df[each_category] == each_item]
                _subdf_summary = _subdf.describe(include='number')

                for each_column in _subdf_summary:
                    _subdf_summary.loc['median', each_column] = _subdf[each_column].median()
                    _subdf_summary.loc['skew', each_column] = _subdf[each_column].skew()
                    _subdf_summary.loc['kurtosis', each_column] = _subdf[each_column].kurtosis()
 
                prt.print_dataframe(_subdf_summary)
                self.reportObj.print_dataframe_as_table(_subdf_summary)  # print into pdf report as well
                self.reportObj.print(rprt.ReportDataType.BODY, "\n\n")  # Print the paragraph to the console as well as the pdf report

    @print_divider("CORRELATION ANALYSIS (applicable to the numeric columns only)")
    def correlation_analysis(self):
        """
        Analyze correlations between numeric variables using Pearson and Spearman methods.
        Uses only the original numeric columns from the dataset.
        """
        
        numeric_columns = self.df.select_dtypes(include=["number"]).columns.tolist()
        
        self.reportObj.open_new_page(page_title="CORRELATION ANALYSIS (for numerical columns only)")  # Add an empty page in the pdf report, and add the page title to the page.

        if len(numeric_columns) == 0:
            #print("No numerical data found in dataset...")
            message = "No numerical data found in dataset..."
            self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report
            return
        
        if len(numeric_columns) < 2:
            #print(f"Found only {len(numeric_columns)} numeric column(s). At least 2 numeric columns are required for correlation analysis.")
            message = f"Found only {len(numeric_columns)} numeric column(s). At least 2 numeric columns are required for correlation analysis."
            self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report
            return
        
        #print(f"Found {len(numeric_columns)} numeric columns in dataset:")
        message = f"Found {len(numeric_columns)} numeric columns in dataset:\n\n"
        self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report
        for i, col_name in enumerate(numeric_columns):
            #print(f"  {i+1}. {col_name}")
            message = f"  {i+1}. {col_name}\n"
            self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report          
        self.reportObj.print(rprt.ReportDataType.BODY, "\n\n")  # Print the paragraph to the console as well as the pdf report
        
        # Select only the original numeric columns
        numeric_df = self.df[numeric_columns]
        
        # Pearson Correlation
        #print(f"\nPearson Correlation Matrix:")
        message = f"Pearson Correlation Matrix:\n\n"
        self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report

        pearson_correlation_matrix = numeric_df.corr(method='pearson')
        #print(f"{pearson_correlation_matrix.to_string()}")
        prt.print_dataframe(pearson_correlation_matrix, justify_numeric="center")
        self.reportObj.print_dataframe_as_table(pearson_correlation_matrix)  # print into pdf report as well
        self.reportObj.print(rprt.ReportDataType.BODY, "\n\n")  # Print the paragraph to the console as well as the pdf report

        # Find strong Pearson correlations
        #print(f"\nStrong Pearson Correlations Criterion: |r| > 0.5")
        message = f"Strong Pearson Correlations Criterion: |r| > 0.5\n\n"
        self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report

        strong_pearson_corrs = []
        for i in range(len(pearson_correlation_matrix.columns)):
            for j in range(i+1, len(pearson_correlation_matrix.columns)):
                corr_val = pearson_correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    col1 = pearson_correlation_matrix.columns[i]
                    col2 = pearson_correlation_matrix.columns[j]
                    strong_pearson_corrs.append((col1, col2, corr_val))
        
        if len(strong_pearson_corrs) > 0:
            for col1, col2, corr_val in strong_pearson_corrs:
                #print(f"  {col1} ↔ {col2}: {corr_val:.3f}")
                message = f"  {col1} ↔ {col2}: {corr_val:.3f}\n"
                self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report
        else:
            #print("\tNo strong Pearson correlations found !")
            message = f"\tNo strong Pearson correlations found !\n"
            self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report

        self.reportObj.print(rprt.ReportDataType.BODY, "\n\n")  # Print the paragraph to the console as well as the pdf report

        # Spearman Correlation
        #print(f"\nSpearman Correlation Matrix:")
        message = f"Spearman Correlation Matrix:\n\n"
        spearman_correlation_matrix = numeric_df.corr(method='spearman')
        #print(f"{spearman_correlation_matrix.to_string()}")
        prt.print_dataframe(spearman_correlation_matrix, justify_numeric="center")
        self.reportObj.print_dataframe_as_table(spearman_correlation_matrix)  # print into pdf report as well
        self.reportObj.print(rprt.ReportDataType.BODY, "\n\n")  # Print the paragraph to the console as well as the pdf report

        # Find strong Spearman correlations
        #print(f"\nStrong Spearman Correlations Criterion: |r| > 0.5:")
        message = f"Strong Spearman Correlations Criterion: |r| > 0.5\n\n"
        self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report
        strong_spearman_corrs = []
        for i in range(len(spearman_correlation_matrix.columns)):
            for j in range(i+1, len(spearman_correlation_matrix.columns)):
                corr_val = spearman_correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    col1 = spearman_correlation_matrix.columns[i]
                    col2 = spearman_correlation_matrix.columns[j]
                    strong_spearman_corrs.append((col1, col2, corr_val))
        
        if len(strong_spearman_corrs) > 0:
            for col1, col2, corr_val in strong_spearman_corrs:
                #print(f"  {col1} ↔ {col2}: {corr_val:.3f}")
                message = f"  {col1} ↔ {col2}: {corr_val:.3f}\n"
                self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report
        else:
            #print("\tNo strong Spearman correlations found !")
            message = f"\tNo strong Spearman correlations found !\n"
            self.reportObj.print(rprt.ReportDataType.BODY, message)  # Print the paragraph to the console as well as the pdf report


    @print_divider("CONFUSION RISK ANALYSIS (to see separability of classes in the dataset)")
    def confusion_risk_analysis(self, features_dict: dict = None, class_column: str = 'gt', save_folder: str = None, display_plots: bool = False):
        """
        This analysis answers the question:
        Which classes overlap in feature space, such that a reasonable classifier might confuse them?
        
        """
        
        # Method 1 - Scatter plots for each pair of features to see if there is any overlap in the feature space.
        # This method shows joint geometry of the features in the feature space.
        self._scatter_plot_analysis(features_dict, class_column, save_folder, display_plots)

        # Method 2 - Kernel Density Estimation (KDE) for each feature to see if there is any overlap in the feature space.
        # This method complements Method 1 and shows:
        #- Marginal separability of the features
        # - where thresholds cut through probability mass
        # - which classes dominate specific value ranges
        self._kde_plot_analysis(features_dict, class_column, save_folder, display_feature_thresholds=True, display_plots=display_plots)

        # Method 3 - Mahalanobis distance analysis to see the level of overlap in the feature space.
        # This will tell us how separable the classes are in feature space, before training a model.
        # In other words, this distance is a measure of how far apart the class centers are in terms of the 
        # measured units of typical within-class noise. (Euclidean distance is not a good fit for this purpose because it is not scale-invariant)
        # We will get one distance per class pair.
        # The calculated distances between pairs of classes will also be comparable.
        # The distance will reflect the signal vs noise, not class size or regime frequency.
        distance_matrix_df = self._mahalanobis_distance_analysis(features_dict, class_column, save_folder, display_plots=display_plots)

        # Once we calculate the Mahalanobis distance matrix using the pooled covariance matrix,
        # we can use it to calculate the expected pairwise separability accuracy of classes under the following assumptions:
        # 1 - Each class has a normal distribution in the feature space. (still holds as an approximation in mild skewness and kurtosis of features)
        # 2 - All classes have the same covariance matrix and only differ in their means. (slight differences still allow an approximation)
        # 3 - All classes are assumed equally likely. (rare classes will be underestimated in terms of accuracy)
        # 4 - Classifier is assumed to know the true means and covariance (i.e., best classifier case) Observed accuracy may vary depending on the classifier used.
        # 5 - Correct feature scaling and whitening is used in distance calculation (Mahalanobis distance calculation already ensures tis)
        # 6 - Covariance is estimated using sufficient number of samples. (small sample size in classes will inject noise, which makes the distances less reliable)
        # 7 - Because only pairwise accuracy is calculated, multiclass accuracy will be overestimated.
        # Due to the many assumptions above, the accuraacy results should be interpreted as an upper-bound only!
        self._calculate_pairwise_accuracy_estimations(distance_matrix_df)        



    def _calculate_pairwise_accuracy_estimations(self, distance_matrix_df: pd.DataFrame):
        """
        Calculate the expected pairwise separability accuracy of classes under the following assumptions:
        1 - Each class has a normal distribution in the feature space. (still holds as an approximation in mild skewness and kurtosis of features)
        2 - All classes have the same covariance matrix and only differ in their means. (slight differences still allow an approximation)
        3 - All classes are assumed equally likely. (rare classes will be underestimated in terms of accuracy)
        4 - Classifier is assumed to know the true means and covariance (i.e., best classifier case) Observed accuracy may vary depending on the classifier used.
        5 - Correct feature scaling and whitening is used in distance calculation (Mahalanobis distance calculation already ensures tis)
        6 - Covariance is estimated using sufficient number of samples. (small sample size in classes will inject noise, which makes the distances less reliable)
        7 - Because only pairwise accuracy is calculated, multiclass accuracy will be overestimated.
        Due to the many assumptions above, the accuraacy results should be interpreted as an upper-bound only!

        Args:
            distance_matrix_df: DataFrame containing the Mahalanobis distance matrix

        Returns:
            None
        """

        labels = list(distance_matrix_df.columns)  # index is the same as the matrix is square and symmetric

        accuracy_matrix = pd.DataFrame(norm.cdf(distance_matrix_df / 2.0), index=labels, columns=labels)  # gives the probability of successfully separating a class on the row from a class on the column

        # Let's initialize the diagonal to 1.0 to indicate perfect accuracy for each class to itself
        for i in range(len(labels)):
            accuracy_matrix.iloc[i, i] = 1.0

        # Round probabilities to 3 decimal points
        accuracy_matrix = accuracy_matrix.round(3)

        # Print the results in a nice table format.
        print(f"Pairwise separability accuracy estimations:")
        prt.print_dataframe(accuracy_matrix, justify_numeric="center")   # Print the pairwise separability accuracy estimations as a nice table.

        # Add result to the pdf report file too
        self.reportObj.open_new_page(page_title="PAIRWISE SEPARABILITY ACCURACY ESTIMATIONS")  # Add an empty page in the pdf report, and add the page title to the page.
        self.reportObj.print_dataframe_as_table(accuracy_matrix)


    def _mahalanobis_distance_analysis(self, features_dict: dict, 
    class_column: str, 
    save_folder: str, 
    display_plots: bool = False,
    standardize: bool = True,
    shrinkage: float = 1e-3,   # diagonal regularization for numerical stability prior to matrix inversion
    ) -> pd.DataFrame:
        """
        Calculate the Mahalanobis distance between each pair of classes for the selected features in the feature space.
        Mahalanobis distance analysis to see the level of overlap in the feature space.
        This will tell us how separable the classes are in feature space, before training a model.
        In other words, this distance is a measure of how far apart the class centers are in terms of the 
        measured units of typical within-class noise. (Euclidean distance is not a good fit for this purpose because it is not scale-invariant)
        - We will get one distance per class pair.
        - The calculated distances between pairs of classes will also be comparable.
        - The distance will reflect the signal vs noise, not class size or regime frequency.

        Args:
            features_dict: Dictionary of features to analyze
            class_column: Name of the column containing the class labels
            save_folder: Folder to save the plots
            display_plots: Whether to display the plots interactively
        """

        all_features = list(features_dict.keys())
        # Only use the features requested by the user         
        features_to_plot = [feature for feature in all_features if features_dict[feature]['use_in_mahalanobis_distance'] == True]

        if len(features_to_plot) < 2:
            raise ValueError("At least 2 FEATURES are required for Mahalanobis distance analysis.")
        
        labels = list(self.df[class_column].unique())
        if len(labels) < 2:
            raise ValueError("At least 2 CLASSES are required for Mahalanobis distance analysis.")

        class_pairs = []
        # Mahalanobis distance will be calculated for each pair of classes.
        for first_label in labels:
            for second_label in labels:
                if first_label != second_label and (first_label, second_label) not in class_pairs and (second_label, first_label) not in class_pairs:
                    class_pairs.append((first_label, second_label))

        df = self.df[features_to_plot + [class_column]]  # portion of interest in our dataset as new dataframe
        X = df[features_to_plot].to_numpy(dtype=float)  # create numpy array with relevant features
        y = df[class_column].to_numpy()  # create numpy array with class labels

        if standardize == True:
            X_mean = X.mean(axis=0)  # mean across all rows for each feature (i.e., column)
            X_std = X.std(axis=0, ddof = 0.0)  # standard deviation across all rows for each feature (i.e., column)
            X_std[X_std == 0] = 1.0  # set to 1.0 to avoid division by zero (result of x-mean = 0 for all values in a column)
            X = (X - X_mean) / X_std  # standardize the features
        
        # Calculate pooled covariance matrix next.
        #C_pooled = sum((n_k - 1) * C_k)) / sum((n_k - 1)), where k represents each of the two classes
        C_pool_numerator = np.zeros((len(features_to_plot), len(features_to_plot)), dtype=float)
        C_pool_denominator = 0
        D = pd.DataFrame(np.zeros([len(labels), len(labels)]), index=labels, columns=labels, dtype=float)  # Mahalanobis distance matrix includes all labels/classes in the dataset.

        for cls_pair in class_pairs:
            centroids = {}

            for each_class in cls_pair:
                Xk = X[y== each_class]  # get all features for the correct classes only
                centroids[each_class] = Xk.mean(axis=0)  # centroid of each feature (i.e., mean) for each class captured here.

                n = Xk.shape[0]  # number of samples in the class
                Ck = np.cov(Xk, rowvar=False, ddof=1)  # within-class covariance matrix
                # Let's computer pool terms
                C_pool_numerator += (n - 1) * Ck  # Building the numerator term
                C_pool_denominator += (n - 1)  # Building the denominator term (i.e. the sum of degrees of freedom based on number of samples in each class)
        
            # Compute the pooled covariance matrix
            C_pooled = C_pool_numerator / C_pool_denominator

            # Apply shrinkage to the pooled covariance matrix
            # Shrinkage is applied to the diagonal elements of the pooled covariance matrix to improve numerical stability.
            # For better representation, shrinkage will also be scaled by the average variance of the features in the pooled covariance matrix.
            sum_of_diagonal_elements = np.trace(C_pooled)
            average_variance = sum_of_diagonal_elements / C_pooled.shape[0]

            if np.isfinite(sum_of_diagonal_elements) and sum_of_diagonal_elements > 0:
                # sum of variances is not np.nan, np.inf, -np.inf AND is positive
                boost_factor = shrinkage * average_variance  # Scale the original small shrinkage factor by the average of variances for better representation of data.
            else:
                boost_factor = shrinkage    # Do not scale the original small shrinkage factor by the average of variances.

            C_pooled = C_pooled + boost_factor * np.eye(C_pooled.shape[0])  # boost the diagonal elements of the pooled covariance matrix

            # Use pseudo-inverse for robustness
            C_pooled_inv = np.linalg.pinv(C_pooled)  # Taking the inverse of the covariance matrix.

            # Calculate Mahalanobis distance for cls_pair
            centroid_delta = centroids[cls_pair[0]] - centroids[cls_pair[1]]  # difference between the centroids of the two classes

            D.loc[cls_pair[0], cls_pair[1]] = float(np.sqrt(centroid_delta.T @ C_pooled_inv @ centroid_delta))  # Mahalanobis distance between the two classes
            # Also populate the reverse direction of the distance in the D matrix for consistency (diagonal elements are 0)
            D.loc[cls_pair[1], cls_pair[0]] = D.loc[cls_pair[0], cls_pair[1]]  # Mahalanobis distance is symmetric and the distance between cls_pair[1] and cls_pair[0] is the same as the distance between cls_pair[0] and cls_pair[1]  

        # Print the results in a nice table format.
        print(f"Mahalanobis distance matrix for features: {features_to_plot}:")
        prt.print_dataframe(D, justify_numeric="center")   # Print all Mahalanobis distances between all pairs of classes in the dataset as a nice table.

        # Update pdf report content
        self.reportObj.open_new_page(page_title="MAHALANOBIS DISTANCE MATRIX")  # Add an empty page in the pdf report, and add the page title to the page.
        self.reportObj.print_dataframe_as_table(D)

        return D


    def _kde_plot_analysis(self, features_dict: dict, class_column: str, save_folder: str, display_feature_thresholds: bool = True, display_plots: bool = False):
        """
        Plot the Kernel Density Estimation (KDE) for each feature to see if there is any overlap in the feature space.
        This method complements Method 1 and shows:
        - Marginal separability of the features
        - where thresholds cut through probability mass
        - which classes dominate specific value ranges
        """

        features_to_plot = list(features_dict.keys())
        labels = list(self.df[class_column].unique())

        for each_feature in features_to_plot:
            # Process one feature per loop iteration
            min_val = self.df[each_feature].min()
            max_val = self.df[each_feature].max()

            x_values = np.linspace(min_val, max_val, 400)  # 400 evenly spaced points between min and max

            label_pairs_to_process = []
            # Generate all possible label pairs to process
            for first_label in labels:
                for second_label in labels:
                    if first_label != second_label:
                        # pairing the label with itself is meaningless in this KDE analysis
                        if (first_label, second_label) not in label_pairs_to_process and (second_label, first_label) not in label_pairs_to_process:
                            # make sure there is no duplicate of labels with alternating order
                            label_pairs_to_process.append((first_label, second_label))

            for each_pair in label_pairs_to_process:
                # Process one label PAIR per loop iteration
                fig = plt.figure(figsize=(8, 6))  # new figure for each pair of labels

                # Handle the first label KDE first
                vals = self.df[self.df[class_column] == each_pair[0]][each_feature].values
                kde = gaussian_kde(vals)

                plt.plot(x_values, kde(x_values), label=each_pair[0])

                # Then Handle the second label KDE
                vals = self.df[self.df[class_column] == each_pair[1]][each_feature].values
                kde = gaussian_kde(vals)

                plt.plot(x_values, kde(x_values), label=each_pair[1])

                # Also plot any feature related thresholds for each_feature if they are defined in the features_dict

                if display_feature_thresholds == True:
                    if len(features_dict[each_feature]['thresholds']) > 0:
                        for idx, each_threshold in enumerate(features_dict[each_feature]['thresholds']):
                            # Use idx to select color from matplotlib's color cycle
                            plt.axvline(each_threshold[1], color=f'C{idx}', linestyle='--', linewidth=1, label=each_threshold[0])               
                
                plt.title(f'KDE: {each_feature} for {each_pair[0]} vs {each_pair[1]}')
                plt.xlabel(each_feature)
                plt.ylabel('Density')
                plt.legend(title='Labels', bbox_to_anchor=(1,1), loc='upper right', fontsize=9)
                plt.grid(True, alpha=0.3)

                # Save the plot if filepath is provided
                if save_folder is not None:
                    filename = f'confusion_risk_KDE_{each_feature}_{each_pair[0]}_vs_{each_pair[1]}.png'
                    self._save_plot(figure=fig, filename=filename, save_folder=save_folder, pdf_page_title="CONFUSION RISK - KDE PLOT")

                if display_plots == True:
                    # Enable interactive mode for non-blocking display
                    # Note: Figure remains open until script terminates - no plt.close() call
                    self._enable_interactive_plots()
                    print(f"Confusion risk KDE plot displayed for {each_feature}: {each_pair[0]} vs {each_pair[1]}")            


    def _scatter_plot_analysis(self, features_dict: dict, class_column: str, save_folder: str, display_plots: bool):
        """
        Scatter plots for each pair of features to see if there is any overlap in the feature space.
        This method shows joint geometry of the features in the feature space.
        
        Args:
            features_to_plot: List of features to plot
            class_column: Name of the column containing the class labels
            save_folder: Folder to save the plots
            display_plots: Whether to display the plots interactively
        """

        # Get unique classes in the class_column to iterate as needed
        labels = list(self.df[class_column].unique())  # class labels present in the dataset

        covered_features = []  # to avoid duplicating scatter plots for the same feature pair.
        features_to_plot = list(features_dict.keys())

        for x_col in features_to_plot:
            for y_col in features_to_plot:
                if x_col != y_col \
                and (x_col, y_col) not in covered_features \
                and (y_col, x_col) not in covered_features:
                    covered_features.append((x_col, y_col))    # Add the feature pair to the covered features list to avoid duplicating the scatter plot.
                    
                    # Create a new figure for each scatter plot
                    fig = plt.figure(figsize=(8, 6))

                    for each_label in labels:
                        _subdf = self.df[self.df[class_column] == each_label]  # extract a sub-dataframe for each_label
                        plt.scatter(_subdf[x_col], _subdf[y_col], s=12, alpha=0.35, label=each_label)  # automatic color for each label using matplotlib's built-in color map.

                    plt.xlabel(x_col)
                    plt.ylabel(y_col)
                    plt.title(f'Scatter: {x_col} vs {y_col}')
                    plt.legend(title='Labels', bbox_to_anchor=(1,1), loc='upper right', fontsize=9)
                    plt.grid(True, alpha=0.3)

                    # Save the plot if filepath is provided
                    if save_folder is not None:
                        filename = f'confusion_risk_SCATTER_{x_col}_vs_{y_col}.png'
                        self._save_plot(figure=fig, filename=filename, save_folder=save_folder, pdf_page_title="CONFUSION RISK - SCATTER PLOT")

                    if display_plots == True:
                        # Enable interactive mode for non-blocking display
                        # Note: Figure remains open until script terminates - no plt.close() call
                        self._enable_interactive_plots()
                        print(f"Confusion risk SCATTER plot displayed for [{x_col}] vs [{y_col}]")

                # else skip the plot to avoid duplication


    def _save_plot(self, figure: plt.figure, filename: str, save_folder: str, pdf_page_title: str = None):
                            # Save the plot if filepath is provided
        save_path = Path(save_folder)  / filename
        # Create directory if it doesn't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Save the figure
        figure.savefig(str(save_path), dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

        if pdf_page_title != None:
            # Add the saved plot filepath to the report
            self.reportObj.open_new_page(page_title=pdf_page_title)
            self.reportObj.print_image(save_path)
    

    def _enable_interactive_plots(self):
        plt.ion()
        plt.show(block=False)
        # Give matplotlib time to render the plot window
        plt.pause(0.1)  # Brief pause to ensure plot window is rendered
        plt.draw()  # Force a draw to update the display