from typing import Any
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm
from sklearn.metrics import classification_report # accuracy_score, precision_score, recall_score, f1_score
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
        self.reportObj.new_page(title=self.pdf_report_title, enable_write=True)  # Start a new page in the pdf report created. Also add the report title here.

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

        self.reportObj.open_new_page(page_title="DATA TYPES SUMMARY", enable_write=True)  # Add an empty page in the pdf report, and add the page title to the page.

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

        self.reportObj.open_new_page(page_title="NUMERIC COLUMNS STATISTICS", enable_write=True)  # Add an empty page in the pdf report, and add the page title to the page.

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
            self.reportObj.open_new_page(page_title="NUMERIC COLUMNS DISTRIBUTION PLOTS", enable_write=True)  # Add an empty page in the pdf report, and add the page title to the page.
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

        self.reportObj.open_new_page(page_title="CATEGORICAL COLUMNS STATISTICS", enable_write=True)  # Add an empty page in the pdf report, and add the page title to the page.

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

        self.reportObj.open_new_page(page_title="NUMERICAL STATISTICS FOR CATEGORICAL COLUMNS", enable_write=True)  # Add an empty page in the pdf report, and add the page title to the page.

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
        
        self.reportObj.open_new_page(page_title="CORRELATION ANALYSIS (for numerical columns only)", enable_write=True)  # Add an empty page in the pdf report, and add the page title to the page.

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
        self._scatter_plot_analysis(features_dict, class_column, save_folder, display_plots, enable_pdf_write=True)

        # Method 2 - Kernel Density Estimation (KDE) for each feature to see if there is any overlap in the feature space.
        # This method complements Method 1 and shows:
        #- Marginal separability of the features
        # - where thresholds cut through probability mass
        # - which classes dominate specific value ranges
        self._kde_plot_analysis(features_dict, class_column, save_folder, display_feature_thresholds=True, display_plots=display_plots, enable_pdf_write=True)

        # Method 3 - Mahalanobis distance analysis to see the level of overlap in the feature space.
        # This will tell us how separable the classes are in feature space, before training a model.
        # In other words, this distance is a measure of how far apart the class centers are in terms of the 
        # measured units of typical within-class noise. (Euclidean distance is not a good fit for this purpose because it is not scale-invariant)
        # We will get one distance per class pair.
        # The calculated distances between pairs of classes will also be comparable.
        # The distance will reflect the signal vs noise, not class size or regime frequency.
        distance_matrix_df = self._mahalanobis_distance_analysis(features_dict, class_column, save_folder, display_plots=display_plots, enable_pdf_write=True)

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
        self._calculate_pairwise_accuracy_estimations(distance_matrix_df, enable_pdf_write=True)        



    def _calculate_pairwise_accuracy_estimations(self, distance_matrix_df: pd.DataFrame, enable_pdf_write: bool = True):
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
        self.reportObj.open_new_page(page_title="PAIRWISE SEPARABILITY ACCURACY ESTIMATIONS", enable_write=enable_pdf_write)  # Add an empty page in the pdf report, and add the page title to the page.
        self.reportObj.print_dataframe_as_table(accuracy_matrix)


    def _mahalanobis_distance_analysis(self, features_dict: dict, 
    class_column: str, 
    save_folder: str, 
    display_plots: bool = False,
    standardize: bool = True,
    shrinkage: float = 1e-3,   # diagonal regularization for numerical stability prior to matrix inversion
    enable_pdf_write: bool = True
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
        self.reportObj.open_new_page(page_title="MAHALANOBIS DISTANCE MATRIX", enable_write=enable_pdf_write)  # Add an empty page in the pdf report, and add the page title to the page.
        self.reportObj.print_dataframe_as_table(D)

        return D


    def _kde_plot_analysis(self, features_dict: dict,
        class_column: str,
        save_folder: str,
        display_feature_thresholds: bool = True,
        display_plots: bool = False,
        enable_pdf_write: bool = True
    ):
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
                    self._save_plot(figure=fig,
                                    filename=filename,save_folder=save_folder,
                                    pdf_page_title="CONFUSION RISK - KDE PLOT",
                                    enable_pdf_write=enable_pdf_write
                                    )

                if display_plots == True:
                    # Enable interactive mode for non-blocking display
                    # Note: Figure remains open until script terminates - no plt.close() call
                    self._enable_interactive_plots()
                    print(f"Confusion risk KDE plot displayed for {each_feature}: {each_pair[0]} vs {each_pair[1]}")            


    def _scatter_plot_analysis(self, features_dict: dict, class_column: str, save_folder: str, display_plots: bool, enable_pdf_write: bool = True):
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
                        self._save_plot(figure=fig,
                                        filename=filename,
                                        save_folder=save_folder,
                                        pdf_page_title="CONFUSION RISK - SCATTER PLOT",
                                        enable_pdf_write=enable_pdf_write
                                        )

                    if display_plots == True:
                        # Enable interactive mode for non-blocking display
                        # Note: Figure remains open until script terminates - no plt.close() call
                        self._enable_interactive_plots()
                        print(f"Confusion risk SCATTER plot displayed for [{x_col}] vs [{y_col}]")

                # else skip the plot to avoid duplication


    def _save_plot(self, figure: plt.figure, filename: str, save_folder: str, pdf_page_title: str = None, enable_pdf_write: bool = True):
                            # Save the plot if filepath is provided
        save_path = Path(save_folder)  / filename
        # Create directory if it doesn't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Save the figure
        figure.savefig(str(save_path), dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

        if pdf_page_title != None:
            # Add the saved plot filepath to the report
            self.reportObj.open_new_page(page_title=pdf_page_title, enable_write=enable_pdf_write)
            self.reportObj.print_image(save_path)


    def rule_based_classifier_analysis(self, features_dict: dict, class_column: str, enable_pdf_write: bool = True):

        """
        This is a simple rule-based classifier that uses naive thresholds on the features to classify the dataset into classes.
        Ground truth labels are created based on carefully computed thresholds based on the features. Therefore, those computations
        are "oracle functions" that will always give 100% accurate label predictions. In this analysis we are trying to answer the following
        question:
        
        "How far can a naive, hand written heuristic get without knowing the (oracle) labeling logic?"

        You can include different types of naive classifiers here to evaluate.

        """

        # Classifier 1 - Naive percentile threshold classifier
        self.percentile_threshold_classifier_evaluation(features_dict, class_column, enable_pdf_write)


        print("Rule-based classifier analysis completed")
        

    # Classifier 1 - Naive percentile threshold classifier
    def percentile_threshold_classifier_evaluation(self, features_dict, class_column, enable_pdf_write):

        thresholds = {
            'up_th': 0.0,  # lower bound for strong upward trend
            'down_th': 0.0,  # upper bound for strong downward trend
            'flat_th': 0.0,  # upper bound for flat trend (abs(slope) perspective)
            'strong_ts_th': 0.0,  # lower bound for strong trend strength (ratio)
            'zcr_th': 0.0,  # lower bound for high oscillation zero crossing rate
            'vol_lo_th': 0.0,  # upper bound for low volatility
            'vol_hi_th': 0.0  # lower bound for high volatility
        }

        # Naive percentile based threshold computations follow. These are likely to be picked up by eyeballing humans without detailed analysis.
        thresholds['up_th'] = self.df['slope'].quantile(0.80)
        thresholds['down_th'] = self.df['slope'].quantile(0.20)
        thresholds['flat_th'] = abs(self.df['slope']).quantile(0.20)  # small mid band around zero slope to catch flat trends (note the abs() operator)
        thresholds['strong_ts_th'] = self.df['trend_strength'].quantile(0.80)
        thresholds['zcr_th'] = self.df['zcr'].quantile(0.70)
        thresholds['vol_lo_th'] = self.df['volatility'].quantile(0.20)
        thresholds['vol_hi_th'] = self.df['volatility'].quantile(0.80)

        # Next let's use the thresholds to predict the label for each row in our dataset.
        # Build predictions in a list and assign once to avoid DataFrame fragmentation.
        labelsDf = pd.DataFrame(self.df[class_column].copy())
        pred_list = []

        for each_row in self.df.index:
            # Identify the bool flags for each row
            is_flat = abs(self.df.loc[each_row, 'slope']) <= thresholds['flat_th']
            is_low_vol = self.df.loc[each_row, 'volatility'] <= thresholds['vol_lo_th']  # not volatile
            is_ok_vol = self.df.loc[each_row, 'volatility'] <= thresholds['vol_hi_th']  # not excessively volatile
            is_strong_ts = self.df.loc[each_row, 'trend_strength'] >= thresholds['strong_ts_th']  # strong trend strength
            is_up_slope = self.df.loc[each_row, 'slope'] >= thresholds['up_th']  # upward slope
            is_down_slope = self.df.loc[each_row, 'slope'] <= thresholds['down_th']  # downward slope
            is_high_zcr = self.df.loc[each_row, 'zcr'] >= thresholds['zcr_th']  # high zero crossing rate

            # Next let's check for each class case in a particular order as below, and assign the predicted label.
            if is_flat and is_low_vol:
                # CHECK 1 - STATIONARY CLASS
                pred_list.append('STATIONARY')
            elif is_up_slope and is_strong_ts and is_ok_vol and (not is_high_zcr):
                # CHECK 2 - TREND_UP CLASS
                pred_list.append('TREND_UP')
            elif is_down_slope and is_strong_ts and is_ok_vol and (not is_high_zcr):
                # CHECK 3 - TREND_DOWN CLASS
                pred_list.append('TREND_DOWN')
            elif is_high_zcr:
                # CHECK 4 - OSCILLATING CLASS
                pred_list.append('OSCILLATING')
            else:
                # CHECK 5 - OTHER CLASS
                pred_list.append('OTHER')

        labelsDf['pred'] = pred_list  # To avoid fragmentation associated with per cell assignment to dataframe in a loop.

        # Performance evaluation of the naive classifier
        y_true = labelsDf[class_column]
        y_pred = labelsDf['pred']
        
        # Confusion matrix analysis (convention: rows = true class, columns = predicted class; matches sklearn.metrics.confusion_matrix)
        cm = pd.crosstab(y_true, y_pred)
        cm.index = [f"gt({idx})" for idx in cm.index]
        cm.columns = [f"pred({col})" for col in cm.columns]
        prt.print_dataframe(cm, justify_numeric="center")

        # Precision, Recall, F1-score per class (tabular form from classification_report)
        report_dict = classification_report(y_true, y_pred, output_dict=True)  # returns a dictionary, which is really helpful here!
        report_df = pd.DataFrame(report_dict).T
        # 'accuracy' is a scalar in the dict so that row has NaNs for precision/recall/f1
        # Therefore, it makes sense to print it outside the dataframe for clarity.
        acc_val = report_dict["accuracy"]
        report_df = report_df.drop(index=["accuracy"])  # removing row 'accuracy' from dataframe.

        report_df['support'] = report_df['support'].astype(int)  # convert the support column to integer type for pretty printing later

        # Report the findings next.
        message = f"Naive Percentile Threshold Classifier Performance Evaluation"
        print(f"{message}")
        prt.print_dataframe(report_df, justify_numeric="center")
        print(f"Accuracy: {acc_val:.2%}")

        self.reportObj.new_page(enable_write=enable_pdf_write)
        self.reportObj.print(rprt.ReportDataType.HEADING_2, message)
        self.reportObj.print_dataframe_as_table(report_df)
        self.reportObj.print(rprt.ReportDataType.BODY, f"\n\n")
        self.reportObj.print(rprt.ReportDataType.BODY, f"Accuracy: {acc_val:.2%}")

    
    def label_noise_analysis(self, features_dict: dict = None, class_column: str = 'gt', save_folder: str = 'plots'):

        # Robust sigma (scale) will be computed over the entire dataset for each feature. Then tolerance per feature will be calculated.
        tolerances = self._compute_tolerances(features_dict)

        # Generate ambiguity flags per sample in dataset
        ambiguityFlagsDf = self._generate_ambiguity_flags(features_dict, tolerances, class_column)

        # Calculate global boundary attribution for each ambiguity class
        globalBoundaryAttributionDf = self._calculate_global_boundary_attribution(ambiguityFlagsDf, class_column)

        # Calculate label boundary attribution for each ambiguity class
        labelBoundaryAttributionDf = self._calculate_label_boundary_attribution(ambiguityFlagsDf, class_column)

        # Update ambiguityFlagsDf with ambiguity metrics columns and calculate each for each row in the dataset
        # The new ambiguityFlagsDf will only include the columns required for the rest of the analysis.
        ambiguityFlagsDf = self._add_ambiguity_metrics(ambiguityFlagsDf, class_column)

        # First calculate ambiguity metrics for the entire dataset (nothing specific per label)
        globalMetricsDf = self._calculate_global_metrics(ambiguityFlagsDf, class_column)

        # Next, repeat the same for each label.
        labelSpecificMetricsDf = self._calculate_label_specific_metrics(ambiguityFlagsDf, class_column)

        # Add the analysis results to the report and the console output
        #####
        self.reportObj.new_page(enable_write=True)
        #####
        message = "Label Noise Analysis - Global Boundary Attribution"
        print(f"{message}")
        prt.print_dataframe(globalBoundaryAttributionDf, justify_numeric="center")
        self.reportObj.print(rprt.ReportDataType.HEADING_2, f"{message}")
        self.reportObj.print_dataframe_as_table(globalBoundaryAttributionDf)
        #####
        message = "Label Noise Analysis - Label Boundary Attribution"
        print(f"{message}")
        prt.print_dataframe(labelBoundaryAttributionDf, justify_numeric="center")
        self.reportObj.print(rprt.ReportDataType.HEADING_2, f"{message}")
        self.reportObj.print_dataframe_as_table(labelBoundaryAttributionDf)
        
        #####
        self.reportObj.new_page(enable_write=True)
        #####
        message = "Label Noise Analysis - Global Ambiguity Metrics"
        print(f"{message}")
        prt.print_dataframe(globalMetricsDf, justify_numeric="center")
        self.reportObj.print(rprt.ReportDataType.HEADING_2, f"{message}")
        self.reportObj.print_dataframe_as_table(globalMetricsDf)
        #####
        message = "Label Noise Analysis - Label Ambiguity Metrics"
        print(f"{message}")
        prt.print_dataframe(labelSpecificMetricsDf, justify_numeric="center")
        self.reportObj.print(rprt.ReportDataType.HEADING_2, f"{message}")
        self.reportObj.print_dataframe_as_table(labelSpecificMetricsDf)


    def _calculate_label_boundary_attribution(self, ambiguityFlagsDf: pd.DataFrame, class_column: str) -> pd.DataFrame:
        """
        Calculate boundary attribution for a GIVEN LABEL for each flag column in the ambiguityFlagsDf dataframe as follows:
        trigger_ratio = ratio of True values in the flag column to the total number of rows in the sub-dataframe for the GIVEN LABEL.

        This provides insights on "in what fraction of a specific label sample does a particular boundary look 'close'?"
        """

        labels = list(ambiguityFlagsDf[class_column].unique())  # all unique labels in the dataset
        boundaries = ambiguityFlagsDf.drop([class_column], axis=1).columns.tolist()  # get all columns except the class column

        labelBoundaryAttributionDf = pd.DataFrame(data=[[0.0] * len(labels)], index=boundaries, columns=labels)

        for each_label in labels:
            labelDf = ambiguityFlagsDf[ambiguityFlagsDf[class_column] == each_label]  # ambiguity flagssub-dataframe for each_label entry only.
            for each_boundary in boundaries:
                labelBoundaryAttributionDf.loc[each_boundary, each_label] = self._get_ambiguity_rate(labelDf, each_boundary)

        return labelBoundaryAttributionDf


    def _calculate_global_boundary_attribution(self, ambiguityFlagsDf: pd.DataFrame, class_column: str) -> pd.DataFrame:
        """
        Calculate global boundary attribution for each flag column in the ambiguityFlagsDf dataframe as follows:
        trigger_ratio = ratio of True values in the flag column to the total number of rows in the dataframe.

        This provides insights on "in what fraction of all samples does a particular boundary look 'close'?"
        """

        boundaries = ambiguityFlagsDf.drop([class_column], axis=1).columns.tolist()  # get all columns except the class column

        globalBoundaryAttributionDf = pd.DataFrame(data=[[0.0]], index=boundaries, columns=['Trigger Ratio'])

        for each_boundary in boundaries:
            globalBoundaryAttributionDf.loc[each_boundary, 'Trigger Ratio'] = self._get_ambiguity_rate(ambiguityFlagsDf, each_boundary)

        return globalBoundaryAttributionDf


    def _calculate_label_specific_metrics(self, ambiguityFlagsDf: pd.DataFrame, class_column: str) -> pd.DataFrame:
        """
        Following metrics will be calculated FOR EACH LABEL in the dataset:
        1 - 'Any Ambiguity per Sample': Ratio of True values in the 'amb_any' column to the total number of rows in the dataframe.
        2 - 'Multi Ambiguity per Sample': Ratio of True values in the 'amb_multi' column to the total number of rows in the dataframe.
        3 - 'Mean Near-Boundary Count per Sample': Ratio of 'amb_count' column to the total number of rows in the dataframe.

        The above three metrics will provide a noise floor per label basis that will indicate how much of a portion of a label's data
        is sensitive to tiny changes.

        There will be an additional column called 'Items' that give the total item count for each label for context.
        """

        labels = list(self.df[class_column].unique())  # all unique labels in the dataset

        labelMetricsDf = pd.DataFrame(data=[[0.0] * 4],
        index=labels,
        columns=['Items', 'Any Ambiguity per Sample', 'Multi Ambiguity per Sample', 'Mean Near-Boundary Count per Sample']
        )

        labelMetricsDf['Items'] = labelMetricsDf['Items'].astype(int)   # convert the Items column to integer type for pretty printing later

        for each_label in labels:
            labelDf = ambiguityFlagsDf[ambiguityFlagsDf[class_column] == each_label]  # ambiguity flagssub-dataframe for each_label entry only.

            labelMetricsDf.loc[each_label, 'Items'] = len(labelDf)
            labelMetricsDf.loc[each_label, 'Any Ambiguity per Sample'] = self._get_ambiguity_rate(labelDf, 'amb_any')
            labelMetricsDf.loc[each_label, 'Multi Ambiguity per Sample'] = self._get_ambiguity_rate(labelDf, 'amb_multi')
            labelMetricsDf.loc[each_label, 'Mean Near-Boundary Count per Sample'] = self._get_ambiguity_rate(labelDf, 'amb_count')

        return labelMetricsDf


    def _calculate_global_metrics(self, ambiguityFlagsDf: pd.DataFrame, class_column: str) -> pd.DataFrame:
        """
        Calculate global metrics for the ambiguityFlagsDf dataframe as follows:
        1 - 'Global Any Ambiguity per Sample': Ratio of True values in the 'amb_any' column to the total number of rows in the dataframe.
        2 - 'Global Multi Ambiguity per Sample': Ratio of True values in the 'amb_multi' column to the total number of rows in the dataframe.
        3 - 'Mean Near-Boundary Count per Sample': Ratio of 'amb_count' column to the total number of rows in the dataframe.

        The above three metrics will provide a noise floor that will indicate how much of the entire dataset is sensitive to tiny changes.

        Args:
            ambiguityFlagsDf: DataFrame containing the ambiguity flags
            class_column: Name of the column containing the class labels

        Returns:
            DataFrame containing the global metrics
        """
        globalMetricsDf = pd.DataFrame(data=[[0.0] * 3],columns=['Global Any Ambiguity per Sample', 'Global Multi Ambiguity per Sample', 'Mean Near-Boundary Count per Sample'])
        
        globalMetricsDf['Global Any Ambiguity per Sample'] = self._get_ambiguity_rate(ambiguityFlagsDf, 'amb_any')
        globalMetricsDf['Global Multi Ambiguity per Sample'] = self._get_ambiguity_rate(ambiguityFlagsDf, 'amb_multi')
        globalMetricsDf['Mean Near-Boundary Count per Sample'] = self._get_ambiguity_rate(ambiguityFlagsDf, 'amb_count')

        return globalMetricsDf

    def _get_ambiguity_rate(self, metricsDf: pd.DataFrame, column: str) -> float:

        if metricsDf[column].dtype == bool:
            return (metricsDf[column]==True).sum() / len(metricsDf)
        elif metricsDf[column].dtype == int:
            return metricsDf[column].mean()
        else:
            raise ValueError(f"Column {column} has an invalid data type: {metricsDf[column].dtype}")



    def _add_ambiguity_metrics(self, ambiguityFlagsDf: pd.DataFrame, class_column: str) -> pd.DataFrame:
        """
        Add ambiguity metrics columns to the ambiguityFlagsDf dataframe as follows:
        1 - 'amb_count': Number of ambiguity flags = True for each row (type: int)
        2 - 'amb_any': Whether any ambiguity flag is True for each row (type: bool)
        3 - 'amb_multi': Whether at least 2 ambiguity flags are True for each row (type: bool)
        """

        flags = ambiguityFlagsDf.drop([class_column], axis=1).columns.tolist()  # get all columns except the class column

        ambiguityFlagsDf['amb_count'] = (ambiguityFlagsDf==True).sum(axis=1)  # count number of True values for each row
        # CAUTION: We added a new column 'amb_count' to the dataframe we are processing !! Hence using flags to filter here on.
        ambiguityFlagsDf['amb_any'] = (ambiguityFlagsDf[flags]==True).any(axis=1)  # check if at least 1 True value exists in each row
        ambiguityFlagsDf['amb_multi'] = ambiguityFlagsDf['amb_count'] > 1  # check if at least 2 True values exist in each row (use amb_count so amb_any is not counted)
        
        # Finally, drop the flags column from the dataframe as they will no longer be needed.
        ambiguityFlagsDf = ambiguityFlagsDf.drop(flags, axis=1)

        return ambiguityFlagsDf


    def _generate_ambiguity_flags(self, features_dict: dict, tolerances: dict, class_column: str) -> pd.DataFrame:

        # Fixed thresholds used to calculate the labels in the dataset we are analysing
        up_th = features_dict['slope']['thresholds'][0][1]    #'upward_th'
        down_th = features_dict['slope']['thresholds'][1][1]  #'downward_th'
        flat_lo = features_dict['slope']['thresholds'][2][1]  #'flatness_lo'
        flat_hi = features_dict['slope']['thresholds'][3][1]  #'flatness_hi'
        ts_th = features_dict['trend_strength']['thresholds'][0][1]  #'strength_th'
        zcr_th = features_dict['zcr']['thresholds'][0][1]#'hi_osc_th' 
        v_hi = features_dict['volatility']['thresholds'][0][1]   #'hi_noise_th'
        v_lo = features_dict['volatility']['thresholds'][1][1]  #'lo_vol_th'

        # Defining one flag per threshold in the features_dict
        flags = ['amb_up', 'amb_down', 'amb_flat_lo', 'amb_flat_hi', 'amb_ts', 'amb_zcr', 'amb_v_hi', 'amb_v_lo']

        ambiguityFlagsDf = pd.DataFrame()
        ambiguityFlagsDf[class_column] = self.df[class_column]  # added the label column to the dataframe

        for each_col in flags:
            ambiguityFlagsDf[each_col] = False  # Add new columns filled with False by default

        # Process one data set row per loop iteration
        for each_row in self.df.index:

            # Assign SLOPE related flags first
            cur_sample = self.df.loc[each_row, 'slope']
            m_up = cur_sample - up_th
            m_down = down_th - cur_sample
            m_flat_lo = cur_sample - flat_lo
            m_flat_hi = flat_hi - cur_sample

            #### Assign slope-driven flags based on the calculated margins
            if abs(m_up) < tolerances['slope']:
                ambiguityFlagsDf.loc[each_row, 'amb_up'] = True
            # else False by default

            if abs(m_down) < tolerances['slope']:
                ambiguityFlagsDf.loc[each_row, 'amb_down'] = True
            # else False by default

            if abs(m_flat_lo) < tolerances['slope']:
                ambiguityFlagsDf.loc[each_row, 'amb_flat_lo'] = True
            # else False by default

            if abs(m_flat_hi) < tolerances['slope']:
                ambiguityFlagsDf.loc[each_row, 'amb_flat_hi'] = True
            # else False by default

            #### Assign TREND STRENGTH related flags next
            cur_sample = self.df.loc[each_row, 'trend_strength']
            m_ts = cur_sample - ts_th
            if abs(m_ts) < tolerances['trend_strength']:
                ambiguityFlagsDf.loc[each_row, 'amb_ts'] = True
            # else False by default

            #### Assign ZCR related flags next
            cur_sample = self.df.loc[each_row, 'zcr']
            m_zcr = cur_sample - zcr_th
            if abs(m_zcr) < tolerances['zcr']:
                ambiguityFlagsDf.loc[each_row, 'amb_zcr'] = True
            # else False by default

            #### Assign VOLATILITY related flags next
            cur_sample = self.df.loc[each_row, 'volatility']
            
            m_vlow = v_lo - cur_sample
            if abs(m_vlow) < tolerances['volatility']:
                ambiguityFlagsDf.loc[each_row, 'amb_v_lo'] = True
            # else False by default

            m_vhi = cur_sample - v_hi
            if abs(m_vhi) < tolerances['volatility']:
                ambiguityFlagsDf.loc[each_row, 'amb_v_hi'] = True
            # else False by default

        return ambiguityFlagsDf


    def _compute_tolerances(self, features_dict: dict) -> dict:
        """
        First, compute the robust sigma (scale) for each feature over the entire dataset.
        These sigmas will be used as a robust estimate of scale. i.e., standard-deviation like quantity that:
        - is not distorted by outliers
        - works with skewed or heavy-tailed distributions
        - is comparable across different features
        
        For this, we use InterQuartile range (IQR) but this is not on the same scale as the standard deviation.
        Therefore, it needs to be scaled using a conversion factor. This conversion assumes:
        If the data were normally distributed, what would be the IQR be relative to the standard deviation?

        IQR of a standard distribution is approximately 1.349. (i.e., the quartile between the 25th and 75th percentiles)
        This will be used as the scaling factor for all features to calculate a robust sigma that will remain stable
        when the data is not normally distributed.

        Then the tolerance is calculated as: tolerance = (IQR / IQR_normal) * k
        where k is the pre-defined feature-specific value.
        """

        IQR_normal = 1.349  # IQR of standard distribution.

        tolerances = {key: 0.0 for key in features_dict.keys()}  # keys are the features of interest.

        for each_feature in tolerances.keys():
            IQR = self.df[each_feature].quantile(0.75) - self.df[each_feature].quantile(0.25)
            tolerances[each_feature] = (IQR / IQR_normal) * features_dict[each_feature]['k']

        return tolerances


    def temporal_stability_analysis(self, temporal_stability_params: dict, class_column='gt', save_folder='plots'):
        """
        Temporal stability analysis is a technique to analyze the stability of a time series over time.
        This analysis involves time drift, and frequency drift.

        Args:
            temporal_stability_params: Dictionary containing the parameters for the temporal stability analysis.
            class_column: Name of the column containing the class labels
            save_folder: Folder to save the plots
        """

        temporal_stability_params['annual_time_step'] = int(temporal_stability_params['annual_time_step'])

        annual_time_step = temporal_stability_params['annual_time_step']
        #sliding_window_size_months = temporal_stability_params['sliding_months']   # For future use

        temporalDf = self.df.copy()

        # Convert 'end_date' column to datetime
        # If the column contains integers in YYYYMMDD format
        if 'end_date' in temporalDf.columns:
            temporalDf['end_date'] = pd.to_datetime(temporalDf['end_date'].astype(str), format='%Y%m%d')

        # Create a new column in temporalDf called 'era' which will contain the era of the dataset
        # e.g. if annual_time_step = 1 then eras between 2014 and 2024: "2014-2014", "2015-2015", ..., "2024-2024"
        # e.g. if annual_time_step = 2 then eras between 2014 and 2024: "2014-2015", "2016-2017", ..., "2023-2024" etc.

        min_year, max_year = np.min(temporalDf['end_date']).year, np.max(temporalDf['end_date']).year
        
        # Call the _assign_era() method to assign the era to each row based on the year in 'end_date' and annual_time_step.
        temporalDf['era'] = temporalDf['end_date'].apply(self._assign_era, annual_time_step=annual_time_step, min_year=min_year, max_year=max_year)  # static arguments need to be passed AFTER the end_date argument.

        # ANALYSIS 1 - Class frequency drift over eras.
        # Goal: Answer the following question: Does the proportion of each class (OSCILLATING, OTHER, TREND_UP, STATIONARY, TREND_DOWN)
        # remain stable across eras, or does it change materially over time?
        # This is to measure data composition stability to flag any issues before building a model.
        # We just need the 'end_date', 'era' and 'gt' columns for this analysis.
        freqDriftDf = self._class_frequency_drift_analysis(temporalDf, class_column, save_folder=save_folder, display_plots=False, enable_pdf_write=True)

        # ANALYSIS 2 - Feature drift over eras.
        # Goal: Answer the following question: Does the distribution of each/any of the features change materially over time.
        # This analysis does not provide an answer to 'why' it changes or whetehr it is likely to hurt the modeling. We are seeking
        # whether there is a change and how.
        features_to_analyze = ['volatility', 'slope', 'zcr', 'trend_strength']
        statistics_to_use = ['count', '10th_prcntl', 'median', '90th_prcntl']
        baseline_years = [2014, 2015, 2016, 2017, 2018, 2019]  # these are the pre-covid years in the dataset (in ascending order!) to use as baseline for feature drift analysis
        self._feature_drift_analysis(
                                        temporalDf, 
                                        features_to_analyze, 
                                        statistics_to_use,
                                        baseline_years,
                                        class_column, 
                                        save_folder=save_folder, 
                                        display_plots=False, 
                                        enable_pdf_write=True
                                    )


    def _feature_drift_analysis(self, temporalDf: pd.DataFrame,
    features_to_analyze: list[str],
    statistics_to_use: list[str],
    baseline_years: list[int],
    class_column: str,
    save_folder: str, display_plots: bool, enable_pdf_write: bool):
        """
        ANALYSIS 2 - Feature drift over eras.
        Goal: Answer the following question: Does the distribution of each/any of the features change materially over time.
        This analysis does not provide an answer to 'why' it changes or whether it is likely to hurt the modeling. We are seeking
        whether there is a change and how.
        """

        # Check temporalDf to make sure there are no NaNs, -inf or + inf values in the features to analyze
        for each_feature in features_to_analyze:
            if temporalDf[each_feature].isnull().any() or temporalDf[each_feature].isin([-np.inf, np.inf]).any():
                raise ValueError(f"Error: {each_feature} contains NaNs, -inf or + inf values")

        # Process one feature per loop iteration
        for each_feature in features_to_analyze:
            
            eras = temporalDf['era'].sort_index(ascending=True).unique()
            resDf = pd.DataFrame(index=eras,columns=statistics_to_use)
            
            for each_era in resDf.index:
                
                featSeries = temporalDf[temporalDf['era']==each_era][each_feature]

                for each_statistic in statistics_to_use:
                    if each_statistic == 'count':
                        resDf.loc[each_era, each_statistic] = featSeries.shape[0]
                    elif each_statistic == '10th_prcntl':
                        resDf.loc[each_era, each_statistic] = featSeries.quantile(0.1).round(4)
                    elif each_statistic == 'median':
                        resDf.loc[each_era, each_statistic] = featSeries.median().round(4)
                    elif each_statistic == '90th_prcntl':
                        resDf.loc[each_era, each_statistic] = featSeries.quantile(0.9).round(4)
                    else:
                        raise ValueError(f"Error: {each_statistic} is not a valid statistic")


            # Generate a plot to make it easier to spot any feature drift over eras
            baseline_median = self._compute_baseline_median(resDf, baseline_years)  # helper function to compute the median of the baseline years
            self._feature_drift_plot_analysis(resDf, each_feature, baseline_median, save_folder, display_plots, enable_pdf_write)

            # Update pdf report content
            print(f"Temporal analysis - feature drift table for {each_feature}:")
            prt.print_dataframe(resDf, justify_numeric="center")
            self.reportObj.print(rprt.ReportDataType.HEADING_2, f"{each_feature} feature drift table:")
            self.reportObj.print_dataframe_as_table(resDf)


    def _compute_baseline_median(self, resDf: pd.DataFrame, baseline_years: list[int]) -> float:
        """
        Index of resDf includes the years of eras in the format 'YYYY-YYYY'. baseline_years includes each year as int in a list in ASCENDING order.
        Therefore, we need to check the minimum and maximum year values in each era to find the baseline median.
        """
        baseline_median = 0.0
        num_years = 0  # needed to calculate the mean at the end
        
        for each_era in resDf.index:
            min_year, max_year= int(each_era.split('-')[0]), int(each_era.split('-')[1])

            first_count = True
            for each_year in baseline_years:
                if each_year >= min_year and each_year <= max_year:
                    if first_count:
                        first_count = False
                        baseline_median += resDf.loc[each_era, 'median']
                        num_years += 1
                    # else skip
        
        return baseline_median / num_years  # return the average median of the baseline years


    def _feature_drift_plot_analysis(self, resDf: pd.DataFrame, current_feature: str, baseline_median: float, save_folder: str, display_plots: bool, enable_pdf_write: bool) -> None:   
        """
        Generate a plot to make it easier to spot any feature drift over eras
        
        Args:
            resDf: DataFrame with index as eras (time axis) and columns: 'median', '10th_prcntl', '90th_prcntl'
            current_feature: Name of the feature being analyzed
            save_folder: Folder path to save the plot
            display_plots: Whether to display the plot
            enable_pdf_write: Whether to add plot to PDF report
        """
        # Extract data from DataFrame
        eras = resDf.index.tolist()  # Time axis (eras)
        median_values = resDf['median'].values.astype(float)
        p10_values = resDf['10th_prcntl'].values.astype(float)
        p90_values = resDf['90th_prcntl'].values.astype(float)
        
        # Convert eras to numeric positions for plotting
        x_positions = range(len(eras))
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot the shaded band (10th to 90th percentile)
        ax.fill_between(x_positions, p10_values, p90_values, alpha=0.3, color='lightblue', label='10th-90th Percentile Band')
        
        # Plot the median as a solid line
        ax.plot(x_positions, median_values, 'b-', linewidth=2, label='Median', marker='o', markersize=4)
        
        # Plot the baseline median as a solid line for comparison
        ax.plot(x_positions, [baseline_median] * len(x_positions), 'r--', linewidth=1, label='baseline median')

        # Formatting
        ax.set_xlabel('Era', fontsize=12)
        ax.set_ylabel(f'{current_feature} Value', fontsize=12)
        ax.set_title(f'current feature: {current_feature}', fontsize=14, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(eras, rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Save plot if requested
        if enable_pdf_write and save_folder:
            save_path = Path(save_folder) / f'{current_feature}_drift_plot.png'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_path), dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")           

            self.reportObj.new_page(enable_write=enable_pdf_write)
            self.reportObj.print(rprt.ReportDataType.HEADING_2, f"Temporal analysis - Feature Drift Plot for {current_feature}")
            self.reportObj.print_image(save_path)
        
        # Display plot if requested
        if display_plots:
            self._enable_interactive_plots()
        

    def _class_frequency_drift_analysis(self, temporalDf: pd.DataFrame, class_column: str, save_folder: str, display_plots: bool, enable_pdf_write: bool):
        """
        Analysis 1 - Class frequency drift over eras.
        Goal: Answer the following question:Does the proportion of each class (OSCILLATING, OTHER, TREND_UP, STATIONARY, TREND_DOWN)
        remain stable across eras, or does it change materially over time?
        This is to measure data composition stability to flag any issues before building a model.
        We just need the 'end_date', 'era' and 'gt' columns for this analysis.        
        """
    
        era_names = temporalDf['era'].unique()  # All eras in dataset
        classes = temporalDf[class_column].unique()  # All classes in dataset
        resDf = pd.DataFrame(columns=['era', 'class', 'proportion'])
        #resDf['era'] = era_names  # initialize 'era' column with all eras in the dataset

        for era_idx, each_era in enumerate(era_names):
            
            era_offset = era_idx * len(classes)     # to assign new items for a new era
            
            for row_idx, each_class in enumerate(classes):          
                resDf.loc[row_idx + era_offset, 'era'] = each_era
                resDf.loc[row_idx + era_offset, 'class'] = each_class
                # First assignt the total count to 'proportion' column
                resDf.loc[row_idx + era_offset, 'proportion'] = temporalDf[(temporalDf[class_column] == each_class) & (temporalDf['era'] == each_era)].shape[0]  # get count of each_class

            # Get the sum of all classes in each_era and update the 'proportion' column by calculating the ratio for each class in each_era
            sum = resDf[(resDf['era'] == each_era)]['proportion'].sum()    
            # Use .loc to properly assign the ratio (chained indexing doesn't work for assignment)
            resDf.loc[resDf['era'] == each_era, 'proportion'] = resDf[(resDf['era'] == each_era)]['proportion'] / sum  #resDf.loc[resDf['era'] == each_era, 'proportion'] / sum

        # Time to visualize the resDf content using stacked area chart
        freqDriftDf = self._stacked_area_chart_analysis(resDf, save_folder, display_plots, enable_pdf_write)

        # Display the final table on terminal and in pdf report
        print(f"Temporal analysis - class frequency drift table:")
        prt.print_dataframe(freqDriftDf, justify_numeric="center")
        # Add to the same pdf page as the plot. No new page created.
        self.reportObj.print(rprt.ReportDataType.HEADING_2, "Temporal analysis - class frequency drift table:")
        self.reportObj.print_dataframe_as_table(freqDriftDf)

    def _stacked_area_chart_analysis(self, proportionsDf: pd.DataFrame, save_folder: str, display_plots: bool, enable_pdf_write: bool):
        """
        Create a stacked area chart showing class proportions over eras.
        
        Args:
            proportionsDf: DataFrame with columns 'era', 'class', 'proportion'
        """
        # Pivot the DataFrame so each class becomes a column
        # era becomes index, class values become columns, proportion becomes values
        pivot_df = proportionsDf.pivot(index='era', columns='class', values='proportion')
        
        # Ensure proportions are numeric (convert from object/string to float if needed)
        pivot_df = pivot_df.astype(float)
        
        # Sort by era to ensure proper order
        pivot_df = pivot_df.sort_index()
        
        # Prepare data for stackplot
        eras = pivot_df.index.tolist()  # era names for x-axis labels
        class_columns = pivot_df.columns.tolist()  # class names for labels
        proportions_by_class = [pivot_df[col].values.astype(float) for col in class_columns]  # y values, one array per class (ensure float type)
        
        # Convert eras to numeric positions for stackplot (stackplot requires numeric x-values)
        x_positions = range(len(eras))
        
        fig = plt.figure(figsize=(12, 6))
        plt.stackplot(x_positions, *proportions_by_class, labels=class_columns, alpha=0.7)
        plt.title('Stacked Area Chart: Class Frequency Drift Over Eras')
        plt.xlabel('Era')
        plt.ylabel('Proportion')
        plt.xticks(x_positions, eras, rotation=45, ha='right')  # Set era names as x-axis labels
        plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()  # Adjust layout to prevent label cutoff
        #plt.show()

        # Save the plot if filepath is provided
        if save_folder is not None:
            filename = f'temporal_class_freq_drift.png'
            self._save_plot(figure=fig,
                            filename=filename,
                            save_folder=save_folder,
                            pdf_page_title="TEMPORAL ANALYSIS - CLASS FREQUENCY DRIFT",
                            enable_pdf_write=enable_pdf_write
                            )

        if display_plots == True:
            # Enable interactive mode for non-blocking display
            # Note: Figure remains open until script terminates - no plt.close() call
            self._enable_interactive_plots()
            print(f"Temporal analysis - class frequency drift plot displayed")

        return pivot_df  # for plotting the table at the caller


    def _assign_era(self, end_date: pd.Timestamp, annual_time_step: int, min_year: int, max_year: int) -> str:
        year = end_date.year
        # Calculate which era this year belongs to
        era_start = ((year - min_year) // annual_time_step) * annual_time_step + min_year
        era_end = era_start + annual_time_step - 1
        if era_end > max_year:
            era_end = max_year
        
        if era_start > era_end:
            # error condition. Should never occur in a valid dataset!
            raise ValueError(f"Error: era_start ({era_start}) is greater than era_end ({era_end})")
            return f"0000-0000"

        return f"{era_start}-{era_end}"


    def _enable_interactive_plots(self):
        plt.ion()
        plt.show(block=False)
        # Give matplotlib time to render the plot window
        plt.pause(0.1)  # Brief pause to ensure plot window is rendered
        plt.draw()  # Force a draw to update the display