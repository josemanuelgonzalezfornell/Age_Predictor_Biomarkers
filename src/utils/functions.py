import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss

#Get univariate analysis
def get_univariate_analysis(df):
    """
    Performs a univariate analysis of a given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be analyzed.

    Returns
    -------
    univar_analysis : pd.DataFrame
        DataFrame with the results of the univariate analysis.
    """

    normal_var = 0
    no_normal_var = 0
    univar_analysis = pd.DataFrame(
        {}, columns=["Mean", "Median", "Mode", "Variance", "Standard_Deviation", "Percentile_25", "Percentile_75", "K_test", "p_value", "Distribution"])

    for col in df.columns:
        print(f"\033[1mUnivariate Analysis of {col}:\033[0m")
        # Perform an analysis if the variable is categorical.
        if (df[col].dtype == object):
            print(f"Categoric feature: {col}")
            print(f"-Uniques values:\n{df[col].value_counts()}")
            print(f"-Number of unique values: {df[col].nunique()}")
            print("\n\n\n")

        # Perform an analysis if the variable is numeric.
        else:
            # Create a histogram and a boxplot of the variable.
            fig, axes = plt.subplots(1, 2, figsize=(10,4));
            sns.histplot(df[col], kde=True, ax= axes[0]);
            axes[0].set_title("Histogram");
            sns.boxplot(df[col], ax= axes[1]);
            axes[1].set_title("Boxplot");
            fig.suptitle(f"Analysis of {col}");
            plt.show();

            # Statistically check using the Kolmogorov-Smirnov test if the variable follows a normal distribution.
            stat, p = ss.kstest(df[col], 'norm')
            alpha = 0.05

            # Add the data to the DataFrame depending on whether H0 is accepted or not.
            if p < alpha:
                no_normal_var += 1

                new_row = pd.DataFrame([{
                    "Feature": col, 
                    "Mean": df[col].mean(), 
                    "Median": df[col].median(), 
                    "Mode": df[col].mode().iloc[0], 
                    "Variance": df[col].var(), 
                    "Standard_Deviation": df[col].std(), 
                    "Percentile_25": df[col].astype(float).quantile(0.25), 
                    "Percentile_75": df[col].astype(float).quantile(0.75), 
                    "K_test": stat, 
                    "p_value": p, 
                    "Distribution": "Not normal"
                }])

                # Concatenate the new row to the existing DataFrame
                univar_analysis = pd.concat([univar_analysis, new_row], ignore_index=True)
                # univar_analysis = univar_analysis.append({"Feature": col, "Mean": df[col].mean(), "Median": df[col].median(
                # ), "Mode": df[col].mode().iloc[0], "Variance": df[col].var(), "Standard_Deviation": df[col].std(), "Percentile_25": df[col].quantile(0.25), "Percentile_75": df[col].quantile(0.75), "K_test": stat, "p_value": p, "Distribution": "Not normal"}, ignore_index=True)
                print(f"The column {col} does not follow a normal distribution.\n\n\n")

            else:
                normal_var += 1
                univar_analysis = univar_analysis.append({"Feature": col, "Mean": df[col].mean(), "Median": df[col].median(
                ), "Mode": df[col].mode().iloc[0], "Variance": df[col].var(), "Standard_Deviation": df[col].std(), "Percentile_25": df[col].quantile(0.25), "Percentile_75": df[col].quantile(0.75), "K_test": stat, "p_value": p, "Distribution": "Normal"}, ignore_index=True)
                print(f"The column {col} follows a normal distribution.\n\n\n")

    # Set the column "Municipio" as the index.
    univar_analysis.set_index("Feature", inplace=True)

    # Print the number of variables that follow a normal distribution and the number that do not.
    print(
        f"\033[1mNumber of variables that follow a normal distribution:\033[0m {normal_var}")
    print(
        f"\033[1mNumber of variables that do not follow a normal distribution:\033[0m {no_normal_var}")
    return univar_analysis

# Obtiene el análisis bivariante de un dataframe


def get_bivariate_analysis(df):
    """
    Performs a bivariate analysis of a given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be analyzed.

    Returns
    -------
    None
    """

    principal_df = df.filter(regex=r'\bInmuebles_totales\b|\b\w+2021\b')
    sns.heatmap(principal_df.corr());
    sns.pairplot(principal_df, diag_kind='kde');