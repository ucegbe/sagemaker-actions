
import os
import tempfile
import numpy as np
import pandas as pd
import datetime as dt
import glob
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import io
import sys
import time
import json
from time import strftime, gmtime
from sklearn import preprocessing

def load_and_combine_csv_files(directory):
    """
    Load all CSV files from a directory and combine them into a single DataFrame.

    Args:
    directory (str): Path to the directory containing CSV files.

    Returns:
    pandas.DataFrame: Combined DataFrame of all CSV files.
    """
    # Use glob to get all the csv files in the folder
    csv_files = glob.glob(os.path.join(directory, "*.csv"))

    # List to hold individual DataFrames
    df_list = []

    total_rows = 0
    for file in csv_files:
        try:
            # Read each file into a DataFrame
            df = pd.read_csv(file)
            total_rows += len(df)
            df_list.append(df)
            print(f"Loaded {file}: {len(df)} rows")
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")

    # Combine all DataFrames in the list
    combined_df = pd.concat(df_list, ignore_index=True)

    print(f"\nTotal files processed: {len(csv_files)}")
    print(f"Total rows in combined DataFrame: {len(combined_df)}")

    return combined_df



def detect_and_encode_categorical(df, max_categories=10, include_dates=True):
    """
    Detect categorical columns (including object, int, and datetime), encode them, 
    and create a mapping of their indexes. Excludes the first column (assumed to be the target).

    Args:
    df (pandas.DataFrame): Input DataFrame
    max_categories (int): Maximum number of unique values to consider a column categorical
    include_dates (bool): Whether to treat date columns as categorical

    Returns:
    tuple: (preprocessed DataFrame, dict of categorical column indexes, dict of label encoders)
    """
    categorical_columns = []
    categorical_indexes = {}
    label_encoders = {}

    # Get the name of the first column (assumed to be the target)
    target_column = df.columns[0]

    for idx, (col, dtype) in enumerate(df.dtypes.items()):
        # Skip the first column (target)
        if col == target_column:
            continue

        if (dtype == 'object' or 
            (df[col].nunique() <= max_categories and dtype != 'float64') or
            pd.api.types.is_integer_dtype(dtype) or
            (include_dates and pd.api.types.is_datetime64_any_dtype(dtype))):

            categorical_columns.append(col)
            categorical_indexes[col] = idx  # Adjust index to account for skipped target column

            # Handle datetime columns
            if pd.api.types.is_datetime64_any_dtype(dtype):
                if include_dates:
                    df[col] = df[col].dt.strftime('%Y-%m-%d')  # Convert to string format
                else:
                    continue  # Skip datetime columns if not included

            # Encode categorical variables
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    print(f"Detected {len(categorical_columns)} categorical columns: {categorical_columns}")
    return df, categorical_indexes




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--split-ratio', type=str, default="0.3",dest='split_ratio')
    args = parser.parse_args()

    base_dir_input = "/opt/ml/processing/input"
    base_dir = "/opt/ml/processing/"
    #Read Data
    df = load_and_combine_csv_files(base_dir_input)
    # Sample Analysis 
    df = df.drop("Phone", axis=1)
    df["Area Code"] = df["Area Code"].astype(object)
    
    df["target"] = df["Churn?"].map({"True.": 1, "False.": 0})
    df.drop(["Churn?"], axis=1, inplace=True)
    
    df = df[["target"] + df.columns.tolist()[:-1]]
    # df = pd.concat([churn]*50, ignore_index=True)
    df, cat_columns = detect_and_encode_categorical(df, max_categories=10, include_dates=True)
    cat_idx = list(cat_columns.values())
    
    # Save categorical information
    with open(f"{base_dir}/train/cat_idx.json", "w") as outfile:
        json.dump({"cat_idx": cat_idx}, outfile)
        
    # train, test, validation
    train, val_n_test = train_test_split(
        df, test_size=float(args.split_ratio), random_state=42, stratify=df["target"]
    )
    validation, test = train_test_split(
        val_n_test, test_size=float(args.split_ratio), random_state=42, stratify=val_n_test["target"]
    )
    
    # Save datasets
    train.to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    validation.to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    test.to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
