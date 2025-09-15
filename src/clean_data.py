import pandas as pd

# For visual confirmation or logging
def print_shape_summary(df, label):
    print(f"{label}: {df.shape[0]} rows, {df.shape[1]} columns")

def summarize_numeric(df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []
    summary = df.drop(columns=exclude_cols).describe().round(2)
    print(summary)


# Helper functions
def remove_outliers_iqr(df, column, multiplier=1.5, verbose=True):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    before = df.shape[0]
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    after = filtered_df.shape[0]
    
    if verbose:
        print(f"[{column}] IQR Filter: {before - after} rows removed")
    
    return filtered_df

def cap_max(df, column, max_value, verbose=True):
    before = df.shape[0]
    filtered_df = df[df[column] <= max_value]
    after = filtered_df.shape[0]

    if verbose:
        print(f"[{column}] Max cap: {before - after} rows removed, capped at {max_value}")
    
    return filtered_df

def preprocess_and_save(input_path, output_path):
    print(f"Loading data from {input_path}\n")
    df = pd.read_csv(input_path)
    print_shape_summary(df, "Original")
    summarize_numeric(df, exclude_cols=['Zipcode'])
    
    print("Removing null values and entries with 0 bathrooms...")
    df = df.dropna(subset=['Bedroom', 'Bathroom', 'RentEstimate'])
    df = df[df['Bathroom'] > 0]
    print_shape_summary(df, "After Null Removal")
    
    # IQR Trimming
    df = remove_outliers_iqr(df, 'Area')
    df = remove_outliers_iqr(df, 'PPSq')
    df = remove_outliers_iqr(df, 'RentEstimate')
    
    # Max capping
    df = cap_max(df, 'Bedroom', 6)
    df = cap_max(df, 'Bathroom', 5)
    
    print()
    
    print_shape_summary(df, "Final Cleaned")
    summarize_numeric(df, exclude_cols=['Zipcode'])
    
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")

# Run the cleaning script
if __name__ == "__main__":
    preprocess_and_save(
        input_path="../data/us_house_listings_zillow_2023.csv",
        output_path="../data/cleaned.csv"
    )