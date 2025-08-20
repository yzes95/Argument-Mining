import pandas as pd
from sklearn.model_selection import train_test_split

# --- Configuration ---
INPUT_20_PERCENT_FILE = "combined_training_processed_corpus.csv" 

def split_data(filepath):
    """
    Loads a CSV file and splits it into two, saving the results.
    """
    try:
        # Step 1: Load the dataset
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} segments from '{filepath}'")
    except FileNotFoundError:
        print(f"Error: The input file was not found at '{filepath}'")
        print("Please make sure the file exists and the INPUT_FILE variable is set correctly.")
        return

    # Step 2: Separate the features (X) from the target label (y)
    # For now, X contains all columns except the Type.
    if 'Type' not in df.columns:
        print("Error: A 'Type' column was not found in the CSV file.")
        return

    X = df.drop('Type', axis=1)
    y = df['Type']
    
    # Step 3: Spliting the data 
    # The test_size=0.5 splits the current data in half.
    X_val, X_test, y_val, y_test = train_test_split(
        X,
        y,
        test_size=0.875,
        random_state=42,  # Ensures the split is the same every time you run it
        stratify=y        # Ensures both new sets have the same class balance
    )
    
    print(f"\nSplitting data...")
    print(f"  - part_1 of data set size: {len(y_val)}")
    print(f"  - part_2 of data set size:       {len(y_test)}")

    # Step 4: Recombine the features and labels into new DataFrames
    vald_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Step 5: Save the new datasets to their own CSV files
    test_df.to_csv("RoBERTa_training_dataset.csv", index=False)
    vald_df.to_csv("RoBERTa_validation_dataset.csv", index=False)
    
    print("\nSuccess! New files created")


# --- Main Execution Block ---
if __name__ == "__main__":
    split_data(INPUT_20_PERCENT_FILE)
