import pandas as pd

# --- Configuration ---
# Define the paths to your data files
TRAIN_FILE = 'combined_training_processed_corpus.csv'
TEST_FILES_TO_CLEAN = {
    'testing_10_percent_processed_corpus_part_1.csv': 'testing_10_percent_processed_corpus_part_1_clean.csv',#rename the files as they were by removing_clean later so the prgram would run
    'testing_10_percent_processed_corpus_part_2.csv': 'testing_10_percent_processed_corpus_part_2_clean.csv',
    'ukp_testing_processed_corpus.csv': 'ukp_testing_processed_corpus_clean.csv'
}

# Define the minimum number of words a sentence must have to be kept.
MINIMUM_WORD_COUNT = 4


def clean_all_files():
    """
    Loads training data, then iterates through test files to perform
    a two-step cleaning process on each.
    """
    try:
        # --- Step 1: Get all unique sentences from the training data ---
        print(f"Loading training data from '{TRAIN_FILE}'...")
        train_df = pd.read_csv(TRAIN_FILE)
        # Using a set provides a very fast way to check for existence
        train_texts_set = set(train_df['Text'])
        print(f"Found {len(train_texts_set)} unique sentences in the training data.")

        # --- Step 2: Process each test file ---
        for input_path, output_path in TEST_FILES_TO_CLEAN.items():
            print(f"\n--- Processing '{input_path}' ---")
            
            # Load the test file
            test_df = pd.read_csv(input_path)
            initial_rows = len(test_df)
            print(f"Original rows: {initial_rows}")

            # CLEANING PASS 1: Remove rows with text found in the training set
            clean_df_pass1 = test_df[~test_df['Text'].isin(train_texts_set)]
            rows_after_pass1 = len(clean_df_pass1)
            print(f"Rows removed (found in training set): {initial_rows - rows_after_pass1}")
            
            # CLEANING PASS 2: Remove rows with short text
            # Calculate word count for the remaining rows
            clean_df_pass1['word_count'] = clean_df_pass1['Text'].str.split().str.len()
            
            # Keep only rows where word count is sufficient
            clean_df_pass2 = clean_df_pass1[clean_df_pass1['word_count'] >= MINIMUM_WORD_COUNT].copy()
            
            # We can drop the temporary 'word_count' column now
            clean_df_pass2.drop(columns=['word_count'], inplace=True)
            
            rows_after_pass2 = len(clean_df_pass2)
            print(f"Rows removed (fewer than {MINIMUM_WORD_COUNT} words): {rows_after_pass1 - rows_after_pass2}")
            
            # Final Report and Save
            final_rows = len(clean_df_pass2)
            print(f"Final rows remaining: {final_rows}")
            clean_df_pass2.to_csv(output_path, index=False)
            print(f"✅ Cleaned data saved to '{output_path}'")

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Make sure all CSV files are in the same directory as the script.")
        print(f"File not found: {e.filename}")
    except KeyError as e:
        print(f"\n❌ ERROR: A required column was not found in one of the CSV files.")
        print(f"Missing column: {e}")

# --- Run the cleaning process ---
if __name__ == "__main__":
    clean_all_files()
    print("\nProcess complete. ✅")
