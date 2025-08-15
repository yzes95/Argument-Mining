import pandas as pd
import csv
# --- Configuration ---
# Define the paths to your input data files
AAEC_FILE_PATH = "combined_training_processed_corpus.csv"
UKP_FILE_PATH = "UKP_sentential_argument_mining/train_data/abortion.tsv"

# Define the name for the new, combined output file
OUTPUT_FILE_PATH = 'combined_training_processed_corpus.csv'


def merge_and_align_data():
    """
    Loads, aligns, and merges the AAEC and UKP datasets into a single,
    unified training file with compatible columns ('Text', 'Type').
    This version is more robust to parsing errors in the .tsv file.
    """
    try:
        # --- Step 1: Load and process the AAEC data ---
        print(f"Loading AAEC data from '{AAEC_FILE_PATH}'...")
        aaec_df = pd.read_csv(AAEC_FILE_PATH)
        # We only need the 'Text' and 'Type' columns
        aaec_final = aaec_df[['Text', 'Type']].copy()
        print(f"Processed {len(aaec_final)} rows from AAEC.")

        # --- Step 2: Load and process the UKP data (More Robust Version) ---
        print(f"\nLoading UKP data from '{UKP_FILE_PATH}'...")
        ukp_df = pd.read_csv(
            UKP_FILE_PATH,
            sep='\t',
            on_bad_lines='warn',
            quoting=csv.QUOTE_NONE, # Tell pandas not to treat quotes specially
            engine='python' # Use the python engine for more flexibility
        )
        
        # --- Add a check to ensure parsing was successful ---
        # The UKP file should have 7 columns. If not, something went wrong.
        if ukp_df.shape[1] < 7:
            print("\n--- PARSING ERROR ---")
            print("The UKP file was not parsed correctly into multiple columns.")
            print("Please ensure the file is correctly tab-separated.")
            print(f"Pandas only detected {ukp_df.shape[1]} column(s).")
            return # Stop execution

        # Map the UKP labels to the AAEC label format
        label_map = {
            'Argument_for': 'Argumentative',
            'Argument_against': 'Argumentative',
            'NoArgument': 'Non-Argumentative'
        }
        ukp_df['Type'] = ukp_df['annotation'].map(label_map)

        # Select and rename columns to match the AAEC format
        ukp_final = ukp_df[['sentence', 'Type']].rename(columns={'sentence': 'Text'})
        
        # Drop any rows where label mapping might have failed
        ukp_final.dropna(subset=['Type'], inplace=True)
        print(f"Processed {len(ukp_final)} rows from UKP.")

        # --- Step 3: Combine the two datasets ---
        print("\nCombining the two datasets...")
        combined_df = pd.concat([aaec_final, ukp_final], ignore_index=True)
        
        # Optional: Shuffle the combined dataset to mix the domains
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Total rows in combined dataset: {len(combined_df)}")

        # --- Step 4: Save the final merged file ---
        combined_df.to_csv(OUTPUT_FILE_PATH, index=False)
        
        print(f"\nSuccessfully merged data and saved to '{OUTPUT_FILE_PATH}'")
        print("\n--- Preview of Final Merged Data ---")
        print(combined_df.head())

    except FileNotFoundError as e:
        print(f"Error: Make sure all data files are in the same directory as the script.")
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Run the merging process ---
if __name__ == "__main__":
    merge_and_align_data()
