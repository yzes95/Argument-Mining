import pandas as pd
import csv
import sys
UKP_FILE_PATH = "UKP_sentential_argument_mining/data/gun_control.tsv"
 



print(f"\nLoading UKP data from '{UKP_FILE_PATH}'...")
ukp_df = pd.read_csv(
    UKP_FILE_PATH,
    sep='\t',
    on_bad_lines='warn',
    quoting=csv.QUOTE_NONE, # Tell pandas not to treat quotes specially
    engine='python' # Use the python engine for more flexibility
)

# --- Add a check to ensure parsing was successful ---
# The UKP file should have 7 columns.
if ukp_df.shape[1] < 7:
    print("\n--- PARSING ERROR ---")
    print("The UKP file was not parsed correctly into multiple columns.")
    print("Please ensure the file is correctly tab-separated.")
    print(f"Pandas only detected {ukp_df.shape[1]} column(s).")
    sys.exit()
    
# Map the UKP labels to our format
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

ukp_final.to_csv("ukp_testing_processed_corpus.csv", index=False)