import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
# TODO: CRITICAL - Ensure these filenames EXACTLY match the files used in your RoBERTa training script.
TRAINING_DATA_FILE = "combined_training_processed_corpus.csv"
# TRAINING_DATA_FILE = "training_processed_corpus.csv"
# VALIDATION_DATA_FILE = "RoBERTa_validation_dataset.csv"
TEST_DATA_FILE = "ukp_testing_processed_corpus.csv" # The file that gives a perfect score
# TEST_DATA_FILE = "testing_10_percent_processed_corpus_part_2.csv" # The file that gives a perfect score

SIMILARITY_THRESHOLD = 0.99 # Threshold for near-duplicate detection (1.0 is identical)

def find_data_leakage():
    """
    Performs three checks for data leakage between training/validation and test sets:
    1. Checks for identical sentences ('Text' column).
    2. Checks for overlap in source documents ('EssayID' column).
    3. Checks for near-duplicate sentences using TF-IDF and Cosine Similarity.
    """
    try:
        # --- Step 1: Load all datasets ---
        print("--- Loading Datasets ---")
        train_df = pd.read_csv(TRAINING_DATA_FILE)
        # val_df = pd.read_csv(VALIDATION_DATA_FILE)
        test_df = pd.read_csv(TEST_DATA_FILE)
        # print(f"Loaded {len(train_df)} training rows, {len(val_df)} validation rows, and {len(test_df)} test rows.")
        print(f"Loaded {len(train_df)} training rows, and {len(test_df)} test rows.")

        # Combine training and validation data, as the model sees both during development
        # combined_train_df = pd.concat([train_df, val_df], ignore_index=True)
        # print(f"Total training + validation rows: {len(combined_train_df)}")
        print(f"Total training + validation rows: {len(train_df)}")

        # --- Step 2: Check for identical text leakage ---
        print("\n--- CHECK 1: IDENTICAL SENTENCE LEAKAGE ---")
        # train_texts = set(combined_train_df['Text'])
        train_texts = set(train_df['Text'])
        leaked_text_df = test_df[test_df['Text'].isin(train_texts)]
        
        if not leaked_text_df.empty:
            print(f"🔴 Found {len(leaked_text_df)} rows in the test set with sentences that are also in the training/validation set.")
        else:
            print("✅ SUCCESS: No identical sentences found between test and training/validation sets.")

        # --- Step 3: Check for EssayID leakage ---
        print("\n--- CHECK 2: ESSAY ID (SOURCE DOCUMENT) LEAKAGE ---")
        # if 'EssayID' in combined_train_df.columns and 'EssayID' in test_df.columns:
            # train_essay_ids = set(combined_train_df['EssayID'])
        if 'EssayID' in train_df.columns and 'EssayID' in test_df.columns:
            train_essay_ids = set(train_df['EssayID'])
            leaked_essay_id_df = test_df[test_df['EssayID'].isin(train_essay_ids)]

            if not leaked_essay_id_df.empty:
                print(f"🔴 Found {len(leaked_essay_id_df)} rows in the test set with EssayIDs that are also in the training/validation set.")
            else:
                print("✅ SUCCESS: No overlapping EssayIDs found between test and training/validation sets.")
        else:
            print("⚠️ WARNING: 'EssayID' column not found in all files. Skipping this check.")

        # --- Step 4: Check for near-duplicate leakage using TF-IDF ---
        print("\n--- CHECK 3: NEAR-DUPLICATE SENTENCE LEAKAGE (TF-IDF) ---")
        
        print("Vectorizing training and test data... (This may take a moment)")
        # Initialize and fit the vectorizer on the training data
        vectorizer = TfidfVectorizer()
        # train_vectors = vectorizer.fit_transform(combined_train_df['Text'])
        train_vectors = vectorizer.fit_transform(train_df['Text'])
        
        # Transform the test data using the same vectorizer
        test_vectors = vectorizer.transform(test_df['Text'])

        print("Calculating cosine similarity between test and training sets...")
        # Calculate the similarity matrix
        similarity_matrix = cosine_similarity(test_vectors, train_vectors)

        # For each test sentence, find its highest similarity score against all training sentences
        max_similarities = similarity_matrix.max(axis=1)

        # Find the indices of test sentences that exceed our high similarity threshold
        near_duplicate_indices = np.where(max_similarities > SIMILARITY_THRESHOLD)[0]

        if len(near_duplicate_indices) > 0:
            print(f"🔴 Found {len(near_duplicate_indices)} rows in the test set that are highly similar (similarity > {SIMILARITY_THRESHOLD}) to sentences in the training/validation set.")
            print("--- Example Near-Duplicate Rows Found in Test Set ---")
            print(test_df.iloc[near_duplicate_indices].head())
        else:
            print(f"✅ SUCCESS: No near-duplicate sentences found with a similarity threshold of > {SIMILARITY_THRESHOLD}.")

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: File not found. Please ensure the filenames in the CONFIGURATION section are correct.")
        print(f"Missing file: {e.filename}")
    except KeyError as e:
        print(f"\n❌ ERROR: A required column was not found in one of the CSV files.")
        print(f"Missing column: {e}")

# --- Run the check ---
if __name__ == "__main__":
    find_data_leakage()
