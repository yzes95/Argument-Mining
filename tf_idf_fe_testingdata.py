import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix, save_npz
import numpy as np
import joblib
import random
import csv

# --- Configuration ---
# The clean, processed data file created in the previous stage.
TESTING_CSV_PATH_PART_2 = "testing_10_percent_processed_corpus_part_2.csv"
TESTING_CSV_PATH_PART_1 = "testing_10_percent_processed_corpus_part_1.csv"
TESTING_CSV_PATH_PART_3 = "ukp_testing_processed_corpus.csv"
TF_IDF_VECTORIZER = "tf_idf_vectorizer.joblib"
TF_IDF_VECTORIZER_COMBINED = "tf_idf_vectorizer_combined.joblib"

# --- Initialization ---
# Load the spaCy model once for use in syntactic feature extraction.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model 'en_core_web_sm' not found. Please run:")
    print("python -m spacy download en_core_web_sm")
    exit()

# --- Feature Extraction Functions ---


def extract_ngram_features(processed_data):
    ngram_features = joblib.load(TF_IDF_VECTORIZER_COMBINED).transform(processed_data) 
    return ngram_features


def extract_lexicon_features(processed_data:pd.DataFrame):
    """
    Generates features based on a custom lexicon of argumentative indicators.
    - PARAMETER: corpus (pandas Series) - A pandas Series of text segments.
    - RETURNS: A numpy array where each row has one feature: the count of indicator words.
    """
    ARGUMENTATIVE_INDICATORS = {
        "because",
        "since",
        "for",
        "as",
        "therefore",
        "thus",
        "consequently",
        "so",
        "should",
        "must",
        "ought",
        "clearly",
        "evidence",
        "proves",
        "demonstrates",
        "reason",
    }

    # TODO:
    # 1. Create an empty list to hold the counts for each segment.
    # 2. Loop through each `text` in the `data_segment`.
    # 3. For each text, count how many words are in ARGUMENTATIVE_INDICATORS set.
    # 4. Append the count to the list.
    # 5. After the loop, convert the list of counts into a numpy array and return it.
    #    Hint: 'np.array(counts).reshape(-1, 1)' will create a column vector.
    Argumentative_statments_count = list()
    count = 0
    processed_data = processed_data.to_dict('records')
    for segment in processed_data:
        words = segment["Text"].lower().split()
        for word in words:
            if word in ARGUMENTATIVE_INDICATORS:
                count += 1
        Argumentative_statments_count.append(count)
        count = 0
    # print(Argumentative_statments_count)
    # print(np.array(Argumentative_statments_count).reshape(-1, 1))
    return np.array(Argumentative_statments_count).reshape(-1, 1)


def extract_structural_features(processed_data:pd.DataFrame):
    """
    Generates features based on the structure and location of segments within each essay.
    - PARAMETER: df (pandas DataFrame) - The full DataFrame, which should include columns for essay_id, sentence_index ++
    - RETURNS: A numpy array of structural features for each segment.
    """
    # TODO:
    # 1. Create an empty list to hold the feature dictionaries for each row.
    # 2. Loop through the rows of the DataFrame (`for index, row in df.iterrows():`).
    # 3. For each row, calculate features like:
    #    a. Segment length in characters: `len(row['text'])`.
    #    b. Segment length in words: `len(row['text'].split())`.
    #    c. Normalized position in essay (cancaled as it un-doable for the ukp data set)
    # 4. Append a dictionary of these features to your list.
    # 5. Convert the list of dictionaries into a DataFrame, then into a numpy array and return it.
    Features_List = list()
    processed_data = processed_data.to_dict('records')
    for segement in processed_data:
        features = {
            "seg_char_len": len(segement["Text"]),
            "seg_words_count": len(segement["Text"].split()), 
        }
        Features_List.append(features)
    # print(Features_List)
    # print(pd.DataFrame(Features_List).to_numpy())
    return pd.DataFrame(Features_List).to_numpy()


def extract_syntactic_features(processed_data:pd.DataFrame):
    """
    Generates features based on the grammatical structure of each text segment.
    - PARAMETER: processed_data (pandas Series) - A pandas Series of text segments.
    - RETURNS: A numpy array of syntactic features for each segment.
    """
    # TODO:
    # 1. Create an empty list to hold the feature dictionaries for each segment.
    # 2. Loop through each `text` in the `corpus`.
    # 3. For each text, create a spaCy `doc` object: `doc = nlp(text)`.
    # 4. From the `doc` object, calculate features like:
    #    a. The count of nouns (`token.pos_ == 'NOUN'`).
    #    b. The count of verbs (`token.pos_ == 'VERB'`).
    #    c. The count of adjectives (`token.pos_ == 'ADJ'`).
    # 5. Append a dictionary of these counts to the list.
    # 6. Convert the list of dictionaries into a DataFrame, then into a numpy array and return it.
    processed_data = processed_data.to_dict('records')
    Features_List = list()
    count_of_nouns = 0
    count_of_verbs = 0
    count_of_adjec = 0
    for segement in processed_data:
        doc = nlp(segement["Text"])
        for token in doc:
            if token.pos_ == "NOUN":
                count_of_nouns += 1
            elif token.pos_ == "VERB":
                count_of_verbs += 1
            elif token.pos_ == "ADJ":
                count_of_adjec += 1
        features = {
            "count_of_nouns": count_of_nouns,
            "count_of_verbs": count_of_verbs,
            "count_of_adjec": count_of_adjec,
        }
        Features_List.append(features)
        count_of_nouns = 0
        count_of_verbs = 0
        count_of_adjec = 0
    # print(Features_List)
    # print(pd.DataFrame(Features_List).to_numpy())
    return pd.DataFrame(Features_List).to_numpy()


def main():
    # --- Step 1: Load the clean data ---
    # TODO: load  `train_corpus.csv`.
    processed_data = pd.read_csv(TESTING_CSV_PATH_PART_3)
    processed_data_df = pd.DataFrame(processed_data)
    print(f"Loaded {len(processed_data_df)} segments from {TESTING_CSV_PATH_PART_3}")
    # --- Step 2: Run Each Feature Extraction Function ---
    print("Extracting n-gram features...")
    ngram_features = extract_ngram_features(processed_data_df["Text"])
    print(f"N-gram feature matrix shape: {ngram_features.shape}")

    # TODO: Call the other feature extraction functions here
    lexicon_features = extract_lexicon_features(processed_data_df)
    structural_features = extract_structural_features(processed_data_df)
    syntactic_features = extract_syntactic_features(processed_data_df)

    # --- Step 3: Combine All Features into One Matrix ---
    print("Combining all feature sets...")
    # TODO: combine feature matrices 
    # The `hstack` function from scipy.sparse is the best way to do this.
    final_feature_matrix = hstack(
        [
            ngram_features,
            csr_matrix(lexicon_features),
            csr_matrix(structural_features),
            csr_matrix(syntactic_features),
        ]
    )
    print(f"Final combined feature matrix shape: {final_feature_matrix.shape}")

    # --- Step 4: Get the Labels ---
    labels = (
        processed_data_df["Type"]
        .map({"Argumentative": 1, "Non-Argumentative": 0})
        .to_numpy()
    )
    # --- Final Output ---
    # These are the inputs you will feed into SVM model for training.
    save_npz("tf_idf_final_features_test_part_3_combined.npz", final_feature_matrix)
    np.save("tf_idf_final_labels_test_part_3_combined.npy", labels)


# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting Feature Engineering guide...")
    main()
    print("\nFeature engineering complete. Features and labels saved to disk.")

