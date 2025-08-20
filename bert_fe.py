import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

# --- Configuration ---
# The clean, processed data file created in the data processing stage.
TRAINING_CSV_PATH = "training_processed_corpus.csv"
COMBINED_TRAINING_CSV_PATH = "combined_training_processed_corpus.csv"
TESTING_CSV_PATH_PART_1 = "Testing_10_percent_processed_corpus_part_1.csv" 
TESTING_CSV_PATH_PART_2 = "Testing_10_percent_processed_corpus_part_2.csv" 
TESTING_CSV_PATH_PART_3 = "ukp_testing_processed_corpus.csv"

# The name of the pre-trained BERT model we will use from Hugging Face.
BERT_MODEL_NAME = "bert-base-uncased"

def load_processed_data(filepath):
    """
    Loads the processed data from a CSV file into a pandas DataFrame.
    - PARAMETER: filepath (string) - The path to the input CSV file.
    - RETURNS: A pandas DataFrame with 'text' and 'label' columns.
    """
    # TODO: Use pd.read_csv() to load the data. Handle potential errors.
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def model_initializer(model_name):
    # It's good practice to check if a GPU is available and use it if possible.
    # This process is much faster on a GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 1. Initialize the Tokenizer for the chosen BERT model.
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # 2. Initialize the pre-trained BERT model itself.
    model = BertModel.from_pretrained(model_name)
    # 3. Move the model to the correct device (GPU or CPU).
    model.to(device)
    # 4. Set the model to evaluation mode. This turns off training-specific layers
    #    like dropout and is crucial for getting consistent results.
    model.eval()
    return (tokenizer,model,device)


def generate_bert_embeddings(corpus,tokenizer,model,device):
    """
    Generates a BERT embedding for each text segment in the corpus.
    - PARAMETER 1: corpus (pandas Series) - A pandas Series containing all text segments.
    - PARAMETER 2: model_name (string) - The name of the Hugging Face model to use.
    - RETURNS: A numpy array where each row is the embedding for a segment.
    """
    # TODO:
    # 5. Create an empty list to store the final embedding for each segment.
    embeddings_list = []
    # 6. Loop through the corpus/(text segments).
    print("Generating embeddings... (This may take a while)")

    for text in corpus:
        # 7. For each text, you need to tokenize it. This converts the string into
        #    the numerical format BERT expects.
        #    `return_tensors='pt'` returns PyTorch tensors.
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=128
        )
        # 8. Move the tokenized inputs to the same device as the model.
        inputs = {key: val.to(device) for key, val in inputs.items()}
        # 9. Get the embeddings from the model. 
        # `with torch.no_grad():` save memory and computation,
        with torch.no_grad():
            outputs = model(**inputs)
        # 10. Extract the embedding. 
        #     `.cpu().numpy()` moves the data back to the CPU and converts it to a numpy array.
        """
        The result (outputs.last_hidden_state) is a "tensor" that lives in the GPU's memory.

        The next line of code, .numpy(), tries to convert this tensor into a NumPy array, which is a data structure that lives in the computer's main memory (CPU memory).

        You cannot directly convert a GPU tensor to a NumPy array. You first have to tell PyTorch to copy the data from the GPU back to the CPU.
        """
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        # 11. Append the resulting numpy array to `embeddings_list`.
        embeddings_list.append(cls_embedding)
    # 12. After the loop, stack all the individual embeddings into a single numpy matrix.
    return np.vstack(embeddings_list)


def main():
    # --- Initialization ---
    tokenizer,model,device = model_initializer(BERT_MODEL_NAME)
    # --- Step 1: Load the clean data ---
    df = load_processed_data(TESTING_CSV_PATH_PART_3)
    if df is not None:
        df["label"] = df["Type"].map({"Argumentative": 1, "Non-Argumentative": 0})
        corpus = df["Text"]
        # --- Step 2: Generate Embeddings ---
        # TODO: Call `generate_bert_embeddings` function.
        feature_matrix = generate_bert_embeddings(corpus,tokenizer,model,device)
        # --- Step 3: Get and Save the Labels ---
        labels = (
            df["label"].to_numpy().astype(int) 
        )
        # --- Final Output ---
        # TODO: Save the `feature_matrix` and `labels` to disk using np.save().
        # These will be the inputs for your SVM training script.
        np.save("bert_final_features_test_part_3.npy", feature_matrix)
        np.save("bert_final_labels_test_part_3.npy", labels)

        

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting BERT Feature Engineering...")
    main()
    print("\nBERT feature engineering guide complete.")



