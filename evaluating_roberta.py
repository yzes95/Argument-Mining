import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer
from sklearn.metrics import classification_report

# --- Configuration ---
# TODO: Set the path to the saved, fine-tuned RoBERTa model directory.
MODEL_PATH = "roberta_argument_classifier_combined_cv_8" 

# TODO: Define the paths to the testing data files
TEST_CSV_PATH_AAEC = "testing_10_percent_processed_corpus_part_1.csv"
TEST_CSV_PATH_UKP = "ukp_testing_processed_corpus.csv"

# TODO: Define the names of the labels in the correct order (0, 1, ...).
CLASS_NAMES = ['Non-Argumentative', 'Argumentative']


def load_model_and_tokenizer(model_path):
    """
    Loads the fine-tuned model and tokenizer from a specified directory.
    """
    print(f"--- Loading model and tokenizer from {model_path} ---")
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    print(f"Model and tokenizer loaded successfully.")
    # The Trainer will handle device placement (GPU/CPU) automatically.
    return model, tokenizer


def load_and_prepare_test_data(csv_path):
    """
    Loads a CSV file and prepares it for the model.
    """
    print(f"--- Loading test data from {csv_path} ---")
    test_df = pd.read_csv(csv_path)
    # Renameing Type field.
    # The Trainer expects the label column to be named 'labels'.
    if 'Type' in test_df.columns:
        test_df.rename(columns={'Type': 'labels'}, inplace=True)

    # Convert string labels to integers if they aren't already
    if test_df['labels'].dtype == 'object':
        label_map = {name: i for i, name in enumerate(CLASS_NAMES)}
        test_df['labels'] = test_df['labels'].map(label_map)

    test_dataset = Dataset.from_pandas(test_df)
    return test_dataset


# A helper function to tokenize the data. The Trainer handles tensor creation.
def tokenize_function(examples):
    return tokenizer(examples["Text"], padding="max_length", truncation=True)


def evaluate_model(model, tokenizer, test_dataset):
    """
    Evaluates the model on the test dataset using the Trainer and prints a classification report.
    """
    print("\n--- Evaluating Model Performance using the Trainer ---")


    # Tokenize the entire dataset
    tokenized_dataset = test_dataset.map(tokenize_function, batched=True)

    # The Trainer only needs 'input_ids', 'attention_mask', and 'labels'.
    tokenized_dataset = tokenized_dataset.remove_columns(["Text"])

    # Instantiate a Trainer. We only need it for the .predict() method.
    # It will automatically use the GPU if it's available.
    trainer = Trainer(model=model)

    # Get model predictions on the test set
    prediction_output = trainer.predict(tokenized_dataset)

    # The raw predictions are logits. Use argmax to get the predicted class ID.
    y_preds = np.argmax(prediction_output.predictions, axis=1)
    
    # The true labels are in the 'label_ids' field
    y_true = prediction_output.label_ids

    # Generate and print the final report
    print("Classification Report:")
    print(classification_report(y_true, y_preds, target_names=CLASS_NAMES))


# --- Main Execution Block ---
if __name__ == "__main__":
    # 1. Load the trained model and tokenizer
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)

    # 2. Load the specific test data you want to evaluate
    test_dataset = load_and_prepare_test_data(TEST_CSV_PATH_UKP)

    # 3. Run the evaluation
    evaluate_model(model, tokenizer, test_dataset)
