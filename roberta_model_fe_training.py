import pandas as pd
from datasets import Dataset, DatasetDict, ClassLabel
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

# --- Configuration ---
# TODO: Define the paths to your processed data files.
TRAIN_CSV_PATH = "RoBERTa_training_dataset.csv"
VALIDATION_CSV_PATH = "RoBERTa_validation_dataset.csv"
TEST_CSV_PATH_AAEC = "testing_10_percent_processed_corpus_part_1.csv"
TEST_CSV_PATH_UKP = "ukp_testing_processed_corpus.csv"

# TODO: Define the name of the pre-trained RoBERTa model from Hugging Face.
MODEL_NAME = 'roberta-base'

# TODO: Define the directory where the final trained model will be saved.
CHECKPOINT_OUTPUT_DIR = "roberta_argument_classifier_combined_cv_8_cuml_chkpoints"
FINAL_DATA_OUTPUT_DIR = "roberta_argument_classifier_combined_cv_8_cuml"

# --- Step 1: Load and Prepare the Data ---
def load_and_prepare_datasets(train_path, val_path, test_path):
    """
    Loads your CSV files and converts them into the Hugging Face `datasets` format.
    """
    # TODO: Use pandas to load three CSV files into DataFrames.
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    # The Hugging Face Trainer specifically looks for a column named 'labels'.
    train_df.rename(columns={'Type': 'labels'}, inplace=True)
    val_df.rename(columns={'Type': 'labels'}, inplace=True)
    test_df.rename(columns={'Type': 'labels'}, inplace=True)

    # TODO: Convert each DataFrame into a `Dataset` object.
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    # TODO: Combine them into a single `DatasetDict`.
    raw_datasets = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })

    # # TODO: Convert the text labels to numerical ClassLabels for the Trainer.
    # # This step is crucial for the model to understand its classification task.
    class_label = ClassLabel(names=['Non-Argumentative', 'Argumentative'])
    raw_datasets = raw_datasets.cast_column("labels", class_label)
    
    return raw_datasets


# --- Step 2: Tokenization ---
# TODO: Initialize the tokenizer for the chosen RoBERTa model.
# TODO: Call the tokenizer on the 'Text' column. `padding` and `truncation` are essential.
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
def tokenize_function(examples):
    """
    A function that tokenizes a batch of text examples.
    """
    return tokenizer(examples["Text"], padding="max_length", truncation=True)


# --- Step 3: Define Evaluation Metrics ---
def compute_metrics(pred):
    """
    A function to compute the evaluation metrics for the Trainer.
    """
    # TODO: Get the true labels and the model's predictions.
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    
    # TODO: Calculate precision, recall, and F1-score using a macro average.
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    # TODO: Calculate accuracy.
    acc = accuracy_score(labels, preds)
    
    # TODO: Return a dictionary of the results.
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def main():
 # --- Load and Tokenize Data ---
    # TODO: Call  `load_and_prepare_datasets` function.
    raw_datasets = load_and_prepare_datasets(TRAIN_CSV_PATH, VALIDATION_CSV_PATH, TEST_CSV_PATH_UKP)
    
    # TODO:  apply the `tokenize_function` to all data splits.
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    # --- Initialize the Model ---
    # TODO: Load the `RobertaForSequenceClassification` model, specifying 2 labels.
    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # --- Define Training Arguments ---
    # TODO: Create an instance of `TrainingArguments` to configure the training process.
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_OUTPUT_DIR,
        eval_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=8, # Adjust based on your GPU
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1", # Use F1-score to select the best model
        report_to="none"
    )

    # --- Initialize the Trainer ---
    # TODO: Create the `Trainer`, passing it all the prepared components.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    # --- Train the Model ---
    print("\n--- Starting Fine-Tuning ---")
    trainer.train()

    # --- Final Evaluation on the Test Set ---
    print("\n--- Evaluating on the Test Set ---")
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    print("Final Test Set Results:")
    print(test_results)
    
    # --- Save the Model ---
    print("\n--- Saving the Fine-Tuned Model ---")
    trainer.save_model(FINAL_DATA_OUTPUT_DIR)


# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting RoBERTa Fine-Tuning Pipeline...")
    main()
    print(f"\nPipeline complete. Best model saved to {FINAL_DATA_OUTPUT_DIR}")
