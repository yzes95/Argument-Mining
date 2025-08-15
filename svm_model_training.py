#
# Guide: Script for Training and Evaluating an SVM Model
#
# This script is a template designed to guide you through the process of training
# your classic SVM model. It will load the features you engineered, find the
# best settings for the model, and evaluate its final performance.
#

import numpy as np
from scipy.sparse import load_npz
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
# from cuml.svm import SVC 
from sklearn.metrics import classification_report
from joblib import dump

# --- Configuration ---
# Paths to the feature and label files created by your feature engineering script.
# Make sure you have separate files for your train, validation, and test sets.
TF_IDF_TRAIN_FEATURES_PATH = "tf_idf_final_features_train.npz"
TF_IDF_TRAIN_LABELS_PATH = "tf_idf_final_labels_train.npy"
TF_IDF_TRAIN_FEATURES_PATH_COMBINED = "tf_idf_final_features_train_combined.npz"
TF_IDF_TRAIN_LABELS_PATH_COMBINED = "tf_idf_final_labels_train_combined.npy"
# --- BERT Configuration ---
# Paths to the feature and label files created by your feature engineering script.
# Make sure you have separate files for your train, validation, and test sets.
BERT_TRAIN_FEATURES_PATH = "bert_final_features_train.npy"
BERT_TRAIN_LABELS_PATH = "bert_final_labels_train.npy"
BERT_TRAIN_FEATURES_PATH_COMBINED = "bert_final_features_train_combined.npy"
BERT_TRAIN_LABELS_PATH_COMBINED = "bert_final_labels_train_combined.npy"

# --- Helper Functions ---
def load_data(features_path:str, labels_path:str):
    """
    Loads the feature matrix and label array from disk.
    - PARAMETERS: Paths to the .npz (features) and .npy (labels) files.
    - RETURNS: The feature matrix (X) and the label array (y).
    """
    # TODO:
    # 1. Use `load_npz()` from `scipy.sparse` to load the feature matrix.
    # 2. Use `np.load()` from `numpy` to load the label array.
    # 3. Return both.
    # 4. Include error handling (try-except) in case the files don't exist.
    try:
        print("Fetching Features File...!")
        if features_path.endswith(".npz"):
            features = load_npz(features_path)
        else:
            features = np.load(features_path)
        print("Fetching Labels File...!")
        labels = np.load(labels_path)
        return [features, labels]
    except FileNotFoundError:
        print("File not found...!")
        exit()


def evaluate_model(model, X_test, y_test):
  
    # Evaluates a trained model on the test set and prints a report.
    # - PARAMETERS: The trained model, the test features (X_test), and test labels (y_test).
    
    # TODO:
    # 1. Use 'model.predict()' on X_test to get the model's predictions.
    # 2. Use 'classification_report()' from 'sklearn.metrics' to print a detailed
    #    report including precision, recall, and F1-score for each class.
    #    The 'target_names' parameter can be used to show 'Argumentative' etc.
    # 3. Print a final, clear statement with the overall F1-score.
    features_predictions = model.predict(X_test)

    # This line makes the report much easier to read.
    Target_names = ["Non-Argumentative", "Argumentative"]
    report = classification_report(
        y_test, features_predictions, target_names=Target_names
    )
    print(report)

# --- Main Training and Tuning Function ---
# The use of X_train and y_train is a very strong and widely followed convention.
# X: By convention, a capital X is used to represent the features.
# y: By convention, a lowercase y is used to represent the target or the labels.
def train_and_tune_svm(X_train, y_train):
    """
    Trains an SVM and uses GridSearchCV to find the best hyperparameters.
    - PARAMETERS: Your training.
    - RETURNS: The best-performing, tuned SVM model.
    """
    print("--- Starting Hyperparameter Tuning with GridSearchCV ---")

    # TODO:
    # 1. Define the "parameter grid". This is a dictionary where keys are the
    #    hyperparameter names (like 'C' and 'kernel') and values are the lists
    #    of settings you want to try.
    #    Example: param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

    # 2. Initialize GridSearchCV. You will pass it:
    #    - An SVM model instance: `SVC(random_state=42)`
    #    - The `param_grid` you just defined.
    #    - A scoring metric to optimize for: `scoring='f1_macro'` is a good choice.
    #    - `cv=3` (for 3-fold cross-validation on the training data).
    #    - `n_jobs=-1` (to use all available CPU cores).

    # 3. "Fit" the GridSearchCV object to your TRAINING data (X_train, y_train).
    #    This will automatically train and test all combinations of parameters.
    #    NOTE: GridSearchCV uses cross-validation, so it doesn't need the separate
    #    validation set (X_val, y_val) for this step. The validation set is for
    #    a final check if you weren't using GridSearchCV.

    # 4. After fitting, print the best parameters found by the search.
    #    Hint: 'grid_search.best_params_'

    # 5. The best model is automatically saved in the GridSearchCV object.
    #    Return the best estimator: 'grid_search.best_estimator_'

    """
    param_grid = [
        {
        "kernel": ["linear","rbf", "poly"], 
        "C": [0.1, 1, 10, 100]
        }
    ]
    """
    param_grid = [
        # First dictionary: for the linear kernel
        {"kernel": ["linear"], "C": [0.1, 1, 10, 100]},
        # Second dictionary: for the rbf and poly kernels
        {
            "kernel": ["rbf", "poly"],
            "C": [0.1, 1, 10, 100],
            "gamma": [0.001],
        },
    ]

    # Why 42? The number itself is completely arbitrary.
    # The important thing is not the number itself, but the fact that you use the same number every time to get consistent, reproducible results.
    svm_model = SVC(random_state=42)

    # grid_search = GridSearchCV(...)
    # grid_search.fit(...)
    grid_search = GridSearchCV(
        estimator=svm_model,
        param_grid=param_grid,
        scoring="f1_macro",  # The metric to optimize for
        cv=8,  # Use 7-fold cross-validation
        verbose=2,  # print/Shows progress
        n_jobs=-1,
    )

    # This one line runs all the experiments.
    # It will automatically train and test multiple SVM models.
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    return best_model


def main():

    # --- Step 1: Load all datasets ---
    # TODO: Call your `load_data` function three times to load the train,
    # validation, and test sets.
    training_features, training_labels = load_data(
        TF_IDF_TRAIN_FEATURES_PATH_COMBINED, TF_IDF_TRAIN_LABELS_PATH_COMBINED
    )
    # --- Step 2: Train and Tune the Model ---
    # TODO: Call your `train_and_tune_svm` function, passing it the training
    # It will return your final, best-performing model.
    model_results = train_and_tune_svm(training_features, training_labels)
    
    # --- Save the model to a file ---
    model_filename = "tf_idf_svm_argument_classifier_cv_8_gamma_combined.joblib"
    print(f"\nSaving the best model to {model_filename}...")
    dump(model_results, model_filename)
    print("Model saved successfully.")
    

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting SVM Training and Evaluation Pipeline...")
    main()
    print("\nPipeline complete.")

