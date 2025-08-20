from svm_model_training import evaluate_model_gpu,load_data
import joblib

BERT_TEST_FEATURES_PATH_PART_1 = "bert_final_features_test_part_1_aaec.npy"
BERT_TEST_LABELS_PATH_PART_1 = "bert_final_labels_test_part_1_aaec.npy"
BERT_TEST_FEATURES_PATH_PART_2 = "bert_final_features_test_part_2_aaec.npy"
BERT_TEST_LABELS_PATH_PART_2 = "bert_final_labels_test_part_2_aaec.npy"
BERT_TEST_FEATURES_PATH_PART_3 = "bert_final_features_test_part_3_ukp.npy"
BERT_TEST_LABELS_PATH_PART_3 = "bert_final_labels_test_part_3_ukp.npy"

print("\n--- Final Evaluation on the Unseen Test Set ---")
hybrid_svm_model = joblib.load("bert_svm_argument_classifier_cv_8_gamma_combined .joblib")

testing_features, testing_labels = load_data(
        BERT_TEST_FEATURES_PATH_PART_3, BERT_TEST_LABELS_PATH_PART_3
    )
evaluate_model_gpu(hybrid_svm_model,testing_features,testing_labels)