from svm_model_training import evaluate_model,load_data
import joblib

TF_IDF_TEST_FEATURES_PATH_PART_1 = "tf_idf_final_features_test_part_1_aaec.npz"
TF_IDF_TEST_LABELS_PATH_PART_1 = "tf_idf_final_labels_test_part_1_aaec.npy"
TF_IDF_TEST_FEATURES_PATH_PART_2 = "tf_idf_final_features_test_part_2_aaec.npz"
TF_IDF_TEST_LABELS_PATH_PART_2 = "tf_idf_final_labels_test_part_2_aaec.npy"
TF_IDF_TEST_FEATURES_PATH_PART_3 = "tf_idf_final_features_test_part_3_ukp.npz"
TF_IDF_TEST_LABELS_PATH_PART_3 = "tf_idf_final_labels_test_part_3_ukp.npy"
TF_IDF_COMBINED_TEST_FEATURES_PATH_PART_1 = "tf_idf_final_features_test_part_1_combined_aaec.npz"
TF_IDF_COMBINED_TEST_LABELS_PATH_PART_1 = "tf_idf_final_labels_test_part_1_combined_aaec.npy"
TF_IDF_COMBINED_TEST_FEATURES_PATH_PART_2 = "tf_idf_final_features_test_part_2_combined_aaec.npz"
TF_IDF_COMBINED_TEST_LABELS_PATH_PART_2 = "tf_idf_final_labels_test_part_2_combined_aaec.npy"
TF_IDF_COMBINED_TEST_FEATURES_PATH_PART_3 = "tf_idf_final_features_test_part_3_combined_ukp.npz"
TF_IDF_COMBINED_TEST_LABELS_PATH_PART_3 = "tf_idf_final_labels_test_part_3_combined_ukp.npy"

print("\n--- Final Evaluation on the Unseen Test Set ---")
hybrid_svm_model = joblib.load("tf_idf_svm_argument_classifier_combined_cv_7_gamma_001_no_linear.joblib")

testing_features, testing_labels = load_data(
        TF_IDF_TEST_FEATURES_PATH_PART_1, TF_IDF_TEST_LABELS_PATH_PART_1
    )
# print(testing_labels_part_3)
evaluate_model(hybrid_svm_model,testing_features,testing_labels)