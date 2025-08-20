from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any
import joblib
import spacy
import numpy as np
from scipy.sparse import hstack, csr_matrix
from typing import Literal
from transformers import (
    BertTokenizer,
    BertModel,
    Trainer,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
)
import pandas as pd
import torch
from xaif import AIF as aif
from datasets import Dataset

# The name of the pre-trained BERT model we will use from Hugging Face.

BERT_MODEL_NAME = "bert-base-uncased"
BERT_SVM_MODEL_PATH = "bert_svm_argument_classifier_cv_8_gamma_combined.joblib"


def model_initializer(model_name):
    # It's good practice to check if a GPU is available and use it if possible.
    # This process is much faster on a GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 1. Initialize the Tokenizer for your chosen BERT model.
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # 2. Initialize the pre-trained BERT model itself.
    model = BertModel.from_pretrained(model_name)
    # 3. Move the model to the correct device (GPU or CPU).
    model.to(device)
    # 4. Set the model to evaluation mode. This turns off training-specific layers
    #    like dropout and is crucial for getting consistent results.
    model.eval()
    return (tokenizer, model, device)


def generate_bert_embeddings(corpus, tokenizer, model, device):
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
        # 9. Get the embeddings from the model. It's important to do this inside
        #    a `with torch.no_grad():` block to save memory and computation,
        #    as we are not training the model here.
        with torch.no_grad():
            outputs = model(**inputs)
        # 10. Extract the embedding. A common strategy is to take the embedding of the
        #     special [CLS] token, which is the first token in the sequence. This
        #     embedding is designed to represent the meaning of the entire sentence.
        #     `.cpu().numpy()` moves the data back to the CPU and converts it to a numpy array.
        # added .cpu since we are using gpu
        """
        The result (outputs.last_hidden_state) is a "tensor" that lives in the GPU's memory.

        The next line of code, .numpy(), tries to convert this tensor into a NumPy array, which is a data structure that lives in the computer's main memory (CPU memory).

        You cannot directly convert a GPU tensor to a NumPy array. You first have to tell PyTorch to copy the data from the GPU back to the CPU.
        """
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        # 11. Append the resulting numpy array to your `embeddings_list`.
        embeddings_list.append(cls_embedding)
    # 12. After the loop, stack all the individual embeddings into a single numpy matrix.
    return np.vstack(embeddings_list)


# --- Load components for Phase 2b: Hybrid BERT+SVM ---
try:
    hybrid_svm_model = joblib.load(BERT_SVM_MODEL_PATH)
    tokenizer, model, device = model_initializer(BERT_MODEL_NAME)
    print("Hybrid SVM and BERT components loaded.")
except FileNotFoundError:
    hybrid_svm_model = None
    print(
        "WARNING: Hybrid SVM model not found. The 'hybrid_svm' model will not be available."
    )





print("Text : The evidence clearly demonstrates the defendant's guilt, therefore the jury must convict.")
# Automated feature engineering using the BERT helper function.
embeddings = generate_bert_embeddings(
    "The evidence clearly demonstrates the defendant's guilt, therefore the jury must convict.", tokenizer, model, device
)
predictions = hybrid_svm_model.predict(embeddings)
print("BERT+SVM Predictions:", predictions)
