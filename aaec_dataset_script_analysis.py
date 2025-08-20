"""
# Guide: Script for Processing the Argument Annotated Essays Corpus (AAEC)
# This script is a template design to the process of reading
# the raw AAEC dataset and transforming it into a clean, labeled format suitable for machine learning tasks.
# Each function has a comment block explaining its purpose, what parameters it expects, and what it should return.
"""

import os
import pandas as pd
import spacy

# --- Configuration ---
# This is where we set the path to dataset folder.
# It should point to the main folder that contains the 'train' and 'test' subdirectories.
BASE_DATA_PATH = "ArgumentAnnotatedEssays-2.0/ArgumentAnnotatedEssays-2.0/brat-project-final/brat-project-final"
TRAIN_TEST_SPLIT = "ArgumentAnnotatedEssays-2.0/ArgumentAnnotatedEssays-2.0/train-test-split.csv"

# --- Initialization ---
# It's good practice to load models or other heavy resources once at the start.
# This spaCy model will be used for sentence segmentation.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model 'en_core_web_sm' not found. Please run:")
    print("python -m spacy download en_core_web_sm")
    exit()


def load_essay_text(txt_filepath):
    """
    Reads the content of a single essay text file.
    - PARAMETER: txt_filepath (string) - The full path to a .txt file.
    - RETURNS: A string containing the entire text of the essay.
    """
    # TODO: Open the file, read its content, and return the text.
    # print(txt_filepath)
    try:
        with open(txt_filepath,encoding='utf-8') as file:
            # print(file.readlines())
            return file.read()
    except FileNotFoundError:
        print("File Doesn't Exist...")


def parse_annotation_file(ann_filepath):
    """
    Reads a .ann file and extracts all argumentative components (Claims and Premises).
    - PARAMETER: ann_filepath (string) - The full path to a .ann file.
    - RETURNS: A list of dictionaries. Each dictionary represents one argumentative
               component and should contain its text, start/end character positions,
               and a label ('Argumentative').
               Example: [{'text': 'Some claim text', 'start': 56, 'end': 180, 'label': 'Argumentative'}, ...]
    """
    # TODO:
    # 1. Checking if the annotation file exists.
    # 2. If it exists, open and read it line by line.
    # 3. For each line that starts with 'T' (a text-bound annotation):
    #    a. Parse the line to get the component type (e.g., 'Claim'), start/end positions, and the text.
    #    b. If the type is 'Claim' or 'Premise', create a dictionary with the required keys
    #       and append it to the `argumentative_components` list.
    file_contents = list()
    file_headers = ["Type","StartPos","EndPos","Text"]
    df = pd.DataFrame(file_contents, columns=file_headers)
    argumentative_components = df.to_dict(orient='records')
    try:
        with open(ann_filepath,encoding='utf-8') as file:
            for line in file:
                if line[0] == "T":
                    info_line = line.strip().split(maxsplit=4)
                    info_line[1] = "Argumentative"
                    info_line.pop(0)
                    file_contents.append(info_line)
        if len(file_contents)<1 :
            raise IndexError 
        elif not file_contents[0][3] :
            raise KeyError
            #here the returned data type from panda would be a dataframe that when printed would be inform of a table of lines for my data
        df = pd.DataFrame(file_contents, columns=file_headers)
            #"Structure the output as a list of records. Each record should be a separate dictionary representing one row of my DataFrame."
            #In database terminology, a single row of data (representing a single item like a person, a product, or in our case, a "claim") is often called a record. 
        argumentative_components = df.to_dict(orient='records')
            # import json
            # print(json.dumps(list_of_dicts, indent=2))
    except FileNotFoundError:
        print("File Doesn't Exist...")

    return argumentative_components


def identify_non_argumentative_segments(full_essay_text, argumentative_components):
    """
    Finds all the text in an essay that has NOT been labeled as argumentative
    and splits it into sentences.
    - PARAMETER 1: full_essay_text (string) - The complete text of the essay.
    - PARAMETER 2: argumentative_components (list) - The list of dictionaries from `parse_annotation_file`.
    - RETURNS: A list of dictionaries. Each dictionary represents one non-argumentative
               sentence and should contain its text and the label 'Non-Argumentative'. and start /end pos
    """
    last_end_pos = 0
    # non_argumentative_segments = []
    nonprocessed_nonargumentative_chunck = list()
    file_headers = ["Type","StartPos","EndPos","Text"]
    # TODO:
    # 1. Sort the `argumentative_components` list by their 'start' character position. This is crucial!
    # 2. Keep track of the end position of the last argumentative component you've seen (`last_end_pos`, starting at 0).
    # 3. Loop through the sorted components:
    #    a. Extract the chunk of text between `last_end_pos` and the `start` of the current component.
    #       This chunk is non-argumentative text.
    #    b. Update `last_end_pos` to be the `end` of the current component.
    # 4. After the loop, get any remaining text from `last_end_pos` to the end of the essay.
    # 5. You now have one or more large strings of non-argumentative text. Use the `nlp()` model
    #    from spaCy to process these strings and split them into individual sentences (`doc.sents`).
    # 6. For each sentence, create a dictionary with its text and the 'Non-Argumentative' label
    #    and append it to the `nonprocessed_nonargumentative_chunck` list.
    argumentative_components = sorted(argumentative_components,key=lambda x:int(x["StartPos"])) 
    # import json
    # print(json.dumps(argumentative_components, indent=2))
    for component in argumentative_components:
        end_pos = int(component["StartPos"])
        if end_pos >last_end_pos:
            Doc_Obj = nlp(full_essay_text[last_end_pos:end_pos].strip())
            for sentence in Doc_Obj.sents:
                nonprocessed_nonargumentative_chunck.append(["Non-Argumentative",f"{last_end_pos}",f"{last_end_pos+len(sentence.text.strip())}",sentence.text.strip()])
                last_end_pos = last_end_pos+len(sentence.text.strip())+1
            last_end_pos = int(component["EndPos"])+1
            if component == argumentative_components[-1]:
                Doc_Obj = nlp(full_essay_text[last_end_pos:].strip())
                for sentence in Doc_Obj.sents:
                    nonprocessed_nonargumentative_chunck.append(["Non-Argumentative",f"{last_end_pos}",f"{last_end_pos+len(sentence.text.strip())}",sentence.text.strip()])
                    last_end_pos = last_end_pos+len(sentence.text.strip())+1

    df = pd.DataFrame(nonprocessed_nonargumentative_chunck, columns=file_headers)
    # non_argumentative_segments = df.to_dict(orient='records')
    # import json
    # print(json.dumps(non_argumentative_segments, indent=2))
    return df.to_dict(orient='records')


def process_essay_directory(essay_id):
    """
    Processes an entire directory of essays (like 'train' or 'test').
    - PARAMETER: essay id.
    - RETURNS: A list containing all processed segments (both argumentative and
               non-argumentative) from all essays in the directory.
    """
    all_segments_from_directory = []

    # TODO:
    # 1. Define the paths for your 'train' and 'test' directories using `os.path.join()`.
    #    a. Construct the full paths for both the .txt and .ann files.
    #    b. Call `load_essay_text()` to get the essay content.
    #    c. Call `parse_annotation_file()` to get the argumentative components.
    #    d. Call `identify_non_argumentative_segments()` to get the non-argumentative sentences.
    #    e. Combine the two lists of segments and extend the `all_segments_from_directory` list with them.

    # print(essay_id)
    txt_path = os.path.join(BASE_DATA_PATH, f"{essay_id}.txt")
    ann_path = os.path.join(BASE_DATA_PATH, f"{essay_id}.ann")
    # print(txt_path,ann_path)
    full_essay_text = load_essay_text(txt_path)
    argumentative_components = parse_annotation_file(ann_path)
    non_argumentative_components = identify_non_argumentative_segments(full_essay_text,argumentative_components)
    all_segments_from_directory.extend(argumentative_components)
    all_segments_from_directory.extend(non_argumentative_components)
    if type(full_essay_text) != str:
        raise FileNotFoundError
    return [all_segments_from_directory,len(full_essay_text.strip())]


def main():
    # TODO:
    # 2. Call `process_essay_directory()` for your training data.
    # 3. Call `process_essay_directory()` for your testing data.
    # 4. (Optional) Combine the train and test lists into one big list.
    # 5. Convert your final list of segments into a pandas DataFrame.
    # 6. Print some information about the DataFrame, like the total number of segments
    #    and the distribution of labels (how many 'Argumentative' vs. 'Non-Argumentative').
    # 7. Save the DataFrame to a CSV file (e.g., "processed_corpus.csv"). This file will
    #    be the input for your machine learning models in the next phases of your project.

    # identify_non_argumentative_segments(load_essay_text(BASE_DATA_PATH + "/essay001.txt"),parse_annotation_file(BASE_DATA_PATH+ "/essay001.ann"))
    # essays_file_path = os.path.join(BASE_DATA_PATH, "essay")
    
    test_train_csv_split_df = pd.read_csv(TRAIN_TEST_SPLIT,sep=";")
    # print(test_train_csv_split_df)
    train_ids = test_train_csv_split_df[test_train_csv_split_df['SET'] == 'TRAIN']['ID'].tolist()
    test_ids = test_train_csv_split_df[test_train_csv_split_df['SET'] == 'TEST']['ID'].tolist()
    print(f"Found {len(train_ids)} essays for training.")
    print(f"Found {len(test_ids)} essays for testing.")

    training_data = []
    for essay_id in train_ids:
        # Build the specific file paths for this essay
        temp_df = pd.DataFrame(process_essay_directory(essay_id)[0])
        temp_df["EssayID"] = essay_id
        temp_df["EssayLength"] = process_essay_directory(essay_id)[1]
        training_data.extend(temp_df.to_dict('records'))
    df_training_dataset = pd.DataFrame(training_data)        
    # index=False is important to prevent pandas from writing an extra unnecessary index column to your file.
    df_training_dataset.to_csv("Training_processed_corpus.csv", index=False)
    print("\nData successfully saved to Training_processed_corpus.csv!")

    testing_data = []
    for essay_id in test_ids:
        # Build the specific file paths for this essay
        temp_df = pd.DataFrame(process_essay_directory(essay_id)[0])
        temp_df["EssayID"] = essay_id
        temp_df["EssayLength"] = process_essay_directory(essay_id)[1]
        testing_data.extend(temp_df.to_dict('records'))  
    df_testing_dataset = pd.DataFrame(testing_data)  
    # index=False is important to prevent pandas from writing an extra unnecessary index column to your file.
    df_testing_dataset.to_csv("Testing_processed_corpus.csv", index=False)
    print("Data successfully saved to Testing_processed_corpus.csv!\n")

    print(f"Total segments processed: {len(df_testing_dataset)+len(df_training_dataset)}")
    print(f"Total Training segments processed: {len(df_training_dataset)}") 
    print(f"Total Testing segments processed: {len(df_testing_dataset)}")

    print("\nClass Distribution for Training Set:")
    print(df_training_dataset['Type'].value_counts().to_string())     

    print("\nClass Distribution for Testing Set:")
    print(df_testing_dataset['Type'].value_counts().to_string())     


if __name__ == "__main__":

    print("Starting data processing guide...")
    main()
   