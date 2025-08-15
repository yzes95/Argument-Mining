import os
import pandas as pd
import spacy

# --- Configuration ---
# TODO: Set this to the path where you unzipped the "brat-project-final" folder
# from the dataset.
# The structure should be: .../brat-project-final/train/... and .../brat-project-final/test/...
BASE_DATA_PATH = "ArgumentAnnotatedEssays-2.0/ArgumentAnnotatedEssays-2.0/brat-project-final/brat-project-final" 

# Load a spaCy model for sentence splitting. This is an essential step for finding
# the non-argumentative segments.
# You may need to run this command in your terminal first: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model 'en_core_web_sm' not found. Please run:")
    print("python -m spacy download en_core_web_sm")
    exit()

def process_brat_files(directory_path):
    """
    Reads all .txt and .ann files from a directory, processes them,
    and returns a list of dictionaries containing the segmented text and its label.
    """
    processed_data = []
    
    # Loop through every file in the given directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            essay_id = filename.split('.')[0]
            txt_filepath = os.path.join(directory_path, filename)
            ann_filepath = os.path.join(directory_path, f"{essay_id}.ann")

            # --- Step 1: Read the Raw Essay Text ---
            with open(txt_filepath, 'r', encoding='utf-8') as f:
                essay_text = f.read()

            # --- Step 2: Read the Annotations to get Argumentative Components ---
            argumentative_components = []
            if os.path.exists(ann_filepath):
                with open(ann_filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        # The .ann file format is: T1    Claim 56 180    Some text...
                        # We only care about lines starting with 'T' (Text-bound annotations)
                        if line.startswith('T'):
                            parts = line.strip().split('\t')
                            component_type = parts[1].split(' ')[0]
                            
                            # As per our plan, we treat both Claim and Premise as 'Argumentative'
                            if component_type in ['Claim', 'Premise']:
                                # Get the start and end character positions
                                start = int(parts[1].split(' ')[1])
                                end = int(parts[1].split(' ')[-1])
                                argumentative_components.append({
                                    'text': parts[2],
                                    'start': start,
                                    'end': end,
                                    'label': 'Argumentative'
                                })

            # Add the identified argumentative segments to our main data list
            processed_data.extend(argumentative_components)

            # --- Step 3: Identify Non-Argumentative Segments ---
            # This is the clever part. We find the text that is *not* covered by any
            # argumentative annotation and split it into sentences.
            
            # Sort components by their start position to make processing easier
            argumentative_components.sort(key=lambda x: x['start'])
            
            last_end = 0
            non_argumentative_text = ""
            
            for component in argumentative_components:
                # Get the text between the end of the last component and the start of this one
                non_arg_chunk = essay_text[last_end:component['start']]
                non_argumentative_text += non_arg_chunk
                last_end = component['end']

            # Get any remaining text after the last argumentative component
            non_argumentative_text += essay_text[last_end:]
            
            # Use spaCy to robustly split the collected non-argumentative text into sentences
            doc = nlp(non_argumentative_text)
            for sent in doc.sents:
                # Add each sentence as a non-argumentative segment
                if sent.text.strip(): # Avoid adding empty strings
                    processed_data.append({
                        'text': sent.text.strip(),
                        'label': 'Non-Argumentative'
                    })
                        
    # Filter out any dictionaries that don't have a label (e.g. from parsing errors)
    # and only keep the text and label for our final DataFrame
    final_data = [{'text': item['text'], 'label': item['label']} for item in processed_data if 'label' in item]
    
    return final_data


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting data processing...")

    # Define the paths for the train and test directories from the corpus
    train_dir = os.path.join(BASE_DATA_PATH, "train")
    test_dir = os.path.join(BASE_DATA_PATH, "test")

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"Error: Could not find 'train' and 'test' directories in '{BASE_DATA_PATH}'")
        print("Please make sure you have downloaded and unzipped the dataset and set the BASE_DATA_PATH correctly.")
    else:
        # Process both the training and testing files
        train_data = process_brat_files(train_dir)
        test_data = process_brat_files(test_dir) # You'd typically keep this separate

        # Combine them for a full view, or keep them separate for a real project
        all_data = train_data + test_data

        # --- Final Output: A Clean DataFrame ---
        # This is the "dopamine hit"! A clean, structured table ready for ML.
        df = pd.DataFrame(all_data)

        print(f"\nProcessing Complete!")
        print(f"Total segments processed: {len(df)}")
        print("\nClass Distribution:")
        print(df['label'].value_counts())

        print("\n--- First 5 Processed Segments ---")
        print(df.head())
        
        # You can now save this DataFrame to a CSV file to use in your other scripts
        df.to_csv("processed_aaec_corpus.csv", index=False)
        print("\nData saved to 'processed_aaec_corpus.csv'")

