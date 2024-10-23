import os
import pandas as pd
from util.preprocessing import normalize
import constants as c

# Load the vocabulary from a .vocab file
def load_vocab(encoder, lang_type, lang):
    lang_type = "multilingual" if lang_type else "monolingual"
    vocab_file_path = f"{encoder}/{lang_type}/{lang}/{lang}.vocab"
    with open(vocab_file_path, 'r') as f:
        vocab = [line.strip() for line in f]  # Read and strip each line
    vocab2id = {term: i for i, term in enumerate(vocab)}  # Create mapping from term to index
    return vocab, vocab2id

def get_file_names(path, sufix):
    files = [f for f in os.listdir(str(path)) if f.endswith(str(sufix))]
    return files

def read_tsv_files(path, columns=3):
    # Read a tsv file based on a number of columns with the last one being the gold label
    tsv_files = get_file_names(str(path), '.tsv')

    sentence_pairs = []
    unique_target_variables = set()

    for file_name in tsv_files:
        file_path = os.path.join(str(path), file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")  # Split by slash
                if len(parts) == int(columns):
                    sentence1 = parts[0].strip()
                    sentence1 = normalize(sentence1)
                    
                    sentence2 = parts[1].strip()
                    sentence2 = normalize(sentence2)

                    target_variable = parts[2].strip()
                    
                    sentence_pairs.append((sentence1, sentence2, target_variable))
                    unique_target_variables.add(target_variable)
                else:
                    raise ImportError("Number of columns and data are not the same")
                
    return sentence_pairs, unique_target_variables

def save_multiclassifier_as_csv(filepath, exp_data, total_predictions, correct_predictions, not_embedded_sentences, accuracy):
    df = pd.DataFrame({
        'experiment': [exp_data], 
        'total_predictions': [total_predictions],
        'correct_predictions': [correct_predictions],
        'not_embedded_sentences': [not_embedded_sentences],
        'accuracy': [accuracy]
    })

    os.makedirs(c.RESULTS, exist_ok=True)
    
    if os.path.isfile(filepath):
        existing_df = pd.read_csv(filepath, sep=';')
        if exp_data in existing_df['experiment'].values:
            existing_df.update(df)
        else:
            existing_df = pd.concat([existing_df, df], ignore_index=True)
        existing_df.to_csv(filepath, sep=';', header=True, index=False, mode="w")
    else:
        df.to_csv(filepath, sep=';', header=True, index=False, mode="w")

    return None

def save_predictions(filepath, df, exp_data):

    os.makedirs(c.RESULTS, exist_ok=True)
    
    if os.path.isfile(filepath):
        existing_df = pd.read_csv(filepath, sep=';')
        existing_df[exp_data] = df[exp_data]
        existing_df.to_csv(filepath, sep=';', header=True, index=False, mode="w")
    else:
        df.to_csv(filepath, sep=';', header=True, index=False, mode="w")

    return None

def read_csv(path):
    csv_files = get_file_names(str(path), '.csv')
    df = pd.concat([pd.read_csv(os.path.join(str(path), file_name), sep=";") for file_name in csv_files],ignore_index=True)
    return df

    
