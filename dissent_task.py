from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from constants import lang2model
from transformers import BertTokenizer
import constants as c
import numpy as np
import os
import argparse
from util.argparse import str2bool
import re

def load_layer_embeddings(encoder, lang_type, lang, num_layers=13, type="Total"):
    lang_type = "multilingual" if lang_type else "monolingual"
    layer_embeddings = []

    if type=="Total":
        for layer in range(int(num_layers)):
            emb_file_path = f"{encoder}/{lang_type}/{lang}/{lang}_{layer}.npy"
            print("Embedding from", emb_file_path)
            layer_embeddings.append(np.load(emb_file_path))
    elif type=="First":
        emb_file_path = f"{encoder}/{lang_type}/{lang}/aggr_first_k/{lang}_aggr_{num_layers}.npy"
        print("Embedding from", emb_file_path)
        layer_embeddings.append(np.load(emb_file_path))
    elif type=="Last":
        emb_file_path = f"{encoder}/{lang_type}/{lang}/aggr_last_k/{lang}_aggr_{num_layers}.npy"
        print("Embedding from", emb_file_path)
        layer_embeddings.append(np.load(emb_file_path))
    else:
        raise ImportError("No such file")
    
    return layer_embeddings

# Argument parsing
def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("-lang", "--language", type=str, required=True)  # Language (e.g., EN, DE, FI)
  parser.add_argument("-multiling", "--use_multiling_enc", type=str2bool, required=True)  # Use multilingual BERT
  parser.add_argument("-encoder", "--context_encoder", type=str, required=True)  # Select which encoder to use
  parser.add_argument("-tlayer", "--type_of_layer", type=str, required=False)  # Select how to average layers
  parser.add_argument("-ilayer", "--index_of_layer", type=str, required=False)  # Select which indeces to average
  return parser.parse_args()

# Load the vocabulary from a .vocab file
def load_vocab(encoder, lang_type, lang):
    lang_type = "multilingual" if lang_type else "monolingual"
    vocab_file_path = f"{encoder}/{lang_type}/{lang}/{lang}.vocab"
    with open(vocab_file_path, 'r') as f:
        vocab = [line.strip() for line in f]  # Read and strip each line
    vocab2id = {term: i for i, term in enumerate(vocab)}  # Create mapping from term to index
    return vocab, vocab2id

# Function to get the embedding for a sentence
def get_sentence_embedding(sentence, layer_embeddings, vocab2id):
    words = sentence.split()  # Assuming simple whitespace tokenization
    sentence_embeddings = []

    for word in words:
        if word in vocab2id:
            token_id = vocab2id[word]  # Get the token id from the vocabulary
            # HERE: Try different subword tokens

            embeddings = [layer[token_id] for layer in layer_embeddings]
            avg_embedding = np.mean(embeddings, axis=0)
            sentence_embeddings.append(avg_embedding)
        else:
            print(f"Word '{word}' not found in vocabulary.")
            continue

    if sentence_embeddings:
        return np.mean(sentence_embeddings, axis=0)
    else:
        return None 

# Function to find the most appropriate discourse marker
def find_best_disco_marker(sentence_embedding, markers, layer_embeddings, vocab2id):
    similarities = {}

    for marker in markers:
        marker_embedding = get_sentence_embedding(marker, layer_embeddings, vocab2id)  # Get marker embedding
        if marker_embedding is not None:
            # Compute cosine similarity
            similarity = cosine_similarity(sentence_embedding.reshape(1, -1), marker_embedding.reshape(1, -1))[0][0]
            similarities[marker] = similarity
        else:
            print(f"Marker '{marker}' could not be embedded.")
            continue

    # Select the marker with the highest similarity score
    best_marker = max(similarities, key=similarities.get)

    return best_marker

def normalize(sentence):
    sentence = sentence.lower()
    # sentence = sentence1.replace(" .", "")
    sentence= re.sub(r"[^a-zA-Z0-9 ]+", '', sentence) # Remove special characters
    return sentence

# Main function
if __name__ == "__main__":
    args = parse_arguments()

    if args.subword_encoder == "AOC":
        encoder = c.AOC_DIR
    elif args.subword_encoder == "ISO":
        encoder = c.ISO_DIR
    else:
        raise ImportError("Encoder is missing")

    data_dir = "./data/dissent/"
    tsv_files = [f for f in os.listdir(data_dir) if f.endswith('.tsv')]

    sentence_pairs = [] 
    unique_discourse_markers = set()

    for file_name in tsv_files:
        file_path = os.path.join(data_dir, file_name)
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")  # Split by slash
                
                if len(parts) == 3:
                    sentence1 = parts[0].strip() #Remove special tokens (normalization)
                    sentence1 = normalize(sentence1)
                    
                    sentence2 = parts[1].strip()
                    sentence2 = normalize(sentence2)

                    discourse_marker = parts[2].strip()
                    
                    sentence_pairs.append((sentence1, sentence2, discourse_marker))
                    unique_discourse_markers.add(discourse_marker)

    vocab, vocab2id = load_vocab(encoder, args.use_multiling_enc, args.language)
    if args.type_of_layer and args.index_of_layer:
        layer_embeddings = load_layer_embeddings(encoder, args.use_multiling_enc, args.language, args.index_of_layer, args.type_of_layer, )
    elif args.index_of_layer:
        layer_embeddings = load_layer_embeddings(encoder, args.use_multiling_enc, args.language, args.index_of_layer)
    else:
        layer_embeddings = load_layer_embeddings(encoder, args.use_multiling_enc, args.language)


    # Initialize prediction tracking
    correct_predictions = 0
    total_predictions = 0
    not_embedded_sentences = 0

    for idx, (s1, s2, actual_marker) in enumerate(sentence_pairs):
        s1_embedding = get_sentence_embedding(s1, layer_embeddings, vocab2id)
        s2_embedding = get_sentence_embedding(s2, layer_embeddings, vocab2id)

        if s1_embedding is not None and s2_embedding is not None:
            combined_embedding = np.mean(np.vstack([s1_embedding, s2_embedding]), axis=0)

        best_marker = find_best_disco_marker(combined_embedding, unique_discourse_markers, layer_embeddings, vocab2id)

        print(f"Sentence Pair Index {idx}: Best Marker - {best_marker}")

        # Compare predicted marker with the actual marker
        if best_marker == actual_marker:
            correct_predictions += 1
        
        total_predictions += 1
    else:
        print(f"Embedding for Sentence Pair {idx} could not be computed.")
        not_embedded_sentences += 1

    zero_one_loss = total_predictions - correct_predictions 
    accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0  # Accuracy calculation

    print(f"Total Predictions: {total_predictions}, Correct Predictions: {correct_predictions}, Accuracy: {accuracy:.2f}%")
    print(f"Zero-One Loss: {zero_one_loss}, Not embedded sentences: {not_embedded_sentences}")
        
#TODO:
#1) Subword tokenization (with args parsing)
#2) Take care of proper saving of results (in csv)
#3) Look for another test