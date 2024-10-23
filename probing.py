from sklearn.metrics.pairwise import cosine_similarity
from constants import lang2model
from transformers import BertTokenizer
import constants as c
import numpy as np
import argparse
from util.argparse import str2bool
from util.preprocessing import * 
from util.probing import *

# Argument parsing
def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("-lang", "--language", type=str, required=True)  # Language (e.g., EN, DE, FI)
  parser.add_argument("-multiling", "--use_multiling_enc", type=str2bool, required=True)  # Use multilingual BERT
  parser.add_argument("-encoder", "--context_encoder", type=str, required=True)  # Select which encoder to use
  parser.add_argument("-tokenization", "--type_of_tokenization", type=str, required=True)  # Select which encoder to use
  parser.add_argument("-tlayer", "--type_of_layer", type=str, required=False)  # Select how to average layers
  parser.add_argument("-ilayer", "--index_of_layer", type=str, required=False)  # Select which indeces to average
  return parser.parse_args()

# Function to get the embedding for a sentence
def get_sentence_embedding(sentence, layer_embeddings, vocab2id, type_tokens):
    sentence_embeddings = []

    sentence = normalize(sentence)

    if type_tokens == "NoSpec":
        words = sentence.split()
    elif type_tokens == "All":
        tokenizer = BertTokenizer.from_pretrained(lang2model[args.language])
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        words = [cls_token] + sentence.split() + [sep_token]
    elif type_tokens == "WithCLS":
        tokenizer = BertTokenizer.from_pretrained(lang2model[args.language])
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        words = [cls_token] + sentence.split()
    else:
        raise ImportError("No such tokenization")

    for word in words:
        if word in vocab2id:
            token_id = vocab2id[word]  # Get the token id from the vocabulary
            embeddings = [layer[token_id] for layer in layer_embeddings]
            avg_embedding = np.mean(embeddings, axis=0)
            sentence_embeddings.append(avg_embedding)
        else:
            # print(f"Word '{word}' not found in vocabulary.")
            continue

    if sentence_embeddings:
        return np.mean(sentence_embeddings, axis=0)
    else:
        return None 

# Load embeddings from saved npy-documents
def load_layer_embeddings(encoder, lang_type, lang, num_layers=13, type_layers="Average"):
    lang_type = "multilingual" if lang_type else "monolingual"
    layer_embeddings = []

    print('##################################################')

    if type_layers=="Average":
        for layer in range(int(num_layers)):
            emb_file_path = f"{encoder}/{lang_type}/{lang}/{lang}_{layer}.npy"
            print("Embedding from", emb_file_path)
            layer_embeddings.append(np.load(emb_file_path))
    elif type_layers=="First":
        emb_file_path = f"{encoder}/{lang_type}/{lang}/aggr_first_k/{lang}_aggr_{num_layers}.npy"
        print("Embedding from", emb_file_path)
        layer_embeddings.append(np.load(emb_file_path))
    elif type_layers=="Last":
        emb_file_path = f"{encoder}/{lang_type}/{lang}/aggr_last_k/{lang}_aggr_{num_layers}.npy"
        print("Embedding from", emb_file_path)
        layer_embeddings.append(np.load(emb_file_path))
    elif type_layers=="Unique":
        emb_file_path = f"{encoder}/{lang_type}/{lang}/{lang}_{num_layers}.npy"
        print("Embedding from", emb_file_path)
        layer_embeddings.append(np.load(emb_file_path))
    else:
        raise ImportError("No such file")
    
    return layer_embeddings

# Function to find the most appropriate discourse marker
def find_best_disco_marker(sentence_embedding, markers, layer_embeddings, vocab2id):
    similarities = {}

    for marker in markers:
        marker_embedding = get_sentence_embedding(marker, layer_embeddings, vocab2id, args.type_of_tokenization)  # Get marker embedding
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

def dissent_task(path, vocab2id):
    sentence_pairs, unique_target_variables = read_tsv_files(str(path))

    correct_predictions = 0
    total_predictions = 0
    not_embedded_sentences = 0

    for idx, (s1, s2, actual_marker) in enumerate(sentence_pairs):
        s1_embedding = get_sentence_embedding(s1, layer_embeddings, vocab2id, args.type_of_tokenization)
        s2_embedding = get_sentence_embedding(s2, layer_embeddings, vocab2id, args.type_of_tokenization)

        if s1_embedding is not None and s2_embedding is not None:
            combined_embedding = np.mean(np.vstack([s1_embedding, s2_embedding]), axis=0)

        best_marker = find_best_disco_marker(combined_embedding, unique_target_variables, layer_embeddings, vocab2id)

        print(f"Sentence Pair Index {idx}: Best Marker - {best_marker}")

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

    return total_predictions, correct_predictions, not_embedded_sentences, accuracy

def probing(path, exp_data):
    df = read_csv(str(path))

    correct_predictions = 0
    total_predictions = 0
    not_embedded_sentences = 0

    sentence_pairs = df[['Before', 'After']].values
    actual_markers = df['PM'].values
    unique_target_variables = df['PM'].unique()  # Unique discourse markers            

    predictions = []

    for x in range(0, len(sentence_pairs)):
        s1_embedding = get_sentence_embedding(str(sentence_pairs[x,0]), layer_embeddings, vocab2id, args.type_of_tokenization)
        s2_embedding = get_sentence_embedding(str(sentence_pairs[x,1]), layer_embeddings, vocab2id, args.type_of_tokenization)
        
        if s1_embedding is not None and s2_embedding is not None:
            combined_embedding = np.mean(np.vstack([s1_embedding, s2_embedding]), axis=0)
        else:
            print(f"Embedding for Sentence Pair {x} could not be computed.")
            not_embedded_sentences += 1
            predictions.append("NA")
            continue

        best_marker = find_best_disco_marker(combined_embedding, unique_target_variables, layer_embeddings, vocab2id)
        predictions.append(str(best_marker))
        sentence = " [MASK] ".join(list(sentence_pairs[x]))
        print(f"Sentence: {sentence}")
        print(f"Best marker: {best_marker}")

        if best_marker == actual_markers[x]:
            correct_predictions += 1
        
        total_predictions += 1        

    zero_one_loss = total_predictions - correct_predictions 
    accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0  # Accuracy calculation

    df[exp_data] = predictions
    save_predictions(os.path.join(str(c.RESULTS), 'predictions.csv'), df, exp_data)

    print('##################################################')

    print(f"Total Predictions: {total_predictions}, Correct Predictions: {correct_predictions}, Accuracy: {accuracy:.2f}%")
    print(f"Zero-One Loss: {zero_one_loss}, Not embedded sentences: {not_embedded_sentences}")

    return total_predictions, correct_predictions, not_embedded_sentences, accuracy

# Main function
if __name__ == "__main__":
    args = parse_arguments()

    if args.context_encoder == "AOC":
        encoder = c.AOC_DIR
    elif args.context_encoder == "ISO":
        encoder = c.ISO_DIR
    else:
        raise ImportError("Encoder is missing")
    
    if args.use_multiling_enc:
        lang_type = "multi"
    else:
        lang_type = "mono"

    if args.type_of_layer and args.index_of_layer:
        layer_embeddings = load_layer_embeddings(encoder, args.use_multiling_enc, args.language, args.index_of_layer, args.type_of_layer, )
        exp_data = f"{args.context_encoder}/{args.language}/{lang_type}/{args.type_of_tokenization}/{args.index_of_layer}/{args.type_of_layer}"
    elif args.index_of_layer:
        layer_embeddings = load_layer_embeddings(encoder, args.use_multiling_enc, args.language, args.index_of_layer)
        exp_data = f"{args.context_encoder}/{args.language}/{lang_type}/{args.type_of_tokenization}/13/{args.index_of_layer}"
    else:
        layer_embeddings = load_layer_embeddings(encoder, args.use_multiling_enc, args.language)
        exp_data = f"{args.context_encoder}/{args.language}/{lang_type}/{args.type_of_tokenization}/13/Total"

    vocab, vocab2id = load_vocab(encoder, args.use_multiling_enc, args.language)

    total_predictions, correct_predictions, not_embedded_sentences, accuracy = probing("./data/frazer_categorization", exp_data)

    save_multiclassifier_as_csv(os.path.join(str(c.RESULTS), 'results.csv'), exp_data, total_predictions, correct_predictions, not_embedded_sentences, accuracy)