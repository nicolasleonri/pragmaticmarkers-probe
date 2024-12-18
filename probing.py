from sklearn.metrics.pairwise import cosine_similarity
from constants import lang2model
from transformers import BertTokenizer
import constants as c
import numpy as np
import argparse
from util.argparse import str2bool
from util.preprocessing import * 
from util.probing import *
import multiprocessing as mp
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import time


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

# Function to get embeddings for each word
def get_word_embedding(word, layer_embeddings, vocab2id):
    if word in vocab2id:
        token_id = vocab2id[word] # Get the token id from the vocabulary
        embeddings = [layer[token_id] for layer in layer_embeddings]
        return np.mean(embeddings, axis=0)
    else:
        return None
    
# Parallelized function to get the embedding for a sentence
def get_sentence_embedding_parallel(sentence, layer_embeddings, vocab2id, type_tokens, tokenizer):
    sentence_embeddings = []

    sentence = normalize(sentence)

    # Determine the tokens
    if type_tokens == "NoSpec":
        words = sentence.split()
    elif type_tokens == "All":
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        words = [cls_token] + sentence.split() + [sep_token]
    elif type_tokens == "WithCLS":
        cls_token = tokenizer.cls_token
        words = [cls_token] + sentence.split()
    else:
        raise ImportError("No such tokenization")
    
    # Use ThreadPoolExecutor to process each word in parallel
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(get_word_embedding, word, layer_embeddings, vocab2id)
            for word in words
        ]

        # Collect results and filter out None values
        sentence_embeddings = [f.result() for f in futures if f.result() is not None]

    # Return the average embedding of the sentence
    if sentence_embeddings:
        return np.mean(sentence_embeddings, axis=0)
    else:
        return None

# Function to get the embedding for a sentence
def get_sentence_embedding(sentence, layer_embeddings, vocab2id, type_tokens, tokenizer):
    sentence_embeddings = []

    sentence = normalize(sentence)

    if type_tokens == "NoSpec":
        words = sentence.split()
    elif type_tokens == "All":
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        words = [cls_token] + sentence.split() + [sep_token]
    elif type_tokens == "WithCLS":
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

def load_single_layer_embedding(layer, encoder, lang_type, lang):
    emb_file_path = f"{encoder}/{lang_type}/{lang}/{lang}_{layer}.npy"
    print("Embedding from", emb_file_path)
    return np.load(emb_file_path)

def load_layer_embeddings_parallel(encoder, lang_type, lang, num_layers=13,  type_layers="Average"):
    lang_type = "multilingual" if lang_type else "monolingual"
    layer_embeddings = []

    print('######################EMBEDDING######################')
        
    if type_layers=="Average":
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Create a list of future tasks for loading each layer embedding in parallel
            futures = [
                executor.submit(load_single_layer_embedding, layer, encoder, lang_type, lang)
                for layer in range(int(num_layers))
            ]
            # Collect all loaded embeddings
            layer_embeddings = [future.result() for future in concurrent.futures.as_completed(futures)]
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

# Load embeddings from saved npy-documents
def load_layer_embeddings(encoder, lang_type, lang, num_layers=13, type_layers="Average"):
    lang_type = "multilingual" if lang_type else "monolingual"
    layer_embeddings = []

    print('######################EMBEDDING######################')

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

# Helper function to calculate similarity for a single marker
def calculate_marker_similarity(marker, sentence_embedding, layer_embeddings, vocab2id, type_of_tokenization, tokenizer):
    marker_embedding = get_sentence_embedding(marker, layer_embeddings, vocab2id, type_of_tokenization, tokenizer)
    if marker_embedding is not None:
        similarity = cosine_similarity(sentence_embedding.reshape(1, -1), marker_embedding.reshape(1, -1))[0][0]
        return marker, similarity
    else:
        print(f"Marker '{marker}' could not be embedded.")
        return marker, None
    
# Parallelized function to find the most appropriate discourse marker
def find_best_disco_marker_parallel(sentence_embedding, markers, layer_embeddings, vocab2id, type_of_tokenization, tokenizer):
    similarities = {}

    # Use ThreadPoolExecutor to process each marker in parallel
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(calculate_marker_similarity, marker, sentence_embedding, layer_embeddings, vocab2id, type_of_tokenization, tokenizer): marker
            for marker in markers
        }

        for future in futures:
            marker, similarity = future.result()
            if similarity is not None:
                similarities[marker] = similarity

    # Select the marker with the highest similarity score
    if similarities:
        best_marker = max(similarities, key=similarities.get)
        return best_marker
    else:
        return None

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

def probing_parallel(path, exp_data, csv_filename):
    df = read_csv(str(path))
    tokenizer = BertTokenizer.from_pretrained(lang2model[args.language])

    correct_predictions = 0
    total_predictions = 0
    not_embedded_sentences = 0

    sentence_pairs = df[['Before', 'After']].values
    actual_markers = df['PM'].values
    unique_target_variables = df['PM'].unique()  # Unique discourse markers            

    predictions = []

    print('######################PROBING######################')

    # Helper function to process each sentence pair
    def process_sentence_pair(x):
        s1_embedding = get_sentence_embedding_parallel(str(sentence_pairs[x, 0]), layer_embeddings, vocab2id, args.type_of_tokenization, tokenizer)
        s2_embedding = get_sentence_embedding_parallel(str(sentence_pairs[x, 1]), layer_embeddings, vocab2id, args.type_of_tokenization, tokenizer)
        print(f"Embedding for Sentence Pair {x} computed.")
        if s1_embedding is not None and s2_embedding is not None:
            combined_embedding = np.mean(np.vstack([s1_embedding, s2_embedding]), axis=0)
            best_marker = find_best_disco_marker_parallel(combined_embedding, unique_target_variables, layer_embeddings, vocab2id, args.type_of_tokenization, tokenizer)
            print(f"Best marker for Sentence Pair {x} computed.")
            if best_marker == actual_markers[x]:
                return best_marker, True
            else:
                return best_marker, False
        else:
            print(f"Embedding for Sentence Pair {x} could not be computed.")
            return "NA", None

    # Using ThreadPoolExecutor for parallel processing of each sentence pair
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_sentence_pair, range(len(sentence_pairs))))

    # Unpacking results and calculating accuracy
    for x, (best_marker, correct) in enumerate(results):
        predictions.append(best_marker)
        if correct is not None:
            if correct:
                correct_predictions += 1
            total_predictions += 1
        else:
            not_embedded_sentences += 1

    # Calculate zero-one loss and accuracy
    zero_one_loss = total_predictions - correct_predictions 
    accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0

    # Add predictions to dataframe and save to CSV
    df[exp_data] = predictions
    save_predictions(os.path.join(str(c.RESULTS), str(csv_filename)), df, exp_data)

    # Print final results
    print('######################RESULTS######################')
    print(f"Total Predictions: {total_predictions}, Correct Predictions: {correct_predictions}, Accuracy: {accuracy:.2f}%")
    print(f"Zero-One Loss: {zero_one_loss}, Not embedded sentences: {not_embedded_sentences}")

    return total_predictions, correct_predictions, not_embedded_sentences, accuracy


def probing(path, exp_data, csv_filename):
    df = read_csv(str(path))

    correct_predictions = 0
    total_predictions = 0
    not_embedded_sentences = 0

    sentence_pairs = df[['Before', 'After']].values
    actual_markers = df['PM'].values
    unique_target_variables = df['PM'].unique()  # Unique discourse markers            

    predictions = []

    print('######################PROBING######################')

    for x in range(0, len(sentence_pairs)):
        s1_embedding = get_sentence_embedding_parallel(str(sentence_pairs[x,0]), layer_embeddings, vocab2id, args.type_of_tokenization)
        s2_embedding = get_sentence_embedding_parallel(str(sentence_pairs[x,1]), layer_embeddings, vocab2id, args.type_of_tokenization)
        
        if s1_embedding is not None and s2_embedding is not None:
            combined_embedding = np.mean(np.vstack([s1_embedding, s2_embedding]), axis=0)
        else:
            print(f"Embedding for Sentence Pair {x} could not be computed.")
            not_embedded_sentences += 1
            predictions.append("NA")
            continue

        best_marker = find_best_disco_marker_parallel(combined_embedding, unique_target_variables, layer_embeddings, vocab2id, args.type_of_tokenization)
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
    save_predictions(os.path.join(str(c.RESULTS), str(csv_filename)), df, exp_data)

    print('######################RESULTS######################')

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

    # if args.type_of_layer and args.index_of_layer:
    layer_embeddings = load_layer_embeddings_parallel(encoder, args.use_multiling_enc, args.language, args.index_of_layer, args.type_of_layer, )
    exp_data = f"{args.context_encoder}/{args.language}/{lang_type}/{args.type_of_tokenization}/{args.index_of_layer}/{args.type_of_layer}"
    pool = mp.Pool(mp.cpu_count())

    # elif args.index_of_layer:
    #     layer_embeddings = load_layer_embeddings(encoder, args.use_multiling_enc, args.language, args.index_of_layer)
    #     exp_data = f"{args.context_encoder}/{args.language}/{lang_type}/{args.type_of_tokenization}/13/{args.index_of_layer}"
    # else:
    #     layer_embeddings = load_layer_embeddings(encoder, args.use_multiling_enc, args.language)
    #     exp_data = f"{args.context_encoder}/{args.language}/{lang_type}/{args.type_of_tokenization}/13/Total"

    vocab, vocab2id = load_vocab(encoder, args.use_multiling_enc, args.language)
    
    tic = time.time()
    total_predictions, correct_predictions, not_embedded_sentences, accuracy = probing_parallel("./data/frazer_categorization", exp_data, "frazer_predictions.csv")
    toc = time.time()
    needed =  toc - tic
    print('######################TIME######################')
    print(f"Time needed for Frazer Classification: {needed} seconds")
    save_multiclassifier_as_csv(os.path.join(str(c.RESULTS), 'frazer_results.csv'), exp_data, total_predictions, correct_predictions, not_embedded_sentences, accuracy)

    tic = time.time()
    total_predictions, correct_predictions, not_embedded_sentences, accuracy = probing_parallel("./data/discourse_connective", exp_data, "discourse_connective_predictions.csv")
    toc = time.time()
    needed =  toc - tic
    print('######################TIME######################')
    print(f"Time needed for Discourse Connective Classification: {needed} seconds")

    save_multiclassifier_as_csv(os.path.join(str(c.RESULTS), 'discourse_connective_results.csv'), exp_data, total_predictions, correct_predictions, not_embedded_sentences, accuracy)
