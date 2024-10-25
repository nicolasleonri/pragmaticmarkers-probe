import os
import torch
import numpy as np
import tqdm
import argparse
from constants import lang2vocab
from constants import lang2model
from transformers import BertModel
from transformers import BertTokenizer
import constants as c
from util.argparse import str2bool

# Argument parsing
def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("-lang", "--language", type=str, required=True)  # Language (e.g., EN, DE, FI)
  parser.add_argument("-gpu", type=str, required=True) # GPU ID for training/inference
  parser.add_argument("-multiling", "--use_multiling_enc", type=str2bool, required=True)  # Use multilingual BERT
  return parser.parse_args()

# Set up the environment
def setup_environment(args):
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # Set GPU device for CUDA
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use CUDA if available, fallback to CPU
  return device

def prepare_output_directory(use_multilingual, language):
  basedir = c.ISO_DIR 
  basedir += "/multilingual/" if use_multilingual else "/monolingual/"
  basedir += language + "/"
  os.makedirs(basedir, exist_ok=True)
  return basedir

# Load the appropriate model and tokenizer
def load_model_and_tokenizer(language, use_multilingual):
  if use_multilingual:
    model_name = lang2model["mbert"]
  else:
    model_name = lang2model[language]

  tokenizer = BertTokenizer.from_pretrained(model_name)
  model = BertModel.from_pretrained(model_name, output_hidden_states=True)

  return tokenizer, model, model_name

def load_vocabulary(language):
  with open(lang2vocab[args.language], "r") as f:
    vocabulary = [line.strip() for line in f.readlines()]
  return vocabulary

# Function to encode and extract embeddings for each vocabulary entry
def extract_embeddings(vocabulary, model, tokenizer, device, num_layers=13):
  layer2embs = {layer: [] for layer in range(num_layers)}
  vocab_entries = []

  model.eval()  # Set model to evaluation mode
  with torch.no_grad():
    for entry in tqdm.tqdm(vocabulary, total=len(vocabulary)):
      encoded = [tokenizer.encode(entry, add_special_tokens=False)]  # Encode vocabulary term
      encoded_tensor = torch.tensor(encoded, dtype=torch.long).to(device)

      if encoded_tensor.nelement() > 0:  # Check if the tensor contains elements
        all_layers = model(encoded_tensor)[-1]  # Get all layers
        vocab_entries.append(entry + "\n")  # Add vocabulary entry

        assert len(all_layers) == num_layers  # Ensure number of layers matches
        for layer_idx, embedding_layer in enumerate(all_layers):
          embedding = torch.squeeze(torch.mean(embedding_layer, dim=1)).cpu().numpy()
          layer2embs[layer_idx].append(embedding)  # Store the embedding for this layer

  return layer2embs, vocab_entries

# Function to save the embeddings and vocabulary to disk
def save_embeddings_and_vocab(layer2embs, vocab_entries, basedir, language, num_layers=13):
  for layer_idx in range(num_layers):
      np.save(f"{basedir}{language}_{layer_idx}.npy", np.array(layer2embs[layer_idx], dtype=np.float64))

  with open(f"{basedir}{language}.vocab", "w") as vocab_file:
      vocab_file.writelines(vocab_entries)

# Main function
if __name__ == "__main__":
  args = parse_arguments()
  device = setup_environment(args)

  basedir = prepare_output_directory(args.use_multiling_enc, args.language)
  
  tokenizer, model, model_name = load_model_and_tokenizer(args.language, args.use_multiling_enc)
  model.to(device, non_blocking=True)
  model.eval()
  print('Model being used: ', model_name)

  vocabulary = load_vocabulary(args.language)

  if args.language == "en2":
    # Bert large has more layers
    layer2embs, vocab_entries = extract_embeddings(vocabulary, model, tokenizer, device, 25)
  else:
    layer2embs, vocab_entries = extract_embeddings(vocabulary, model, tokenizer, device)

  print("Saving embeddings")
  save_embeddings_and_vocab(layer2embs, vocab_entries, basedir, args.language)

  print("DONE!")
