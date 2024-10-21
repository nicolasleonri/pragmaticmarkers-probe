import os
import tqdm
import torch
import argparse
import numpy as np
from random import shuffle
from itertools import chain
from collections import Counter

from util.encoding import *  
from constants import lang2vocab, lang2model # Maps languages to vocab/model files
from transformers import BertModel, BertTokenizer # Helper function to convert string arguments to boolean values
from util.argparse import str2bool
import constants as c  # Custom constants, such as directories and models

# Argument parsing
def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("-lang", "--language", type=str, required=True)  # Language (e.g., EN, DE, FI)
  parser.add_argument("-gpu", type=str, required=True) # GPU ID for training/inference
  parser.add_argument("-cs", "--context_size", type=int, required=True)  # Maximum context size for terms
  parser.add_argument("-multiling", "--use_multiling_enc", type=str2bool, required=True)  # Use multilingual BERT
  return parser.parse_args()

# Set up the environment
def setup_environment(args):
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # Set GPU device for CUDA
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use CUDA if available, fallback to CPU
  return device

# Load vocabulary and corpus
def load_data(language):
  with open(lang2vocab[language], "r") as f:
    vocab_set = set([line.strip() for line in f.readlines()])

  with open(c.AOC_DOC_DIR + f"{language}-low-1m.txt", "r") as f:
    corpus = [line.strip() for line in f]

  return vocab_set, list(corpus)

# Prepare the base directory for output
def prepare_output_directory(context_size, use_multilingual, language):
  basedir = f"{c.AOC_DIR}/"
  basedir += "multilingual/" if use_multilingual else "monolingual/"
  basedir += f"{language}/"
  os.makedirs(basedir, exist_ok=True)
  return basedir

# Load the appropriate model and tokenizer
def load_model_and_tokenizer(language, use_multilingual):
  if use_multilingual:
    model_name = lang2model["mbert"]
    encode = encode_BERT
  else:
    model_name = lang2model[language]
    encode = encode_BERT

  tokenizer = BertTokenizer.from_pretrained(model_name)
  model = BertModel.from_pretrained(model_name, output_hidden_states=True)

  return tokenizer, model, model_name, encode

def create_term_index(raw_corpus, vocab_set, context_size):
  index = {}
  doc2termlist = {}
  buffer = 10
  max_elems = context_size + buffer if context_size > 0 else int(1e10)  # int(1e10) = Inf

  # Special tokens (CLS and SEP)
  cls_token = tokenizer.cls_token
  sep_token = tokenizer.sep_token

  for did, sent in tqdm.tqdm(enumerate(raw_corpus), total=len(raw_corpus)):
    sent_tokens = [cls_token] + sent.split() + [sep_token]  # Insert CLS at the beginning and SEP at the end    sent_set = set(sent_tokens)
    sent_set = set(sent_tokens)
    words_positions = []

    for word in sent_set:
      if word not in vocab_set and word not in {cls_token, sep_token}:
        continue  # skip if word is not in vocab or special tokens        continue

      position = sent_tokens.index(word)
      word_wordposition = (word, position)
      doc_wordposition = (did, position)

      if word not in index:
        words_positions.append(word_wordposition)
        index[word] = [doc_wordposition]
      elif len(index[word]) < max_elems:
        words_positions.append(word_wordposition)
        index[word].append(doc_wordposition)

    doc2termlist[did] = words_positions

  return index, doc2termlist

# Helper to process wordpiece embeddings into word embeddings
def process_wordpiece_embeddings(all_layers, mask):
  tmp = []
  for i, layer in enumerate(all_layers):
    grouped_wp_embs = torch.split(layer[0], mask)
    word_embeddings = [torch.mean(wp_embs, dim=0).detach().cpu().numpy() for wp_embs in grouped_wp_embs]
    tmp.append(word_embeddings)
  return tmp

# Perform embedding extraction for the corpus
def embed_corpus(model, tokenizer, selected_docs, doc2termlist, raw_corpus, context_size, device):
  layer2term2emblist = {}  # store incomplete embeddings
  layer2term2emb = {}  # store complete embeddings
  contextcounts = {}
  skipped_sequences = 0

  model.to(device, non_blocking=True)
  model.eval()

  print("Embedding corpus")

  for did in tqdm.tqdm(selected_docs, total=len(selected_docs)):
    # Document encoding
    doc = raw_corpus[did]
    sent, length, mask = encode(doc, tokenizer)
    sent_tensor = torch.tensor([sent]).to(device) # Convert 'sent' to a tensor and move to the correct device
    all_layers = model(sent_tensor)[-1]  # Run the model on the input tensor

    all_layers = process_wordpiece_embeddings(all_layers, mask)

    # Embedding updates
    for i, emb_seq in enumerate(all_layers):
      term2emblist = layer2term2emblist[i] if i in layer2term2emblist else {}
      term2emb = layer2term2emb[i] if i in layer2term2emb else {}
      for term, position in doc2termlist[did]:
        # skip if term is done already
        if term not in term2emb:
          emblist = term2emblist[term] if term in term2emblist else []
          if position < len(emb_seq):
            emblist.append(emb_seq[position])
            # all contexts collected -> emb done
            if len(emblist) == context_size:
              term2emb[term] = np.mean(emblist, axis=0)
              contextcounts[term] = context_size
              del term2emblist[term]
            else:
              term2emblist[term] = emblist
          else:
            skipped_sequences += 1
    
      layer2term2emblist[i] = term2emblist
      layer2term2emb[i] = term2emb

  for i in range(len(layer2term2emb)):
    term2emblist = layer2term2emblist[i] if i in layer2term2emblist else {}
    term2emb = layer2term2emb[i] if i in layer2term2emb else {}
    for term, emblist in term2emblist.items():
      assert term not in term2emb
      context_size = len(emblist)
      term2emb[term] = np.mean(emblist, axis=0) if context_size > 1 else emblist[0]
      contextcounts[term] = context_size
    layer2term2emb[i] = term2emb

  return layer2term2emb, contextcounts

def save_embeddings(layer2term2emb, basedir, vocab, language):
  for layer, term2emb in layer2term2emb.items():
    embedding_table = [term2emb[term] for term in vocab]
    with open(f"{basedir}{language}_{layer}.npy", "wb") as f:
      np.save(f, np.array(embedding_table, dtype=np.float64))

  vocab = [entry + "\n" for entry in vocab]
  with open(f"{basedir}{language}.vocab", "w") as f:
    f.writelines(vocab)

# Main function
if __name__ == "__main__":
  args = parse_arguments()
  device = setup_environment(args)

  # Load data
  vocab_set, raw_corpus = load_data(args.language)

  # Prepare output directory
  basedir = prepare_output_directory(
      args.context_size, args.use_multiling_enc, args.language)
  print(f"Base directory: {basedir}")

  # Load model and tokenizer
  tokenizer, model, model_name, encode = load_model_and_tokenizer(args.language, args.use_multiling_enc)
  print('Model being used: ', model_name)

  print("Shuffling corpus")
  shuffle(raw_corpus)
  sentences = raw_corpus

  print("Creating index")
  index, doc2termlist = create_term_index(raw_corpus, vocab_set, args.context_size)

  print("Context-count distribution:", end=" ")
  context_distr = {k: len(v) for k, v in index.items()}
  print(Counter(context_distr.values()))

  selected_docs = set([sid for sid, _ in chain(*list(index.values()))]) # keep only docs that are associated with a term
  print("Effective corpus size: %s (full size: %s)" %
        (str(len(selected_docs)), str(len(raw_corpus))))
  print("Effective vocabulary size: %s" % str(len(index)))

  layer2term2emb, contextcounts = embed_corpus(model, tokenizer, selected_docs, doc2termlist, raw_corpus, args.context_size, device)

  print("Saving embeddings")
  save_embeddings(layer2term2emb, basedir, list(layer2term2emb[0].keys()), args.language) # Save embeddings and vocab

  print("DONE!")
