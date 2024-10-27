import os
import numpy as np
from copy import copy
import constants as c
import random

aoc_dir = c.AOC_DIR
iso_dir = c.ISO_DIR
combined_dir = c.AOC_ISO_DIR
os.makedirs(combined_dir, exist_ok=True)

for encoder in ["monolingual", "multilingual"]:
  for lang in ["en", "en2"]: # "de", "fi", "ru", "tr"
    if lang == "en2" and encoder == "multilingual":
      # Handles case
      continue
    iso_vocab_file = iso_dir + "/" + encoder + "/" + lang + "/" + lang + ".vocab"
    aoc_vocab_file = aoc_dir + "/" + encoder + "/" + lang + "/" + lang + ".vocab"
    combined_vocab_path = combined_dir + "/" + encoder + "/" + lang + "/"

    with open(iso_vocab_file, "r") as f:
      iso_vocab = [line.strip() for line in f]
    iso_vocab2id = {term: i for i, term in enumerate(iso_vocab)}
    with open(aoc_vocab_file, "r") as f:
      aoc_vocab = [line.strip() for line in f]
    aoc_vocab2id = {term: i for i, term in enumerate(aoc_vocab)}

    aoc_layer_embs = []
    iso_layer_embs = []
    if lang == "en2":
      num_layers = 24
    else:
      num_layers = 13
    for layer in range(num_layers):
      aoc_layer_emb_file = aoc_dir + "/%s/%s/%s_%s.npy" % (encoder, lang, lang, str(layer))
      iso_layer_emb_file = iso_dir + "/%s/%s/%s_%s.npy" % (encoder, lang, lang, str(layer))

      with open(aoc_layer_emb_file, "rb") as f:
        aoc_layer_embs.append(np.load(f))
      assert aoc_layer_embs[-1].shape[0] == len(aoc_vocab)
      with open(iso_layer_emb_file, "rb") as f:
        iso_layer_embs.append(np.load(f))
      assert iso_layer_embs[-1].shape[0] == len(iso_vocab)
      print("loaded layer %s" % str(layer))

    combo_vocab = copy(aoc_vocab)
    combo_layer_embs = copy(aoc_layer_embs)

    difference_vocab = set(iso_vocab).difference(combo_vocab)
    missing_layer_embs = [[] for _ in range(num_layers)]
    for missing_term in difference_vocab:
      combo_vocab.append(missing_term)
      for i, layer in enumerate(iso_layer_embs):
        missing_layer_emb = layer[iso_vocab2id[missing_term]]
        missing_layer_embs[i].append(missing_layer_emb)
    print("Missing ISO embeddings collected")

    for i in range(num_layers):
      aoc_embs = combo_layer_embs[i]
      added_iso_embs = np.stack(missing_layer_embs[i])
      combo_layer_embs[i] = np.concatenate([aoc_embs, added_iso_embs])
    print("Missing ISO embedding integrated")

    # verify first half (aoc embs)
    np.testing.assert_array_equal(combo_layer_embs[0][:len(aoc_vocab)], aoc_layer_embs[0])
    # verify second half (iso embs)
    np.testing.assert_array_equal(combo_layer_embs[0][-len(difference_vocab):], missing_layer_embs[0])
    # verify random aoc embedding
    random_aoc_term = random.choice(aoc_vocab)
    np.testing.assert_equal(combo_layer_embs[2][aoc_vocab2id[random_aoc_term]], aoc_layer_embs[2][aoc_vocab2id[random_aoc_term]])
    # verify random iso embedding
    random_missing_term = list(difference_vocab)[3]
    np.testing.assert_array_equal(combo_layer_embs[6][combo_vocab.index(random_missing_term)], iso_layer_embs[6][iso_vocab2id[random_missing_term]])
    print("Verification tests passsed")

    os.makedirs(combined_vocab_path, exist_ok=True)
    with open(combined_vocab_path + lang + ".vocab", "w") as f:
      f.writelines([term + "\n" for term in combo_vocab])
    for i in range(num_layers):
      combo_layer_emb_path = combined_dir + "/%s/%s/" % (encoder, lang)
      os.makedirs(combo_layer_emb_path, exist_ok=True)
      with open(combo_layer_emb_path + "%s_%s.npy" % (lang, str(i)), "wb") as f:
        np.save(f, combo_layer_embs[i])

    print("done with %s (%s)" % (lang, encoder))
