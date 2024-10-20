import torch
from constants import lang2model

def encode_BERT(txt, tokenizer):
  tokens = []
  reconstruction_mask = []
  word_pieces = []

  for word_piece in tokenizer.tokenize(txt):
    is_first_wordpiece = not word_piece.startswith("##")
    if is_first_wordpiece:
      if len(word_pieces) == 0:
        word_pieces.append(word_piece)
      else:
        if len(tokens) + len(word_pieces) < 512:
          tokens.extend(word_pieces)
          reconstruction_mask.append(len(word_pieces))
          word_pieces = [word_piece]
        else:
          break
    else:
      word_pieces.append(word_piece)

  if word_pieces and len(tokens) + len(word_pieces) < 512:
    tokens.extend(word_pieces)
    reconstruction_mask.append(len(word_pieces))

  sent = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
  sent_len = torch.tensor(len(sent))
  return sent, sent_len, reconstruction_mask
