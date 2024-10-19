lang2model = {
  'en': 'bert-base-uncased',
  # 'de': 'dbmdz/bert-base-german-uncased',
  # 'fi': 'TurkuNLP/bert-base-finnish-uncased-v1',
  # 'ru': 'DeepPavlov/rubert-base-cased',
  # 'tr': 'dbmdz/bert-base-turkish-uncased',
  'mbert': 'bert-base-multilingual-uncased'
}

# directory where language vocabularies are placed
vocab_basedir = "./vocab/"
lang2vocab = {
  'en': vocab_basedir + "wiki.en.100k.vocab",
  # 'de': vocab_basedir + "wiki.de.100k.vocab",
  # 'fi': vocab_basedir + "wiki.fi.100k.vocab",
  # 'ru': vocab_basedir + "wiki.ru.100k.vocab",
  # 'tr': vocab_basedir + "wiki.tr.100k.vocab"
}

# enter directory where to place ISO/AOC/combined embeddings
ISO_DIR = "./iso"
AOC_DIR = "./aoc"
AOC_ISO_DIR = "./iso-aoc"

# place where corpora are placed
AOC_DOC_DIR = "./corpora/"

# place where to save cka plots
CKA_PLOT_DIR = ""

# directory with bli dictionaries
BLI_DIR = ""
