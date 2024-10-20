import numpy as np
import os
import constants as c

for first in [False, True]:
  for lang_type in ["monolingual"]: #"multilingual", 
    for lang in ["en", ]: # "de", "it", "fi", "ru"

      path = f"{c.AOC_DIR}/{lang_type}/{lang}/"
      all_layers = []
      mBERT_num_layers = 13

      for i in range(mBERT_num_layers):
        with open(path + lang + "_%s" % str(i) + ".npy", "rb") as f:
          all_layers.append(np.load(f))
        print("loaded layer %s" % str(i))

      aggregations = []
      if first:
        # first two layers (+emb layer)
        aggregations.append(np.mean(all_layers[:3], axis=0))
        # first four layers (+emb layer)
        aggregations.append(np.mean(all_layers[:5], axis=0))
        # first six layers (+emb layer)
        aggregations.append(np.mean(all_layers[:7], axis=0))
        # first eight layers (+emb layer)
        aggregations.append(np.mean(all_layers[:9], axis=0))
        # first ten layers (+emb layer)
        aggregations.append(np.mean(all_layers[:11], axis=0))
        # first twelve layers (+emb layer)
        aggregations.append(np.mean(all_layers[:13], axis=0))
      else:
        # first two layers (+emb layer)
        aggregations.append(np.mean(all_layers[-3:], axis=0))
        # first four layers (+emb layer)
        aggregations.append(np.mean(all_layers[-5:], axis=0))
        # first six layers (+emb layer)
        aggregations.append(np.mean(all_layers[-7:], axis=0))
        # first eight layers (+emb layer)
        aggregations.append(np.mean(all_layers[-9:], axis=0))
        # first ten layers (+emb layer)
        aggregations.append(np.mean(all_layers[-11:], axis=0))
        # first twelve layers (+emb layer)
        aggregations.append(np.mean(all_layers[-13:], axis=0))

      if first:
        subdir = path + "aggr_first_k/"
      else:
        subdir = path + "aggr_last_k/"

      os.makedirs(subdir, exist_ok=True)
      for i, aggr in enumerate(aggregations):
        with open(subdir + lang + "_aggr_%s.npy" % str(i), "wb") as f:
          np.save(f, aggr)

print("DONE")
