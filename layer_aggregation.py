import numpy as np
import os
import constants as c

for first in [False, True]:
  for lang_type in ["monolingual", "multilingual"]: #"multilingual", 
    for lang in ["en", "en2"]: # "de", "it", "fi", "ru"
      if lang == "en2" and lang_type == "multilingual":
        # Handles case
        continue

      path = f"{c.ISO_DIR}/{lang_type}/{lang}/"
      all_layers = []
      
      if lang == "en2":
        num_layers = 24
      else:
        num_layers = 13

      for i in range(num_layers):
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
        if lang == "en2":
          aggregations.append(np.mean(all_layers[:15], axis=0))
          aggregations.append(np.mean(all_layers[:17], axis=0))
          aggregations.append(np.mean(all_layers[:19], axis=0))
          aggregations.append(np.mean(all_layers[:21], axis=0))
          aggregations.append(np.mean(all_layers[:24], axis=0))
      else:
        aggregations.append(np.mean(all_layers[-3:], axis=0))
        aggregations.append(np.mean(all_layers[-5:], axis=0))
        aggregations.append(np.mean(all_layers[-7:], axis=0))
        aggregations.append(np.mean(all_layers[-9:], axis=0))
        aggregations.append(np.mean(all_layers[-11:], axis=0))
        aggregations.append(np.mean(all_layers[-13:], axis=0))
        if lang == "en2":
          aggregations.append(np.mean(all_layers[-15], axis=0))
          aggregations.append(np.mean(all_layers[-17], axis=0))
          aggregations.append(np.mean(all_layers[-19], axis=0))
          aggregations.append(np.mean(all_layers[-21], axis=0))
          aggregations.append(np.mean(all_layers[-24], axis=0))

      if first:
        subdir = path + "aggr_first_k/"
      else:
        subdir = path + "aggr_last_k/"

      os.makedirs(subdir, exist_ok=True)
      for i, aggr in enumerate(aggregations):
        with open(subdir + lang + "_aggr_%s.npy" % str(i), "wb") as f:
          np.save(f, aggr)

      print("done with %s %s %s" % (lang, lang_type, first))

print("DONE")
