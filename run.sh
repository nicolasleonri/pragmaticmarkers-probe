#!/bin/bash

source "/data/pragmaticmarkers-probe/venv/bin/activate"

#python3 -u aoc.py -lang en -gpu 0 -cs 50 -multiling True > logs/aoc_en_multi.out
#python3 -u aoc.py -lang en -gpu 0 -cs 50 -multiling False > logs/aoc_en_mono.out

#python3 -u iso.py -lang en -gpu 0 -multiling True > logs/iso_en_multi.out
#python3 -u iso.py -lang en -gpu 0 -multiling False > logs/iso_en_mono.out

#python3 -u aoc_iso_integration.py > logs/aoc_iso_integration.out
#python3 -u layer_aggregation.py > logs/layer_integration.out

languages=("en")
use_multiling_enc=("True" "False")
context_encoders=("AOC" "ISO")
tokenizations=("NoSpec" "All" "WithCLS")
type_of_layers=("Average" "First" "Last" "Unique")  # Optional argument
index_of_layers_all=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12")  # Optional argument
index_of_layers_short=("0" "1" "2" "3" "4" "5")  # For "First" and "Last" layers

# Loop through all possible combinations of arguments
for lang in "${languages[@]}"; do
  for multiling in "${use_multiling_enc[@]}"; do
    for encoder in "${context_encoders[@]}"; do
      for tokenization in "${tokenizations[@]}"; do
        for layer in "${type_of_layers[@]}"; do
        
        # # Conditional check for "First" or "Last" layers
          if [[ "$layer" == "First" || "$layer" == "Last" ]]; then
            index_of_layers=("${index_of_layers_short[@]}")
          else
            index_of_layers=("${index_of_layers_all[@]}")
          fi

          for ilayer in "${index_of_layers[@]}"; do
            
            # Run the Python script with the current set of arguments
            python3 probing.py \
              -lang "$lang" \
              -multiling "$multiling" \
              -encoder "$encoder" \
              -tokenization "$tokenization" \
              -tlayer "$layer" \
              -ilayer "$ilayer"

            echo "Ran with lang=$lang, multiling=$multiling, encoder=$encoder, tokenization=$tokenization, tlayer=$layer, ilayer=$ilayer" >> logs/experiment_log.out

          done
        done
      done
    done
  done
done
