#!/bin/bash

directory="$PWD"/"logs"/"experiments"

if [ ! -d $directory ]; then
echo "Directory doesn't exist ("$directory")"
mkdir -p $directory
fi

directory="$PWD"/"venv"/"bin"/"activate"
source $directory

# python3 -u iso.py -lang en2 -gpu 0 -multiling False > logs/iso_en2_mono.out
# python3 -u aoc_iso_integration.py > logs/aoc_iso_integration.out
# python3 -u layer_aggregation.py > logs/layer_integration.out

languages=("en2")
use_multiling_enc=("False")
context_encoders=("ISO")
tokenizations=("NoSpec" "All" "WithCLS")
type_of_layers=("Average" "First" "Last" "Unique") 
index_of_layers_all_en=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12")
index_of_layers_all_en2=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24")
index_of_layers_short_en=("0" "1" "2" "3" "4" "5")
index_of_layers_short_en2=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10")

# Loop through all possible combinations of arguments
for lang in "${languages[@]}"; do
  for multiling in "${use_multiling_enc[@]}"; do
    if [[ "$lang" == "en2" && "$multiling" == "True" ]]; then
    continue
    fi
    for encoder in "${context_encoders[@]}"; do
      for tokenization in "${tokenizations[@]}"; do
        for layer in "${type_of_layers[@]}"; do
        
          if [[ "$layer" == "First" || "$layer" == "Last" ]] && [[ "$lang" == "en2" ]]; then
            index_of_layers=("${index_of_layers_short_en2[@]}")
          elif [[ "$layer" == "First" || "$layer" == "Last" ]] && [[ "$lang" == "en" ]]; then
            index_of_layers=("${index_of_layers_short_en[@]}")
          elif [[ "$lang" == "en2" ]]; then
            index_of_layers=("${index_of_layers_all_en2[@]}")
          elif [[ "$lang" == "en" ]]; then
            index_of_layers=("${index_of_layers_all_en[@]}")
          fi

          for ilayer in "${index_of_layers[@]}"; do
            
            echo "Running with lang=$lang, multiling=$multiling, encoder=$encoder, tokenization=$tokenization, tlayer=$layer, ilayer=$ilayer" 

            python3 probing.py \
              -lang "$lang" \
              -multiling "$multiling" \
              -encoder "$encoder" \
              -tokenization "$tokenization" \
              -tlayer "$layer" \
              -ilayer "$ilayer" >> logs/experiments/"$lang"_"$multiling"_"$encoder"_"$tokenization"_"$layer"_.out

          done
        done
      done
    done
  done
done
