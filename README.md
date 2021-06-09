# LyricGen
## Authors
Junhwi Kim (junhwi.kim.20@dartmouth.edu)  
Perry Zhang (perry.h.zhang.24@dartmouth.edu)  
  
Created on Tuesday, June 8th, 2021.

## Description
Our final project fine-tunes a pretrained GPT-2 model supplied by huggingface, and generates realistic song lyrics from a dataset scraped from the Genius Lyrics' website.

## Installation
Download and install huggingface transformers **inside** the project root:  
```console
git clone https://github.com/huggingface/transformers.git  
cd transformers  
pip install .  
cd transformers/examples/pytorch/language_modelling  
pip install -r requirements.txt
```  

## Usage
To train, at the project root:
```console
python transformers/examples/pytorch/language_modelling/run_clm.py \  
--model_name_or_path distilgpt2 \  
--train_file data/all_lyrics_train.txt \
--validation_file data/all_lyrics_val.txt \
--do_train \
--output_dir outputs/ \
--block_size 512
```

