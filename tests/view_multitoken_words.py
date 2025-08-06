'''
This script is used to view words in a dataset.
Specifically, it looks for words that get converted to multiple tokens by a specific model.
This can be useful for determining extra words in a definition that are not necessary.
'''
from transformers import AutoTokenizer
import argparse
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.datasets import loadRedditPairs
from src.wordFiltering import split_multitext_entries, convert_to_single_token

# Load arguments
parser = argparse.ArgumentParser(description="View words that are converted to multiple tokens by a model.")
parser.add_argument("lang", type=str, choices=["Spanish", "Japanese"],
                    help="The secondary language of the words to view (English is always the primary).")
parser.add_argument("dataset", type=str, choices=["Reddit"],
                    help="The dataset to use.")
parser.add_argument("model", type=str, choices=["Gemma", "Pythia"],
                    help="The model to use for tokenization.")
parser.add_argument("--verbose", "-v", action="store_true",
                    help="Print information about all words, not just those that are multi-token.")
args = parser.parse_args()

# Load the appropriate dataset
if args.dataset == "Reddit":
    dataset: pd.DataFrame = loadRedditPairs(args.lang)
else:
    raise ValueError("Unsupported dataset: {}".format(args.dataset))

dataset = split_multitext_entries(dataset)
dataset_count = len(dataset)

# Load the model
if args.model == "Gemma":
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b')
elif args.model == "Pythia":
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-2.8b')

# Process each word pair, printing those that are multi-token
bad_count = 0
bad_english_count = 0
bad_lang_count = 0
for row in dataset.iterrows():
    word_loc = row[0]
    word_english = row[1]['lang1']
    word_lang = row[1]['lang2']
    # Tokenize the words
    tokens_english = tokenizer.tokenize(word_english)
    tokens_lang = tokenizer.tokenize(word_lang)
    print(tokens_lang)
    dfasdf

    # Check if original word is multi-token
    tokens_english_original_length = len(tokens_english)
    tokens_lang_original_length = len(tokens_lang)

    # Process the word, then check if it is multi-token
    word_english_processed = convert_to_single_token(tokenizer, word_english)
    word_lang_processed = convert_to_single_token(tokenizer, word_lang)

    tokens_english_processed = tokenizer.tokenize(word_english_processed)
    tokens_lang_processed = tokenizer.tokenize(word_lang_processed)

    tokens_english_processed_length = len(tokens_english_processed)
    tokens_lang_processed_length = len(tokens_lang_processed)

    # Print information under following conditions:
    bad_english_conditions = \
        tokens_english_processed_length > 1 or \
        (tokens_english_original_length == 1 and len(word_english) != len(word_english_processed))
    bad_lang_conditions = \
        tokens_lang_processed_length > 1 or \
        (tokens_lang_original_length == 1 and len(word_lang) != len(word_lang_processed))
    
    if args.verbose or bad_english_conditions or bad_lang_conditions:
        print(f"{word_loc}: ")
        print(f"\tEnglish: Original: {word_english} ({tokens_english_original_length} tokens), Processed: {word_english_processed} ({tokens_english_processed_length} tokens)")
        print(f"\t{args.lang}: Original: {word_lang} ({tokens_lang_original_length} tokens), Processed: {word_lang_processed} ({tokens_lang_processed_length} tokens)")

    if bad_english_conditions or bad_lang_conditions:
        bad_count += 1
        if bad_english_conditions:
            bad_english_count += 1
        if bad_lang_conditions:
            bad_lang_count += 1

print(f"A total of {dataset_count} words were processed, of which {bad_count} ({bad_count/dataset_count:.2%})% were multi-token or had unnecessary information.")
print(f"Of these, {bad_english_count} ({bad_english_count/dataset_count:.2%})% were multi-token in English, and {bad_lang_count} ({bad_lang_count/dataset_count:.2%})% were multi-token in {args.lang}.")