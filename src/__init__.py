from datasets import loadRedditPairs, loadTatoebaPairs
from wordFiltering import split_multitext_entries, convert_to_single_token
from transformers import AutoTokenizer

# Decorators for loading datasets
def filter_words(func):
    def wrapper(*args, model: AutoTokenizer, **kwargs):
        dataset = func(*args, **kwargs)

        dataset = split_multitext_entries(dataset)
        dataset["lang1"] = dataset["lang1"].apply(lambda x: convert_to_single_token(model, x))
        dataset["lang2"] = dataset["lang2"].apply(lambda x: convert_to_single_token(model, x))

        return dataset
    return wrapper

# Loading words
@filter_words
def loadJapaneseWords():
    return loadRedditPairs("Japanese")

@filter_words
def loadSpanishWords():
    return loadRedditPairs("Spanish")

def loadJapaneseSentences():
    return loadTatoebaPairs("English", ["Japanese"])

def loadSpanishSentences():
    return loadTatoebaPairs("English", ["Spanish"])

def loadAllSentences():
    """Load all sentences from Tatoeba dataset."""
    return loadTatoebaPairs("English", ["Japanese", "Spanish"])