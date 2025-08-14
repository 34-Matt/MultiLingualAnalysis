from .datasets import loadRedditPairs, loadTatoebaPairs, LANGUAGE_LONGHAND, LANGUAGE_SHORTHAND
from .wordFiltering import split_multitext_entries, convert_to_single_token, remove_non_single_token_entries
from transformers import AutoTokenizer
import pandas as pd
from typing import Dict, Tuple, Union


source_types = {
    "word": ["reddit"],
    "sentence": ["tatoeba"]
}

def load_datasets(
        lang: str,
        source: str,
        *,
        model: AutoTokenizer = None,
        filter_words: bool = False,
        remove_non_single_token: bool = False,
        keep_tokens: bool = False
        ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Load datasets based on language and dataset type."""
    # Validate inputs
    lang = lang.lower()
    source = source.lower()

    if source in source_types['word']:
        # Load words
        if lang in LANGUAGE_LONGHAND["Japanese"]:
            data = loadRedditPairs("Japanese")
        elif lang in LANGUAGE_LONGHAND["Spanish"]:
            data = loadRedditPairs("Spanish")
        elif lang in ["all", "a"]:
            data = {
                "jpn": loadRedditPairs("Japanese"),
                "esp": loadRedditPairs("Spanish")
            }
        else:
            raise ValueError(f"Unsupported language: {lang}")
        
        # Filter words if specified
        if (filter_words or remove_non_single_token) and model is not None:
            if isinstance(data, dict):
                for key in data:
                    data[key]['lang1'] = data[key]['lang1'].apply(lambda x: convert_to_single_token(model, x))
                    data[key]['lang2'] = data[key]['lang2'].apply(lambda x: convert_to_single_token(model, x))
            else:
                data['lang1'] = data['lang1'].apply(lambda x: convert_to_single_token(model, x))
                data['lang2'] = data['lang2'].apply(lambda x: convert_to_single_token(model, x))
        
        if remove_non_single_token and model is not None:
            if isinstance(data, dict):
                for key in data:
                    data[key] = remove_non_single_token_entries(data[key], model, keep_tokens)
            else:
                data = remove_non_single_token_entries(data, model, keep_tokens)
    
    elif source in source_types["sentence"]:
        # Load sentences
        if lang in LANGUAGE_LONGHAND["Japanese"]:
            data = loadTatoebaPairs("English", "Japanese")
        elif lang in LANGUAGE_LONGHAND["Spanish"]:
            data = loadTatoebaPairs("English", "Spanish")
        elif lang in ["all", "a"]:
            data = {
                "jpn": loadTatoebaPairs("English", "Japanese"),
                "spn": loadTatoebaPairs("English", "Spanish"),
            }
        else:
            raise ValueError(f"Unsupported language: {lang}")
        
    else:
        raise ValueError(f"Unsupported source: {source}")

    return data
