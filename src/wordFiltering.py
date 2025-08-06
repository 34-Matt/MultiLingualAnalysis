import re
import pandas as pd
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from typing import Union, List

def tokenize(model: Union[HookedTransformer, AutoTokenizer], text: str) -> List[int]:
    '''Tokenizes the input with the given model.
    
    Args:
        model (HookedTransformer | Autoencoder): The transformer model
        text (str): The text to tokenize
    '''
    if isinstance(model, HookedTransformer):
        return model.to_tokens(text, prepend_bos=False).flatten().tolist()
    elif isinstance(model, AutoTokenizer):
        return model.tokenize(text)
    else:
        raise TypeError(f"Unable to work is model of type: {type(model)}")

def is_single_token_filter(model: HookedTransformer, text: str) -> bool:
    '''Filters out text that consists of more than one token.
    
    Args:
        model (HookedTransformer): The transformer model to use for tokenization.
        text (str): The text to filter.
    
    Returns:
        keep (bool): True if the text consists of a single token, False otherwise.
    '''
    tokens = tokenize(model, text)
    return len(tokens) == 1

def split_multitext_entries(dataset: pd.DataFrame) -> pd.DataFrame:
    '''Some words in the dictionary have comma separations when there are multiple translations. Split these into multiple entries.
    
    Args:
        dataset (pd.DataFrame): The dataset to process.
    
    Returns:
        pd.DataFrame: The processed dataset with split entries.
    '''
    # Split and explode each column
    for column in dataset.columns:
        dataset[column] = dataset[column].str.split(pat=',(?![^(]*\))', expand=False)
        dataset = dataset.explode(column)
        dataset[column] = dataset[column].str.strip()
        dataset = dataset[dataset[column] != '']  # Remove empty entries
        dataset = dataset.drop_duplicates(subset=[column])  # Remove duplicates
    
    dataset = dataset.reset_index(drop=True)
    return dataset
    
def convert_to_single_token(model: AutoTokenizer, text: str) -> str:
    '''Converts text to a single token using the transformer model.
    
    Args:
        model (AutoTokenizer): The transformer model to use for tokenization.
        text (str): The text to convert.
    
    Returns:
        token (int): The single token representation of the text.
    '''
    tokens = tokenize(model, text)

    text_length = len(text)
    if len(tokens) > 1:
        # Remove additional information in () or []
        text = re.sub(r'\(.*?\)|\[.*?\]', '', text)

        # Remove prepositions from english text
        if text.startswith('to '):
            text = text[3:]
        elif text.startswith('for '):
            text = text[4:]
        elif text.startswith('in '):
            text = text[3:]
        elif text.startswith('on '):
            text = text[3:]
        elif text.startswith('with '):
            text = text[5:]
        elif text.startswith('at '):
            text = text[3:]
        elif text.startswith('by '):
            text = text[3:]
        elif text.startswith('be '):
            text = text[3:]

        # Remove articles from english text
        if text.startswith('the '):
            text = text[4:]
        elif text.startswith('a '):
            text = text[2:]
        elif text.startswith('an '):
            text = text[3:]
        
        # Remove ' de' from spanish text
        if text.endswith(' de'):
            text = text[:-3]
        
        if text.startswith('de '):
            text = text[3:]

        # Remove articles from spanish text
        if text.startswith('la '):
            text = text[3:]
        elif text.startswith('el '):
            text = text[3:]
        elif text.startswith('los '):
            text = text[4:]
        elif text.startswith('las '):
            text = text[4:]
        elif text.startswith('un '):
            text = text[3:]
        elif text.startswith('una '):
            text = text[4:]

    text = text.strip()
    if len(text) == text_length:
        return text
    else:
        return convert_to_single_token(model, text)
    
def remove_non_single_token_entries(dataset: pd.DataFrame, model: HookedTransformer, keep_tokens: bool = False) -> pd.DataFrame:
    '''Removes entries from the dataset where either language is not a single token.
    
    Args:
        dataset (pd.DataFrame): The dataset to filter.
        model (HookedTransformer): The transformer model to use for tokenization.
        keep_tokens (bool): Whether to keep the token entries (lang#_token) (True) or not (False)

    Returns:
        pd.DataFrame: The filtered dataset containing only single-token entries.
    '''
    # Tokenize and filter the dataset
    dataset['lang1_tokens'] = dataset['lang1'].apply(lambda x: tokenize(model, x))
    dataset['lang2_tokens'] = dataset['lang2'].apply(lambda x: tokenize(model, x))
    
    dataset = dataset[dataset['lang1_tokens'].apply(len) == 1]
    dataset = dataset[dataset['lang2_tokens'].apply(len) == 1]

    if not keep_tokens:
        dataset = dataset.drop(columns=['lang1_tokens', 'lang2_tokens'])
    
    return dataset

def japanese_is_kanji(text: str) -> bool:
    '''Checks if the text contains Kanji.
    
    Args:
        text (str): The text to check.
    
    Returns:
        keep (bool): True if the text contains Kanji, False otherwise.
    '''
    return any('\u4e00' <= char <= '\u9faf' for char in text)  # Kanji range in Unicode

def japanese_is_katakana(text: str) -> bool:
    '''Checks if the text contains Katakana.
    
    Args:
        text (str): The text to check.
    
    Returns:
        keep (bool): True if the text is in Katakana, False otherwise.
    '''
    return any('\u30a0' <= char <= '\u30ff' for char in text)  # Katakana range in Unicode

def japanese_is_hiragana(text: str) -> bool:
    '''Checks if the text contains Hiragana.
    
    Args:
        text (str): The text to check.
    
    Returns:
        keep (bool): True if the text is in Hiragana, False otherwise.
    '''
    return any('\u3040' <= char <= '\u309f' for char in text)  # Hiragana range in Unicode

def is_japanese(text: str) -> bool:
    '''Checks if the text is in Japanese.
    
    Args:
        text (str): The text to check.
    
    Returns:
        keep (bool): True if the text is in Japanese, False otherwise.
    '''
    return japanese_is_kanji(text) or japanese_is_katakana(text) or japanese_is_hiragana(text)