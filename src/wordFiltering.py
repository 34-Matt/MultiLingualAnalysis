from transformer_lens import HookedTransformer

def is_single_token_filter(model: HookedTransformer, text: str) -> bool:
    '''Filters out text that consists of more than one token.
    
    Args:
        model (HookedTransformer): The transformer model to use for tokenization.
        text (str): The text to filter.
    
    Returns:
        keep (bool): True if the text consists of a single token, False otherwise.
    '''
    tokens = model.to_tokens(text, prepend_bos=False, append_eos=False)
    return len(tokens) == 1

def convert_to_single_token(model: HookedTransformer, text: str) -> str:
    '''Converts text to a single token using the transformer model.
    
    Args:
        model (HookedTransformer): The transformer model to use for tokenization.
        text (str): The text to convert.
    
    Returns:
        token (int): The single token representation of the text.
    '''
    tokens = model.to_tokens(text, prepend_bos=False, append_eos=False)
    if len(tokens) == 1:
        return text
    else:
        # Remove 'to ', 'the ', 'a ', 'an ' from english text
        if text.startswith('to '):
            text = text[3:]
        elif text.startswith('the '):
            text = text[4:]
        elif text.startswith('a '):
            text = text[2:]
        elif text.startswith('an '):
            text = text[3:]
        
        # Remove 'la ', 'el ', 'los ', 'las ', 'un ', 'una ' from spanish text
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

        # Remove ' de' from spanish text
        if text.endswith(' de'):
            text = text[:-3]

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