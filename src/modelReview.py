import torch
from transformer_lens import HookedTransformer

from src.wordFiltering import is_japanese

from typing import Union, Tuple

def find_japanese_layer(model: HookedTransformer, text: str, max_jpn_token_length: int = 1) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """Finds the first layer where the model predicts the next token as Japanese.

    Args:
        model (HookedTransformer): The transformer model to use for tokenization.
        text (str): The text to check.
        max_jpn_token_length (int): The maximum length of the average Japanese token. (Required if Japanese letters are tokenized as multiple tokens)
    
    Returns:
        layer_number (int): The layer where the model predicts the next token as Japanese (-1 if no such layer exists).
        logits (torch.Tensor): The logits of the model.
        cache (torch.Tensor): The cache of the model.
    """
    if max_jpn_token_length <= 0:
        raise ValueError("max_jpn_token_length must be greater than 0")

    # Convert input to tokens
    tokens = model.to_tokens(text)
    if len(tokens) == 0:
        return -1
    
    # If Japanese tokens are longer than 1, generate tokens until next token is the max length
    if max_jpn_token_length > 1:
        tokens = tokens + model.generate(tokens, max_new_tokens=max_jpn_token_length-1)[1:]

    # Get cache of next token predictions
    logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)

    # Check when the next token is Japanese
    for layer in range(model.cfg.n_layers):
        # Get the logits for the next token
        next_token_logits = logits[layer, -1, :]
        
        # Get the top token prediction
        top_token = torch.argmax(next_token_logits).item()
        
        # Check if the top token is Japanese
        if is_japanese(model.to_string(top_token)):
            return layer, logits, cache
    return -1, logits, cache

def find_japanese_probability(model: HookedTransformer, text: str, max_jpn_token_length: int = 1) -> Tuple[float, int, torch.Tensor]:
    """Finds the probability of the model predicting the next token as Japanese.

    Args:
        model (HookedTransformer): The transformer model to use for tokenization.
        text (str): The text to check.
        max_jpn_token_length (int): The maximum length of the average Japanese token. (Required if Japanese letters are tokenized as multiple tokens)
    
    Returns:
        probability (float): The probability of the next token being Japanese.
        nonjapanese_count (int): The number of non-Japanese tokens until the first Japanese token (-1 if no Japanese token is found).
        sorted_indices (torch.Tensor): The sorted indices of the most likely next token.
    """
    logits = model(text, return_type="logits")
    if len(logits) == 0:
        return 0.0, -1, logits
    
    logits_next = logits[:, -1, :].flatten()

    logits_sorted, indice = torch.sort(logits_next, descending=True)
    logits_sorted = logits_sorted.softmax(dim=-1)

    for i in range(len(logits_sorted)):
        if is_japanese(model.to_string(indice[i])):
            return logits_sorted[i].item(), i, indice
    return 0.0, -1, indice