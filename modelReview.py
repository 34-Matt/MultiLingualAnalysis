import torch
from transformer_lens import HookedTransformer

from typing import Union

Sentences = {
    "Morning" :{
        "English": "Good morning.",
        "Spanish": "Buenos dias.",
        "Japanese": "おはようございます",
    },

    "Shopping" :{
        "English": "I want to buy a new bicycle.",
        "Spanish": "Quiero comprar una nueva bicicleta.",
        "Japanese": "新しい自転車を買いたい。",
    },

    "Eating" :{
        "English": "It is better not to eat before going to bed.",
        "Spanish": "Es mejor no comer antes de acostarse.",
        "Japanese": "寝る前には食べない方がいいですよ。",
    },
}

Words ={
    "I" :{
        "English": "I",
        "Spanish": "Yo",
        "Japanese": "私",
    },

    "You" :{
        "English": "You",
        "Spanish": "Tú",
        "Japanese": "あなた",
    },

    "to run" :{
        "English": "to run",
        "Spanish": "correr",
        "Japanese": "走る",
    },

    "to walk" :{
        "English": "to walk",
        "Spanish": "caminar",
        "Japanese": "歩く",
    },

    "fast" :{
        "English": "fast",
        "Spanish": "rápido",
        "Japanese": "速い",
    },

    "slow" :{
        "English": "slow",
        "Spanish": "lento",
        "Japanese": "遅い",
    },

    "rabbit" :{
        "English": "rabbit",
        "Spanish": "conejo",
        "Japanese": "兎",
    },

    "turtle" :{
        "English": "turtle",
        "Spanish": "tortuga",
        "Japanese": "亀",
    },
}

def compare_word_embeddings(model: HookedTransformer, token1: int, token2: int) -> Union[torch.Tensor, float]:
    '''Calculate the vector that describes the translation from token1 to token2.
    
    Args:
        model (HookedTransformer): The model to use to embed the tokens.
        token1 (int): The first token to compare.
        token2 (int): The second token to compare.

    Returns:
        Vector: The normalized vector that describes the translation from token1 to token2.
        Magnitude: The magnitude of the vector.
    '''
    # Get the embedding of the tokens
    embedding1 = model.embed(token1)
    embedding2 = model.embed(token2)

    # Calculate the difference between the two embeddings
    difference = embedding2 - embedding1  # Shape: (1, d_model)

    # Normalize the difference vector
    norm = torch.norm(difference, p=2)  # L2 norm
    if norm == 0:
        return difference, norm.item()
    else:
        return difference / norm, norm.item()

