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

