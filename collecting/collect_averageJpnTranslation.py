from transformer_lens import HookedTransformer
from src import datasets



eng_jpn_words = datasets.loadMyWordPairs("japanese")



model = HookedTransformer.from_pretrained("google/gemma-2b-it", dtype="float16")