import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

from typing import List, Dict, Tuple, Union

from src import load_datasets, source_types
import src.embeddingCompare as EmbComp
from src.plotting_util import print_sorted_indices

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze language model behavior.")
    parser.add_argument("model", type=str, choices=["gemma", "pythia"], 
                        help="Model to use.")
    parser.add_argument("lang", type=str, choices=["all", "jpn", "esp"],
                        help="Language to use (`all` is only available for loading sentences).")
    parser.add_argument("source", type=str, choices=["reddit", "tatoeba"],
                        help="The source ")
    parser.add_argument("output_dir", type=str, default="output",
                        help="Output directory for the analysis results.")
    return parser.parse_args()

def load_info(args: argparse.Namespace) -> Tuple[Union[pd.DataFrame, Dict[str, pd.DataFrame]], HookedTransformer]:
    """
    Load the necessary information for the analysis.
    
    Args:
        args: Command line arguments containing model and dataset information.
    
    Returns:
        words (pd.DataFrame | Dict[str, pd.DataFrame]): DataFrame containing the words from the dataset (if all is selected for words, will return a list of dictionaries.).
        model (HookedTransformer): The pre-trained model to use for embeddings.
    """
    # Load model and tokenizer
    model_name = args.model.lower()
    if model_name in ["gemma", "gemma-2b"]:
        model = HookedTransformer.from_pretrained_no_processing("google/gemma-2b-it", dtype="float16")
    elif model_name in ["pythia", "pythia-70", "pythia-70m"]:
        model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m", dtype="float16")
    else:
        raise ValueError("Unsupported model. Choose 'gemma' or 'pythia'.")
    
    # Load dataset
    dataset = load_datasets(args.lang, args.source, model=model, remove_non_single_token=True, keep_tokens=True)

    return dataset, model

def embeddingCompare(args: argparse.Namespace) -> None:
    '''Main function for comparing embeddings of words.
    
    Args:
        args (argparse.Namespace): Parsed command line arguments.
    '''
    # Loads info
    print("Loading dataset and model...")
    if args.source not in source_types['word']:
        raise NotImplementedError("Sentence comparison is not implemented for comparing embeddings.")

    dataset, model = load_info(args)
    if isinstance(dataset, dict):
        lang2 = list(dataset.keys())
        raise NotImplementedError("Comparing embeddings for multiple languages is not implemented beyond this point.")
    else:
        lang2 = "Japanese" if args.lang.lower() in ["japanese", "jpn"] else "Spanish"

    lang1_tokens = dataset['lang1_tokens']
    lang2_tokens = dataset['lang2_tokens']

    # Compare embeddings
    print("Comparing embeddings...")
    fig = plt.figure(figsize=(12, 8))
    #ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(121)

    print("Plotting PCA...")
    transformer =EmbComp.plot_embedding_pca(ax, model, lang1_tokens, lang2_tokens, 2, "English", lang2)
    print(transformer)
    ax.set_xlabel("PCA Dimension 1")
    ax.set_ylabel("PCA Dimension 2")
    #ax.set_zlabel("PCA Dimension 3")
    plt.tight_layout()

    ax2 = fig.add_subplot(122)
    print("Plotting PCA differences...")
    EmbComp.plot_embedding_pca(ax2, model, lang1_tokens, lang2_tokens, 2, "English", lang2, transform=transformer, scatter=False)
    ax2.set_xlabel("PCA Dimension 1")
    ax2.set_ylabel("PCA Dimension 2")
    #ax2.set_zlabel("PCA Dimension 3")

    fig.suptitle(f"Embedding Comparison: {args.model} - {lang2} vs English")
    ax.set_title("")
    ax2.set_title("")

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    
    # Save Plot
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"{lang2}embedding.png"))

    # Save Transform
    with pd.ExcelWriter(os.path.join(args.output_dir, f"{lang2}embedding.xlsx")) as writer:
        for ind in range(transformer.shape[1]):
            pd.Series(transformer[:,ind]).to_excel(writer, sheet_name=f"Axis{ind}")

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    embeddingCompare(args)