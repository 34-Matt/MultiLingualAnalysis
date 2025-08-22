import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

from typing import List, Dict, Tuple, Union

from src import load_datasets, source_types
import src.embeddingCompare as EmbComp
from src.plotting_util import print_sorted_indices
from src.datasets import getLanguage

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze language model behavior.")

    parser_info = parser.add_argument_group("General Information")
    parser_info.add_argument("model", type=str, choices=["gemma", "pythia"], 
                        help="Model to use.")
    parser_info.add_argument("lang", type=str, choices=["all", "jpn", "esp"],
                        help="Language to use (`all` is only available for loading sentences).")
    parser_info.add_argument("source", type=str, choices=["reddit", "tatoeba"],
                        help="The source of the data")
    parser_info.add_argument("output_dir", type=str, default="output",
                        help="Output directory for the analysis results.")

    parser_action = parser.add_argument_group("Action Selector")
    parser_action.add_argument("embedding", action="store_true",
                        help="Add to run comparisons on single token embedding.")
    parser_action.add_argument("residual", action="store_true",
                        help="Add to run comparisons on single token words across the models.")

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
    elif model_name in ["t5", "t5_base"]:
        model = HookedTransformer.from_pretrained("google-t5/t5_base", dtype="float16")
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
        for lang in lang2:
            lang = getLanguage(lang)
            transform, differences = _embeddingCompare_single(args, lang, dataset, model)
        _embeddingCompare_multiple(args, dataset, model)
    else:
        transform, differences = _embeddingCompare_single(args, getLanguage(args.lang), dataset, model)
        differences = np.mean(differences, axis=0)
        diff_vector = np.dot(differences, transform[:,0].reshape((-1, 1))) * transform[:,0]
        data = pd.concat([
            EmbComp.check_conversion(model, dataset['lang1'], dataset['lang2'], diff_vector),
            EmbComp.check_conversion(model, dataset['lang2'], dataset['lang1'], -diff_vector),
        ], ignore_index=True)
        data.to_csv(os.path.join(args.output_dir, f"Conversion_difference_{args.model}_{args.lang}.csv"))

def _embeddingCompare_single(args: argparse.Namespace, lang2: str, dataset: pd.DataFrame, model: HookedTransformer) -> None:
    
    lang1_tokens = dataset['lang1_tokens']
    lang2_tokens = dataset['lang2_tokens']

    # Compare embeddings
    print("Comparing embeddings...")
    fig = plt.figure(figsize=(12, 8))
    #ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(121)

    print("Plotting PCA...")
    transformer, differences = EmbComp.plot_embedding_pca(ax, model, lang1_tokens, lang2_tokens, 2, "English", lang2)
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
    
    return transformer, differences

def _embeddingCompare_multiple(args: argparse.Namespace, dataset: pd.DataFrame, model: HookedTransformer):
    pass


def residualCompare(args: argparse.Namespace) -> None:
    '''Main function for comparing residuals of words.
    
    Args:
        args (argparse.Namespace): Parsed command line arguments.
    '''
    # Loads info
    print("Loading dataset and model...")
    if args.source not in source_types['word']:
        raise NotImplementedError("Sentence comparison is not implemented for comparing residuals.")

    dataset, model = load_info(args)
    if isinstance(dataset, dict):
        lang2 = list(dataset.keys())
        for lang in lang2:
            lang = getLanguage(lang)
            transform, differences, accuracies = _residualCompare_single(args, lang, dataset, model)
        _residualCompare_multiple(args, dataset, model)
    else:
        transform, differences, accuracies = _residualCompare_single(args, getLanguage(args.lang), dataset, model)
        best_ind = np.argmax(accuracies)


def _residualCompare_single(
        args: argparse.Namespace,
        language: str,
        dataset: pd.DataFrame,
        model: HookedTransformer) -> Tuple[
            List[np.ndarray],
            List[np.ndarray]
        ]:
    
    lang1 = dataset['lang1'].to_list()
    lang2 = dataset['lang2'].to_list()

    # Get number of residuals
    # Only care about the first few
    resid_count = model.cfg.n_layers
    resid_count = 4 if resid_count > 4 else resid_count

    # get number of unique tokens in model
    unique_token_count = len(model.tokenizer.vocab)
    topk = unique_token_count // 2000 # Set to looking at first 0.05% of predictions
    topk = 10 if topk < 10 else topk

    # Run model & cache residual stream
    _, cache1 = model.run_with_cache(lang1, prepend_bos=False)
    _, cache2 = model.run_with_cache(lang2, prepend_bos=False)

    # Compare embeddings
    print("Comparing embeddings...")
    fig = plt.figure(figsize=(12, 8))

    # Iterate over first few residuals
    subplot_number = 20 + resid_count*100
    transformers = []
    differences = []
    average_differences = []
    accuracies = []
    for rsc in range(resid_count):
        transformer, difference, accuracy = EmbComp.compare_forward_residual(model, lang1, lang2, 2, rsc, topk)
        transformers.append(transformer)
        differences.append(difference)
        average_differences.append(np.mean(difference, axis=0))
        accuracies.append(accuracy)

        print("Plotting PCA...")
        ax = fig.add_subplot(subplot_number + 1 + rsc*2)
        EmbComp.plot_residual_pca(ax, cache1, cache2, 2, rsc, "English", language, transform=transformer)

        print("Plotting PCA differences...")
        ax2 = fig.add_subplot(subplot_number + 2 + rsc*2)
        EmbComp.plot_residual_pca(ax2, cache1, cache2, 2, rsc, "English", language, transform=transformer, scatter=False)
        
        # Overwrite Titles
        if rsc == 0:
            ax.set_title("Difference")
            ax2.set_title("Direction")
        else:
            ax.set_title("")
            ax2.set_title("")
        
        ax.set_ylabel(f"Residual Layer {rsc}")
    
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax2.set_xlim(xmin, xmax)
        ax2.set_ylim(ymin, ymax)
    
    # Save Plot
    fig.suptitle(f"Embedding Comparison: {args.model} - {language} vs English")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"{language}residual.png"))

    # Save Transform
    with pd.ExcelWriter(os.path.join(args.output_dir, f"{language}residual.xlsx")) as writer:
        for trans_ind in range(len(transformers)):
            for axis_ind in range(transformers[trans_ind].shape[1]):
                pd.Series(transformers[trans_ind][:,axis_ind]).to_excel(writer, sheet_name=f"Layer{trans_ind}_Axis{axis_ind}")
            pd.Series(average_differences[trans_ind]).to_excel(writer, sheet_name=f"Layer{trans_ind}_Difference")
        pd.Series(accuracies).to_excel(writer, sheet_name="LayerAccuracy")

    # Find the most accurate result
    best_index = np.argmax(accuracies)
    print(accuracies)
    print(f"Layer with the best accuracy is layer {best_index}")
    print(f"Had the best accuracy of {accuracies[best_index]:.2f}% when looking at the top {topk} predicted tokens")
    
    return transformer, differences, accuracies

def _residualCompare_multiple(args: argparse.Namespace, dataset: pd.DataFrame, model: HookedTransformer):
    pass

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    if args.embedding:
        embeddingCompare(args)
    if args.residual:
        residualCompare(args)
