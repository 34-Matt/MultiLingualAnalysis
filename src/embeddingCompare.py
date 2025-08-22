import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformer_lens import HookedTransformer, ActivationCache

from .plotting_util import line_segment
from .math_util import my_pca
from .wordFiltering import tokenize, detokenize

from typing import Union, List, Tuple, Dict

def compare_word_embeddings(model: HookedTransformer, token1: int, token2: int) -> Union[np.ndarray, float]:
    '''Calculate the vector that describes the translation from token1 to token2.
    
    Args:
        model (HookedTransformer): The model to use to embed the tokens.
        token1 (int): The first token to compare.
        token2 (int): The second token to compare.

    Returns:
        Vector (ndarray): The normalized vector that describes the translation from token1 to token2.
        Magnitude (float): The magnitude of the vector.
    '''
    # Get the embedding of the tokens
    embedding1 = model.embed(token1)
    embedding2 = model.embed(token2)

    # Calculate the difference between the two embeddings
    difference = embedding2 - embedding1

    # Normalize the difference vector
    norm = torch.norm(difference, p=2)
    if norm == 0:
        return difference.numpy(), norm.item()
    else:
        return difference.numpy() / norm, norm.item()
    
def compute_average_embedding_direction(model: HookedTransformer, token1: List[int], token2: List[int]) -> Union[np.ndarray, np.ndarray]:
    '''Calculate the average vector that describes the translation from token1 to token2.
    
    Args:
        model (HookedTransformer): The model to use to embed the tokens.
        token1 (List[int]): The first tokens to compare.
        token2 (List[int]): The second tokens to compare.

    Returns:
        mean (ndarray): The mean vector that describes the translation from token1 to token2.
        std (ndarray): The standard deviation of the vector.
    '''
    # Get the embedding of the tokens
    embeddings1 = [model.embed(t).detach().numpy() for t in token1]
    embeddings2 = [model.embed(t).detach().numpy() for t in token2]

    # Calculate the difference between the two embeddings
    differences = np.array(embeddings2) - np.array(embeddings1)

    mean_vector = np.mean(differences, axis=0).flatten()
    std_vector = np.std(differences, axis=0).flatten()

    min_vector = np.min(differences, axis=0).flatten()
    max_vector = np.max(differences, axis=0).flatten()

    return mean_vector, std_vector, min_vector, max_vector

def print_embedding_compare(mean_vector: np.ndarray, std_vector: np.ndarray, min_vector: np.ndarray, max_vector: np.ndarray, n_interest: int) -> None:
    '''Print the top n dimensions with the highest magnitude, along with the corresponding magnitude and std.
    
    Args:
        mean_vector (ndarray): The mean vector that describes the translation from token1 to token2.
        std_vector (ndarray): The standard deviation of the vector.
        n_interest (int): The number of dimensions from mean and std to print.
    '''
    mean_vector_sorted_ind = np.argsort(mean_vector)

    print("Highest mean values:")
    for n in range(n_interest):
        ind = mean_vector_sorted_ind[-(n+1)]
        print(f"Index: {ind},\tMean: {mean_vector[ind]:.4f},\tStd: {std_vector[ind]:.4f},\tMin: {min_vector[ind]:.4f},\tMax: {max_vector[ind]:.4f}")

def plot_embedding_pca(
        ax: plt.Axes,
        model: HookedTransformer,
        token1: List[int],
        token2: List[int],
        n_dim: int,
        token1_name: str,
        token2_name: str,
        transform: np.ndarray = None,
        scatter: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    '''Plot the PCA of the embeddings of the tokens.
    
    Args:
        ax (plt.Axes): The axes to plot on.
        model (HookedTransformer): The model to use to embed the tokens.
        token1 (List[int]): The first tokens to compare.
        token2 (List[int]): The second tokens to compare.
        n_dim (int): The number of dimensions when creating the PCA.
        token1_name (str): The name of the first token.
        token2_name (str): The name of the second token.
        transform (np.ndarray, optional): The transform matrix to use for PCA. If None, it will be calculated.
        scatter (bool, optional): Whether to plot the points as a scatter plot (True) or as lines from token 1 to 2 (False). Defaults to True.
    
    Returns:
        transform (np.ndarray): The transform matrix used for PCA.
        differences (np.ndarray): The average difference from token1 to token2
    '''

    # Verify n_dim is plottable
    if n_dim <= 0 or n_dim > 3:
        raise ValueError("n_dim must be between 1, 2, or 3.")

    # Get the embedding of the tokens
    for t in token1:
        e = model.embed(t)
        _, data = compare_unembedding_distribution(model, e, t, 5)

    embeddings1 = [model.embed(t).detach().numpy() for t in token1]
    embeddings2 = [model.embed(t).detach().numpy() for t in token2]

    embeddings1 = np.array(embeddings1).reshape(len(token1), -1)
    embeddings2 = np.array(embeddings2).reshape(len(token2), -1)

    if transform is None:
        # Calculate the difference between the two embeddings
        differences = embeddings2 - embeddings1
        
        # Normalize difference
        diff_max = np.max(differences)
        differences = differences * (100/diff_max)
        #differences *= 10  # Scale average difference can be very small

        # Perform PCA on the differences
        _, transform = my_pca(differences, n_components=n_dim)

    differences = embeddings2 - embeddings1
    
    # Apply the PCA transformation
    token1_pca = embeddings1 @ transform
    token2_pca = embeddings2 @ transform

    # Plot the PCA result
    if n_dim == 1:
        VP = ax.boxplot([token1_pca[:, 0], token2_pca[:, 0]],
                        positions=[1, 3],
                        widths=1.5,
                        patch_artist=True,
                        showmeans=False,
                        showfliers=False,
                        medianprops={"color": "white", "linewidth": 0.5},
                        boxprops={"facecolor": "C0", "edgecolor": "white", "linewidth": 0.5},
                        whiskerprops={"color": "C0", "linewidth": 1.5},
                        capprops={"color": "C0", "linewidth": 1.5}
                        )

        ax.set(xlim=(0, 4), xticks=np.arange(1, 4))
        ax.set_xticklabels([token1_name, token2_name])
        ax.set_ylabel("PCA Dimension 1")
    
    elif n_dim == 2:
        if scatter:
            ax.scatter(token1_pca[:, 0], token1_pca[:, 1], color="Blue", label=token1_name)
            ax.scatter(token2_pca[:, 0], token2_pca[:, 1], color="Red", label=token2_name)
            ax.legend()
        else:
            line_segment(token1_pca, token2_pca, ax=ax, cmap=plt.get_cmap("RdBu_r"))
        ax.set_xlabel("PCA Dimension 1")
        ax.set_ylabel("PCA Dimension 2")
    
    elif n_dim == 3:
        if scatter:
            ax.scatter(token1_pca[:, 0], token1_pca[:, 1], token1_pca[:, 2], color="Blue", label=token1_name)
            ax.scatter(token2_pca[:, 0], token2_pca[:, 1], token2_pca[:, 2], color="Red", label=token2_name)
        else:
            line_segment(token1_pca, token2_pca, ax=ax, cmap=plt.get_cmap("RdBu_r"))
        ax.set_xlabel("PCA Dimension 1")
        ax.set_ylabel("PCA Dimension 2")
        ax.set_zlabel("PCA Dimension 3")
        ax.legend()
    
    else:
        raise ValueError("n_dim must be between 1, 2, or 3.")
    ax.set_title(f"PCA of {token1_name} and {token2_name} embeddings")
    
    return transform, differences

def compare_unembedding_distribution(
        model: HookedTransformer,
        input_embedding: torch.Tensor,
        target_token: int,
        n_interest: int
) -> Tuple[float, Dict[str, float]]:
    '''Calculates the probability of an embedding being a desired token.
    
    Args:
        model (HookedTransformer): The model used for embedding.
        input_embedding (torch.Tensor): The embedding to compare.
        target_token (int): The token to compare against.
        n_interest (int): The number of top probabilities to return.
    
    Returns:
        probability (float): The probability of the embedding being the target token.
        top_tokens (Dict[str, float]): The top n tokens with the highest probabilities.
    '''
    # Get the logits for the input embedding
    logits = model.unembed(input_embedding)
    logits = logits.squeeze(0)
    logits = logits.softmax(dim=0)

    # Get target token probability
    target_prob = logits[target_token].item()

    # Get top n tokens
    top_probabilities, top_indices = torch.topk(logits, n_interest)
    
    top_tokens = {}
    for i in range(n_interest):
        token = model.tokenizer.decode([top_indices[i].item()])
        top_tokens[token] = top_probabilities[i].item()
    
    return target_prob, top_tokens

def check_conversion(
        model: HookedTransformer,
        input_str: List[str],
        target_str: List[str],
        difference: np.ndarray,
        top_n: int = 5
) -> pd.DataFrame:
    '''Computes what the input string should be based on the difference array.
    
    Args:
        model (HookedTransformer): The model used to tokenize and embed the input text
        input_str (List[str]): A list of strings to test
        target_str (List[str]): A list of strings that should be generated from the input strings
        difference (np.ndarray): The difference array that should get the input string to the target string
        top_n (int): The number of top predictions for the input text

    Returns:
        output (pd.DataFrame): The output for each text. The columns are:
            - Input (str) for the input string
            - Top_n (str) for each of the top top_n predictions
            - Prop_n (float) for each of probabilities of the top top_n predictions
            - Target (str) for the target string
            - Target_Prop (float) for the probability of the prediction being the target
    '''
    results = []
    for inp, tgt in zip(input_str, target_str):
        # Tokenize input and target
        inp_token = tokenize(model, inp)
        tgt_token = tokenize(model, tgt)

        # Get input embedding and apply difference
        inp_embedding = model.embed(inp_token)
        new_embedding = inp_embedding + torch.tensor(difference, dtype=inp_embedding.dtype, device=inp_embedding.device)

        # Get probability and top tokens
        target_prob, top_tokens = compare_unembedding_distribution(model, new_embedding, tgt_token, top_n)

        # Prepare row for DataFrame
        row = {
            "Input": inp,
            "Target": tgt,
            "Target_Prob": target_prob
        }

        # Add top_n predictions
        for i, (tok, prob) in enumerate(top_tokens.items()):
            row[f"Top_{i+1}"] = tok
            row[f"Prop_{i+1}"] = prob
        results.append(row)

    return pd.DataFrame(results)

def plot_residual_pca(
        ax: plt.Axes,
        cache1: ActivationCache,
        cache2: ActivationCache,
        n_dim: int,
        layer: int,
        token1_name: str,
        token2_name: str,
        transform: np.ndarray = None,
        scatter: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    '''Plot the PCA of the embeddings of the tokens.
    
    Args:
        ax (plt.Axes): The axes to plot on.
        cache1 (ActivationCache): The first texts to compare.
        cashe2 (ActivationCache): The second texts to compare.
        n_dim (int): The number of dimensions when creating the PCA.
        token1_name (str): The name of the first token.
        token2_name (str): The name of the second token.
        transform (np.ndarray, optional): The transform matrix to use for PCA. If None, it will be calculated.
        scatter (bool, optional): Whether to plot the points as a scatter plot (True) or as lines from token 1 to 2 (False). Defaults to True.
    
    Returns:
        transform (np.ndarray): The transform matrix used for PCA.
        differences (np.ndarray): The average difference from token1 to token2
    '''

    # Verify n_dim is plottable
    if n_dim <= 0 or n_dim > 3:
        raise ValueError("n_dim must be between 1, 2, or 3.")

    # Extract residual stream at chosen layer (just final position)
    res1 = cache1["resid_post", layer][:, -1, :]
    res2 = cache2["resid_post", layer][:, -1, :]

    if transform is None:
        # Calculate the difference between the two embeddings
        differences = res2 - res1
        
        # Normalize difference
        diff_max = np.max(differences)
        differences = differences * (100/diff_max)
        #differences *= 10  # Scale average difference can be very small

        # Perform PCA on the differences
        _, transform = my_pca(differences, n_components=n_dim)

    differences = res2 - res1
    
    # Apply the PCA transformation
    token1_pca = res1 @ transform
    token2_pca = res2 @ transform

    # Plot the PCA result
    if n_dim == 1:
        VP = ax.boxplot([token1_pca[:, 0], token2_pca[:, 0]],
                        positions=[1, 3],
                        widths=1.5,
                        patch_artist=True,
                        showmeans=False,
                        showfliers=False,
                        medianprops={"color": "white", "linewidth": 0.5},
                        boxprops={"facecolor": "C0", "edgecolor": "white", "linewidth": 0.5},
                        whiskerprops={"color": "C0", "linewidth": 1.5},
                        capprops={"color": "C0", "linewidth": 1.5}
                        )

        ax.set(xlim=(0, 4), xticks=np.arange(1, 4))
        ax.set_xticklabels([token1_name, token2_name])
        ax.set_ylabel("PCA Dimension 1")
    
    elif n_dim == 2:
        if scatter:
            ax.scatter(token1_pca[:, 0], token1_pca[:, 1], color="Blue", label=token1_name)
            ax.scatter(token2_pca[:, 0], token2_pca[:, 1], color="Red", label=token2_name)
            ax.legend()
        else:
            line_segment(token1_pca, token2_pca, ax=ax, cmap=plt.get_cmap("RdBu_r"))
        ax.set_xlabel("PCA Dimension 1")
        ax.set_ylabel("PCA Dimension 2")
    
    elif n_dim == 3:
        if scatter:
            ax.scatter(token1_pca[:, 0], token1_pca[:, 1], token1_pca[:, 2], color="Blue", label=token1_name)
            ax.scatter(token2_pca[:, 0], token2_pca[:, 1], token2_pca[:, 2], color="Red", label=token2_name)
        else:
            line_segment(token1_pca, token2_pca, ax=ax, cmap=plt.get_cmap("RdBu_r"))
        ax.set_xlabel("PCA Dimension 1")
        ax.set_ylabel("PCA Dimension 2")
        ax.set_zlabel("PCA Dimension 3")
        ax.legend()
    
    else:
        raise ValueError("n_dim must be between 1, 2, or 3.")
    ax.set_title(f"PCA of {token1_name} and {token2_name} embeddings")
    
    return transform, differences

def compare_forward_residual(
        model: HookedTransformer,
        text1: List[str],
        text2: List[str],
        n_dim: int,
        layer: int,
        topk: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    '''Plot the PCA of the embeddings of the tokens.
    
    Args:
        model (HookedTransformer): The model used to predict the text.
        text1 (List[str]): The first texts to compare.
        text2 (List[str]): The second texts to compare.
        n_dim (int): The number of dimensions when creating the PCA.
        layer (int): The layer of the model to evaluate
        topk (int): The number of samples to validate if model predicts the correct word

    Returns:
        transform (np.ndarray): The transform matrix used for PCA.
        differences (np.ndarray): The average difference from text1 to text2
        accuracy (float): The prediction accuracy going from text1 to text2
    '''
    # Run model & cache residual stream
    _, cache1 = model.run_with_cache(text1, prepend_bos=False)
    _, cache2 = model.run_with_cache(text2, prepend_bos=False)

    # Extract residual stream at chosen layer (just final position)
    res1 = cache1["resid_post", layer][:, -1, :]  # shape [N, d_model]
    res2 = cache2["resid_post", layer][:, -1, :]

    # Compute average transformation vector
    diffs = res2 - res1
    avg_diff = diffs.mean(dim=0)#, keepdim=True)
    
    # Normalize difference
    diff_max = diffs.max()
    differences = diffs * (100/diff_max)

    # Perform PCA on the differences
    _, transform = my_pca(differences.detach().numpy(), n_components=n_dim)
    
    # Calculate the difference between the two embeddings
    diffs = res2 - res1

    # Continue forward pass
    hook_edit_res = lambda resid, hook: resid + avg_diff

    # Run again with hook that overwrites this layer’s residual stream
    logits = model.run_with_hooks(
        text1, 
        prepend_bos=False, 
        fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_edit_res)]
    )

    # Get predictions
    topk_preds = torch.topk(logits[:, -1, :], k=topk, dim=-1).indices
    pred_words = [[model.to_string(tok) for tok in row] for row in topk_preds]

    # Compute accuracy
    correct = sum(text2[i] in pred_words[i] for i in range(len(text2)))
    accuracy = 100 * (correct / len(text2))

    return transform, diffs.detach().numpy(), accuracy