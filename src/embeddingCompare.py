import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformer_lens import HookedTransformer

from typing import Union, List

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

    return mean_vector, std_vector

def print_embedding_compare(mean_vector: np.ndarray, std_vector: np.ndarray, n_interest: int) -> None:
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
        print(f"Index: {ind}, Mean: {mean_vector[ind]}, Std: {std_vector[ind]}")

def plot_embedding_pca(
        ax: plt.Axes,
        model: HookedTransformer,
        token1: List[int],
        token2: List[int],
        n_dim: int,
        token1_name: str,
        token2_name: str) -> PCA:
    '''Plot the PCA of the embeddings of the tokens.
    
    Args:
        ax (plt.Axes): The axes to plot on.
        model (HookedTransformer): The model to use to embed the tokens.
        token1 (List[int]): The first tokens to compare.
        token2 (List[int]): The second tokens to compare.
        n_dim (int): The number of dimensions when creating the PCA.
        token1_name (str): The name of the first token.
        token2_name (str): The name of the second token.
    
    Returns:
        pca (PCA): The PCA object used to transform the data.
    '''

    # Verify n_dim is plottable
    if n_dim <= 0 or n_dim > 3:
        raise ValueError("n_dim must be between 1, 2, or 3.")

    # Get the embedding of the tokens
    embeddings1 = [model.embed(t).detach().numpy() for t in token1]
    embeddings2 = [model.embed(t).detach().numpy() for t in token2]

    embeddings1 = np.array(embeddings1).reshape(len(token1), -1)
    embeddings2 = np.array(embeddings2).reshape(len(token2), -1)

    # Calculate the difference between the two embeddings
    differences = embeddings2 - embeddings1

    # Perform PCA on the differences
    pca = PCA(n_components=n_dim)
    pca.fit_transform(differences)

    token1_pca = pca.transform(embeddings1)
    token2_pca = pca.transform(embeddings2)

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
        ax.scatter(token1_pca[:, 0], token1_pca[:, 1], color="C0", label=token1_name)
        ax.scatter(token2_pca[:, 0], token2_pca[:, 1], color="C1", label=token2_name)
        ax.set_xlabel("PCA Dimension 1")
        ax.set_ylabel("PCA Dimension 2")
        ax.legend()
    
    elif n_dim == 3:
        ax.scatter(token1_pca[:, 0], token1_pca[:, 1], token1_pca[:, 2], color="C0", label=token1_name)
        ax.scatter(token2_pca[:, 0], token2_pca[:, 1], token2_pca[:, 2], color="C1", label=token2_name)
        ax.set_xlabel("PCA Dimension 1")
        ax.set_ylabel("PCA Dimension 2")
        ax.set_zlabel("PCA Dimension 3")
        ax.legend()
    
    else:
        raise ValueError("n_dim must be between 1, 2, or 3.")
    ax.set_title(f"PCA of {token1_name} and {token2_name} embeddings")
    
    return pca
