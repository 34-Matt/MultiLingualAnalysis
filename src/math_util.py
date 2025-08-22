import numpy as np
from typing import List, Tuple

def remove_projection(X: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Remove the information from the data along a specified direction.

    Args:
        X (np.ndarray): The data to remove the projection from.
        direction (np.ndarray): The direction to remove the projection along.
    
    Returns:
        X_transformed (np.ndarray): The data with the projection removed.
    """
    u = direction / np.linalg.norm(direction)
    projection = np.dot(X, u[:, np.newaxis]) * u
    return X - projection

def project_to_lower_dim(X: np.ndarray, directions: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Project the data onto a lower-dimensional space defined by the given directions.

    Args:
        X (np.ndarray): The data to project.
        directions (List[np.ndarray]): A list of direction vectors defining the lower-dimensional space.

    Returns:
        X_transformed (np.ndarray): The projected data.
        U (np.ndarray): The matrix of direction vectors used for projection.
    """
    # Ensure directions are unit vectors
    U = None
    for direction in directions:
        if U is None:
            U = [direction / np.linalg.norm(direction)]
        else:
            for u in U:
                direction -= np.dot(direction, u) * u
            U.append(direction / np.linalg.norm(direction))

    U = np.stack(U, axis=1)

    # Project the data onto the lower-dimensional space
    return np.dot(X, U), U
    

def my_pca(X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Finds the principal components of the data.

    Args:
        X (np.ndarray): The data to perform PCA on.
        n_components (int): The number of principal components to return.

    Returns:
        X_transformed (np.ndarray): The transformed data.
        U (np.ndarray): The matrix of principal components.
    """
    # Validate input
    if n_components <= 0:
        raise ValueError("n_components must be a positive integer.")
    elif n_components > X.shape[1]:
        raise ValueError("n_components must be less than or equal to the number of features.")
    
    original_X = X.copy()
    
    # Compute first mean vector
    mean = [np.mean(X, axis=0)]

    # Compute remaining mean vectors
    for _ in range(n_components - 1):
        X = remove_projection(X, mean[-1])
        mean.append(np.mean(X, axis=0))
    
    # Project the data onto the lower-dimensional space defined by the mean vectors
    return project_to_lower_dim(original_X, mean)