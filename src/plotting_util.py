import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

def line_segment(
        points_start: np.ndarray,
        points_end: np.ndarray,
        segment_count: int=10,
        ax: plt.Axes=None,
        cmap: plt.Colormap=plt.get_cmap("RdBu_r")) -> None:
    """Draw a line segment on the given axes.

    Args:
        points_start (np.ndarray): Starting points of the line segments.
        points_end (np.ndarray): Ending points of the line segments.
        segments (int, optional): The number of segments to divide the line into. Defaults to 10.
        ax (plt.Axes): The axes to draw the line segment on.
        cmap (plt.Colormap, optional): The colormap to use for the line segment. Defaults to "RdBu_r".
    """
    for i in range(len(points_start)):
        # Create line segments between the start and end points
        x = []
        for j in range(len(points_start[i])):
            x.append(np.linspace(points_start[i, j], points_end[i, j], segment_count))


        # Reshape the points for LineCollection
        points = np.array(x).T.reshape(-1, 1, len(x))
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Get the axes to plot on
        if ax is None:
            ax = plt.gca()
        
        # Create a LineCollection and color
        colors = np.linspace(0, 1, len(segments))[:, np.newaxis]
        lc = LineCollection(segments, cmap=cmap, norm=Normalize(0, 1))
        lc.set_array(colors.ravel())
        lc.set_linewidth(2)

        # Plot
        ax.add_collection(lc)

def print_sorted_indices(vec: np.ndarray, *, end_value:float=None, end_index:int=None) -> str:
    """Prints `index:value` for each element in descending order.
    
    Args:
        vec (np.ndarray): The vector to print
        end_value (float): The final value to
    
    Results:
        output (str): The string to print out
    """
    sorted_indices = np.argsort(vec)[::-1]

    if end_index is None:
        end_index = len(sorted_indices+1)
    if end_value is None:
        end_value = min(vec) - 1
    
    count = 0
    out = ''

    for idx in sorted_indices:
        if (vec[idx] < end_value) or (count > end_index):
            break
        
        out += f"\n{idx}:{vec[idx]}"
        count += 1
