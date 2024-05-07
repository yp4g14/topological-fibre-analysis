import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from math import floor, ceil, sqrt
import pandas as pd
import seaborn as sns

from ph import persistent_homology_cubical

def thickening_filtration(binary_image):
    """Takes in a binary image, replace the 1s with 0s and
    the 0s with shortest Euclidean distance to a former 1.

    Args:
        binary_image (np array): np array binary image

    Returns:
        np array: distance values for background, original signal as 0s
    """
    inverse_image = np.logical_not(binary_image).astype(int)
    distance_image = ndimage.distance_transform_edt(inverse_image)
    return distance_image

def thickening_ph(
        binary_image,
        name,
        save_path,
        homology_dims=[0,1]):
    """Take a binary image, and compute the 
    cubical persistent homology on the Euclidean distance or
    thickening filtration

    Args:
        binary_image (np array): binary image as np array
        name (string): string filename for saving
        save_path (string): string location for saving
        homology_dims (list, optional): Dimensions for homology computation. Defaults to [0,1].
    """
    save_name = f"thickening_{name}"
    filtration_image = thickening_filtration(binary_image)
    persistent_homology_cubical(
        filtration_image,
        homology_dims,
        save_path,
        save_name)


if __name__ == "__main__":
    import os
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from pathlib import Path

    #  run example
    path = str(Path(os.getcwd()).parent) +'\\example\\'
    save_path = path+'thickening\\'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    filenames = [name for name in os.listdir(path) if name.split('.')[-1]=='npy']

    for name in tqdm(filenames):
        binary_image = np.load(path+name)
        thickening_ph(
            binary_image,
            name[:-4],
            save_path,
            homology_dims=[0,1])

