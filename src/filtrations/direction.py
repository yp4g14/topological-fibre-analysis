import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt,cos,sin
from scipy import ndimage

from ph import persistent_homology_cubical

def directional_levelsets(
    binary_image,
    direction,
    step_size,
    max_steps=False):
    """takes in a binary image and a direction vector,
    returns an image replacing binary image 1s with distance 
    or step number in that direction, and 0s with np.inf

    Args:
        binary_image (np array): np array of 0s and 1s shape (n,m)
        direction (tuple): tuple of float direction (dx, dy)
        step_size (integer): size of directional steps in pixels
        max_steps (bool, optional): If True, also returns the maximum number of steps. Defaults to False.

    Returns:
        np array: directional stepped version of binary image
    """
    x = np.arange(0,binary_image.shape[0],step=1)
    y = np.arange(0,binary_image.shape[1],step=1)
    xx, yy = np.meshgrid(x,y)
    direction_cts = direction[0]*xx + direction[1]*yy
    # want the minimum in the directional image to be one
    shift = abs(np.min(direction_cts))+1
    direction_cts = direction_cts+shift
    direction_steps = direction_cts//step_size
    direction_steps+=1
    direction_image = direction_steps*binary_image
    max_num_steps = np.max(direction_steps)
    direction_image = np.where(direction_image==0.,np.inf,direction_image)
    if max_steps:
        return direction_image, max_num_steps
    else:
        return direction_image

def direction_ph(
    save_path,
    binary_image,
    name,
    direction_s1,
    step_size=None,
    n_steps=20,
    homology_dims=[0,1]
    ):
    """Direction cubical persistent homology on 2D binary image.

    Args:
        save_path (string): string location for saving
        binary_image (np array): numpy array binary image shape (n,m)
        name (string): string filename for saving
        direction_s1 (tuple): tuple direction vector on s1
        step_size (integer, optional): Number of pixels to step over in direction per levelset step. Defaults to None.
        n_steps (int, optional): Number of steps if step size not given. Defaults to 20.
        homology_dims (list, optional): List of integers for homology dimensions. Defaults to [0,1].
    """
    dimension = len(binary_image.shape)
    assert dimension==2, f"image dimension {dimension} not equal to 2"
    config = {}
    config['save_path'] = save_path
    config['name'] = name
    config['direction'] = direction_s1
    config['homology_dims'] = homology_dims
   
    # get step_size to take in image from n_steps
    d = sorted(binary_image.shape)[-2:]
    max_imsize = round(sqrt(d[0]**2+d[1]**2))

    if not step_size:
        assert n_steps, "step_size and n_steps are both None"
        config['n_steps'] = n_steps
        step_size = int((max_imsize-0)/n_steps)
        assert n_steps <= np.min(binary_image.shape), "n_steps too large, step size cannot be less than a pixel/voxel"
    assert step_size < max_imsize, "step_size larger than total image size"
    config['step_size'] = step_size

    directional_image, max_steps_poss = directional_levelsets(
        binary_image,
        direction_s1,
        step_size,
        max_steps=True)

    save_name = f"direction_{name}_{'_'.join([str(round(d,ndigits=2)) for d in direction_s1])}"

    persistent_homology_cubical(
        directional_image,
        homology_dims,
        save_path,
        save_name)

   # save out run details
    with open(f"{save_path}config_{name[:-4]}.txt", 'w') as f: 
        for key, value in config.items(): 
            f.write('%s:%s\n' % (key, value))


def convert_ctfire_degree_to_s1_coords_PHT(angle):
    """Degree to direction conversion

    Args:
        angle (int): integer in [0,180], converts to s1 vector

    Returns:
        tuple: change in x, change in y
    """
    from math import radians
    if angle == 0:
        dx, dy = -1., 0.
    elif angle == 90:
        dx, dy = 0., 1.
    elif angle == 180:
        dx, dy = 1., 0.
    elif angle < 90:
        angle_rad = radians(angle)
        dx,dy = -cos(angle_rad), sin(angle_rad)
    elif angle > 90:
        minor_angle_rad = radians(180-angle)
        dx, dy = cos(minor_angle_rad), sin(minor_angle_rad)
    return [dx, dy]


if __name__ =="__main__":
    import os
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from pathlib import Path

    #  run example
    path = str(Path(os.getcwd()).parent) +'\\example\\'
    save_path = path+'direction\\'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    filenames = [name for name in os.listdir(path) if name.split('.')[-1]=='npy']
    # directions between 0 and 180, as in ctfire output, 
    # will be converted to s1 coords direction
    directions = dict(list(zip(filenames,[150,150])))
    
    for name in filenames:
        name= filenames[0]
        binary_image = np.load(path+name)
        direction_s1 = convert_ctfire_degree_to_s1_coords_PHT(directions[name])
        direction_ph(
            save_path,
            binary_image,
            name,
            direction_s1,
            step_size=1,
            homology_dims=[0,1])
