import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt,cos,sin
from scipy import ndimage
import sys
from pathlib import Path
sys.path.append(str(Path(os.getcwd()).parent) +'\\plotting\\')
from pd_plots import two_phase_filtration_plot
import pd_plots
from importlib import reload
reload(pd_plots)
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

def direction_thickening_PH(
    save_path,
    binary_image,
    name,
    direction_s1,
    step_size=None,
    n_steps=20,
    homology_dims=[0,1,2]
    ):
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

    # for each direction generate a levelset image with nsteps
    directional_image, max_steps_poss = directional_levelsets(
        binary_image,
        direction_s1,
        step_size,
        max_steps=True)
    # make values negative in direction transform
    neg_shift = max_steps_poss+1
    negative_directional_image = directional_image - neg_shift
    # now transform background thickening
    inverse_image = np.logical_not(binary_image)
    thickening_image = ndimage.distance_transform_edt(inverse_image)
    filtration_image = np.where(negative_directional_image==np.inf, 0, negative_directional_image)+thickening_image
    plt.imshow(filtration_image)
    save_name = f"{name}_{'_'.join([str(round(d,ndigits=2)) for d in direction_s1])}"
    two_phase_filtration_plot(
            filtration_image,
            save_path,
            save_name)

    # run gtda cubical homology on sublevel sets for one direction, saving persistence diagrams
    persistent_homology_cubical(
        filtration_image,
        homology_dims,
        save_path,
        save_name)
   # save out run details
    with open(f"{save_path}config_{name[:-4]}.txt", 'w') as f: 
        for key, value in config.items(): 
            f.write('%s:%s\n' % (key, value))



def choose_best_angle_ctfire_hist(
        ang):
    """Takes in the path to the ctfire output location and the original filename.
    Returns the best angle from the angle histogram.

    Best angle is chosen by rounding to integer degree angles 
    and choosing angle with maximum frequency on histogram with 180 bins. 
    If max frequency value is not unique, 
    bins are iteratively widened around previous max values
    until there is a unique maximum.

    Args:
        ang (np.array): numpy array of angles

    Returns:
        integer: best angle to nearest degree 
    """
    ang[ang<0.5] = 180+ang[ang<0.5]
    bins = np.arange(0.5,181.5,step=1)
    freq_val, bin_val = np.histogram(ang, bins=bins)

    ind_best_angs = np.where(freq_val==max(freq_val))[0]
    number_best = len(ind_best_angs)
    if len(ind_best_angs)==1:
        best_angle = [bin_val[ind_best_angs[0]]+.5]
    else:
        step=0
        while number_best>1:
            step+=0.5
            best_nums = []
            for i in range(number_best):
                bin_min, bin_max = bin_val[ind_best_angs[i]]-step, bin_val[ind_best_angs[i]+1]+step 
                if bin_min < 0.5:
                    bin_min = 180+bin_min
                    isolated_bin = np.where(ang >= bin_min, ang, np.nan)
                    isolated_bin_lower = np.where(ang <= bin_max, ang, np.nan)
                    isolated_bin = np.concatenate((isolated_bin, isolated_bin_lower))
                else:
                    isolated_bin = np.where(ang >= bin_min, ang, np.nan)
                    isolated_bin = np.where(isolated_bin <= bin_max, isolated_bin, np.nan)
                best_nums.append(len(isolated_bin[~np.isnan(isolated_bin)]))
            best_nums = np.array(best_nums)
            ind_best_nums = np.where(best_nums==max(best_nums))[0]
            best_angle = ind_best_angs[ind_best_nums]+1
            # print(step, ind_best_angs, best_nums, best_angle)
            number_best = len(ind_best_nums)
    return best_angle[0]

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
    save_path = path+'direction_thickening\\'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    filenames = [name for name in os.listdir(path) if name.split('.')[-1]=='npy']
    # directions between 0 and 180, as in ctfire output, 
    # will be converted to s1 coords direction
    directions = dict(list(zip(filenames,[150,150])))
    
    for name in filenames:
        binary_image = np.load(path+name)
        direction_s1 = convert_ctfire_degree_to_s1_coords_PHT(directions[name])
        direction_thickening_PH(
            save_path,
            binary_image,
            name,
            direction_s1,
            step_size=1,
            homology_dims=[0,1])
