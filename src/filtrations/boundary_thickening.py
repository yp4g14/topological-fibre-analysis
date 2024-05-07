import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import ndimage
import matplotlib.pyplot as plt
from gtda.homology import CubicalPersistence
from math import sqrt, floor, ceil
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.append(str(Path(os.getcwd()).parent) +'\\plotting\\')
from pd_plots import two_phase_filtration_plot


def boundary_thickening_PH(
    save_path,
    binary_image,
    name,
    centre,
    inwards=True,
    homology_dims=[0,1]
    ):
    """Takes in a path and a 2D binary image, 
    calculates the radial or boundary-thickening cubical persistent homology.
    Where distance of radial/boundary filtrations are determined by chosen centre point.
    Radial-thickening if inwards = False
    Boundary-thickening if inwards = True

    Args:
        save_path (string): location to save as string
        binary_image (np array): array of image shape (n,m)
        name (string): string of filename wihout extension
        centre (tuple): tuple of integer pixel location for centre point choice
        inwards (bool, optional): Boundary-thickening if inwards True, else radial-thickening. Defaults to True.
        homology_dims (list, optional): dimensions to compute homology for. Defaults to [0,1].
    """
    dimension = len(binary_image.shape)
    assert dimension == 2, f"image dimension {dimension} not equal 2"
    config = {}
    config['save_path'] = save_path
    config['name'] = name
    config['inwards'] = inwards
    config['homology_dims'] = homology_dims
    centre_str = f"point_{centre[0]}_{centre[1]}_"
    config['centre'] = f"({centre[0]},{centre[1]})"
    if inwards:
        save_name = 'boundary_thickening_inwards_'+centre_str+name
    else:
        save_name = 'boundary_thickening_outwards_'+centre_str+name

    boundary_image = negative_valued_boundary_point_filtration(
        binary_image,
        centre,
        inwards=inwards)
    inverse_image = np.logical_not(binary_image)
    thickening_image = ndimage.distance_transform_edt(inverse_image)
    filtration_image = np.where(boundary_image==1., 0, boundary_image)+thickening_image

    two_phase_filtration_plot(
            filtration_image,
            save_path,
            save_name,
            centre)

    persistent_homology_boundary_thickening(
        filtration_image,
        homology_dims,
        save_path,
        save_name)
    with open(f"{save_path}config_{name}.txt", 'w') as f: 
        for key, value in config.items(): 
            f.write('%s:%s\n' % (key, value))

def negative_valued_boundary_point_filtration(
        binary_image,
        centre,
        inwards=True):
    """Takes in a binary image and a `center' point for slice.
    Returns an image where each pixel that was 1 is replaced with the distance to the centre of the image.
    If inwards multiplies be -1.
    Then shifts to be negative, so max original pixel value is 0, and the background is 1.

    Args:
        binary_image (np.array): np array binary image. 1s are data, 0s background
        centre (tuple floats): (c_x,c_y) coords of slice centre.
        inwards (bool, optional): centre in or centre out filtration. 
        Defaults to boundary in (True).
    Returns:
        np.array: distance to centre image for boundary in or centre out filtrations
    """

    # create a blank binary image in the same shape with just the centre as 1s 
    height, width = binary_image.shape
    centre_im = np.zeros([height,width])
    centre_im[centre[0]][centre[1]]=1.

    # turn the centre only image into a distance to centre image
    boundary_mask = ndimage.distance_transform_edt(np.logical_not(centre_im).astype(int))
    if inwards:
        boundary_mask = -1* boundary_mask
    else:
        max_val = ceil(np.max(boundary_mask))
        boundary_mask = boundary_mask - max_val
    assert np.max(boundary_mask) <= 0., f"max value {np.max(boundary_mask)}>0"

    inverse_image = np.logical_not(binary_image).astype(int)
    filtration = (binary_image * boundary_mask) + inverse_image
    return filtration

def persistent_homology_boundary_thickening(
    filtration_image,
    homology_dims,
    save_path,
    name):
    """Persistent homology computation using giotto-tda
    cubical sublevel persistence with 2 coefficients.

    Args:
        filtration_image (np array): filtration image to take sublevelsets over
        homology_dims (list of int): list of integers of homology groups to compute
        save_path (string): string location for saving persistence
        name (string): string filenames for saving purposes
    """
    cub = CubicalPersistence(
        homology_dimensions=homology_dims,
        coeff=2,
        reduced_homology=False,
        infinity_values=np.inf,
        n_jobs=-1)
    cub.fit([filtration_image], y=None)
    Xt = cub.transform([filtration_image])
    df = pd.DataFrame(Xt[0], columns=['birth','death','H_k'])
    df = df[df['birth']<df['death']]
    df.to_csv(f"{save_path}ph_{name}.csv",index=False)
    cub.plot(Xt)
    plt.tight_layout()
    plt.savefig(f"{save_path}ph_{name}.svg")
    plt.close()


if __name__ =="__main__":
    import os
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from pathlib import Path

    #  run example
    path = str(Path(os.getcwd()).parent) +'\\example\\'
    save_path = path+'radial_boundary_thickening\\'

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(save_path+'inwards/'):
        os.mkdir(save_path+'inwards/')
    if not os.path.exists(save_path+'outwards/'):
        os.mkdir(save_path+'outwards/')
    # find files
    filenames = [name for name in os.listdir(path) if name.split('.')[-1]=='npy']
    # set centre point for radial distance transform
    centres = dict(list(zip(filenames,[(250,250),(250,250)])))
    # for each name run inwards (boundary-thickening) and outwards (radial-thickening)
    for name in tqdm(filenames):
        binary_image = np.load(path+name)
        centre = np.array(centres[name])
        boundary_thickening_PH(
            save_path+'inwards/',
            binary_image,
            name[:-4],
            centre,
            inwards=True,
            homology_dims=[0,1])
        boundary_thickening_PH(
            save_path+'outwards/',
            binary_image,
            name[:-4],
            centre,
            inwards=False,
            homology_dims=[0,1])

