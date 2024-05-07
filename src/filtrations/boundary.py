import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from math import floor, ceil, sqrt
from gtda.homology import CubicalPersistence

def boundary_filtration(
        binary_image,
        centre,
        inwards=True):
    """Takes in a binary image.
    Returns an image where each pixel that was 1 is replaced with the distance to the centre of the image.
    If inwards multiplies be -1.

    Args:
        binary_image (np.array): np array binary image. 1s are data, 0s background
        inwards (bool, optional): centre in or centre out filtration. 
        Defaults to boundary in (True).
    Returns:
        np.array: distance to centre image for boundary in or centre out filtrations
        integer: maximum value for background pixels. If inwards max val is 1, else is max distance + 1
    """

    height, width = binary_image.shape
    centre_im = np.zeros([height,width])
    centre_im[centre[0]][centre[1]]=1.

    # turn the centre only image into a distance to centre image
    boundary_mask = ndimage.distance_transform_edt(np.logical_not(centre_im).astype(int))
    if inwards:
        boundary_mask = -1* boundary_mask
        max_val=1
    else:
        max_val = ceil(np.max(boundary_mask))
        boundary_mask = boundary_mask - max_val
    assert np.max(boundary_mask) <= 0., f"max value {np.max(boundary_mask)}>0"

    inverse_image = np.logical_not(binary_image).astype(int)
    filtration = (binary_image * boundary_mask) + inverse_image
    return filtration, max_val

def cubical_ph_boundary(
    binary_image,
    centre,
    name,
    save_path,
    inwards=True,
    plots=True,
    homology_dims=[0,1]):

    filtration_image, max_val = boundary_filtration(
        binary_image,
        centre,
        inwards)

    if inwards:
        prefix = 'boundary_filt_inwards_'
    else:
        prefix = 'boundary_filt_outwards_'

    if plots:
        cmap = plt.get_cmap('viridis')
        cmap.set_bad('white')
        fig,ax = plt.subplots()
        filt_im_plot = np.ma.masked_equal(filtration_image, max_val)
        im = ax.imshow(filt_im_plot, interpolation='nearest',cmap=cmap)
        ax.axis('off')
        plt.colorbar(im)
        plt.savefig(f"{save_path}{prefix}{name}.svg")
        plt.close()

    # persistence
    cub = CubicalPersistence(
        homology_dimensions=homology_dims,
        coeff=2,
        reduced_homology=False,
        infinity_values=np.inf,
        n_jobs=-1)
    cub.fit([filtration_image], y=None)
    Xt = cub.transform([filtration_image])
    df = pd.DataFrame(Xt[0], columns=['birth','death','H_k'])
    df = df[df['death']>df['birth']]
    df.to_csv(f"{save_path}ph_{name}.csv",index=False)
    if plots:
        cub.plot(Xt)
        plt.tight_layout()
        plt.savefig(f"{save_path}ph_{prefix}{name}.svg")
        plt.close()


if __name__ == "__main__":
    import os
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from pathlib import Path

    #  run example
    path = str(Path(os.getcwd()).parent) +'\\example\\'
    save_path = path+'boundary\\'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(save_path+'inwards/'):
        os.mkdir(save_path+'inwards/')
    if not os.path.exists(save_path+'outwards/'):
        os.mkdir(save_path+'outwards/')

    filenames = [name for name in os.listdir(path) if name.split('.')[-1]=='npy']

    centres = dict(list(zip(filenames,[(250,250),(250,250)])))
    # for each name run inwards (boundary-thickening) and outwards (radial-thickening)
    for name in tqdm(filenames):
        binary_image = np.load(path+name)
        centre = np.array(centres[name])
        cubical_ph_boundary(
            binary_image,
            centre,
            name[:-4],
            save_path+'outwards/',
            inwards=False,
            plots=True,
            homology_dims=[0,1])
        binary_image = np.load(path+name)
        cubical_ph_boundary(
            binary_image,
            centre,
            name[:-4],
            save_path+'inwards/',
            inwards=True,
            plots=True,
            homology_dims=[0,1])
