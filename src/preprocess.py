# convert ctfire NOl image to binary image for filtration building
from skimage import io
import os
import numpy as np
import matplotlib.pyplot as plt

path = os.getcwd()+'\\example\\ctFIREout\\'
save_path = os.getcwd()+'\\example\\'
nol_name = "NOL_ctFIRE_lines_angles_500_10_20_0p16667pi.tif"

nol_image = io.imread(path+nol_name)
binary_image = (np.sum(nol_image, axis=2) < 3*255).astype(int)
np.save(f"{save_path}binary_{nol_name[:-4]}.npy",binary_image)

fig,ax = plt.subplots(1,2)
ax[0].imshow(nol_image)
ax[0].set_xlabel("NOL from ctFIRE")
ax[1].imshow(binary_image, cmap='binary_r')
ax[1].set_xlabel("binary version of NOL")

