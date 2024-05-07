import numpy as np
import pandas as pd
from gtda.homology import CubicalPersistence
import matplotlib.pyplot as plt

def persistent_homology_cubical(
    filtration_image,
    homology_dims,
    save_path,
    save_name):
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
    df.to_csv(f"{save_path}ph_{save_name}.csv",index=False)
    cub.plot(Xt)
    plt.tight_layout()
    plt.savefig(f"{save_path}ph_{save_name}.svg")
    plt.close()

