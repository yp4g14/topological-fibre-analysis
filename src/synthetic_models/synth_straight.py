import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt, tan, cos, sin
import pandas as pd

def single_line(
        shape,
        angle,
        thickness,
        start_point,
        length,
        ):
    start_x, start_y = start_point
    
    # set up blank array
    image = np.zeros((shape,shape))
    # set up coords
    x = np.linspace(0, shape, shape)
    y = np.linspace(0, shape, shape)
    xx, yy = np.meshgrid(x, y)
    # line gradient
    if angle/pi == 0.5:
        l = ((xx>=0) & (xx<=0+thickness)).astype(int)
        image+=l
    else:
        gradient = tan(angle)
        intercept = start_y - (gradient*start_x)
        # work out constant step up y_axis so lines are distance apart
        thickness_intercept = thickness/sin(pi/2-angle)
        l = ((yy>=gradient*xx+intercept) & (yy<=gradient*xx+intercept+thickness_intercept)).astype(int)
        # line ends
        gradient_perp = tan(pi/2+angle)
        intercept_perp = start_y - (gradient_perp*start_x)
        end_x, end_y = start_x+(length*cos(angle)), start_y+(length*sin(angle))
        intercept_perp_end = end_y - (gradient_perp*end_x)
        l_perp = ((yy>=gradient_perp*xx+intercept_perp)).astype(int)
        l_perp_end = ((yy<=gradient_perp*xx+intercept_perp_end)).astype(int)
        image+= l*l_perp*l_perp_end
    return image

def synth_random_angles_lines(
        save_path,
        shape,
        main_angle,
        thickness=10,
        number_of_lines=20,
        divergence=pi/6,
        length_lower_bound=20,
        rep=0
        ):
    start_points = np.random.randint(0,shape,(number_of_lines,2))
    # max_len = round(sqrt(2*(shape**2)))
    # lengths = np.random.randint(10,max_len,(number_of_lines))
    lengths = np.random.default_rng().exponential(
            scale=26.438318051575934,
            size=number_of_lines)+20.028
    lengths = np.where(lengths<length_lower_bound, length_lower_bound,lengths)
    line_angles = main_angle + (np.random.random(number_of_lines)*(2*divergence) -(divergence))
    lines_to_plot = np.c_[start_points,lengths,line_angles]
    params = pd.DataFrame(
        lines_to_plot,
        columns=[
            'start_point_x',
            'start_point_y',
            'lengths',
            'line_angles'])
    main_str = "{0:.5f}".format(main_angle/pi)
    div_str = "{0:.5f}".format(divergence/pi)
    name = f"synth_lines_angles_{shape}_{thickness}_{number_of_lines}_{main_str}pi_divergence_{div_str}pi_rep_{i}"
    params.to_csv(save_path+name+'.csv')

    image = np.zeros([shape,shape])
    for start_x, start_y, length, line_angle in lines_to_plot:
        image += single_line(
            shape,
            line_angle,
            thickness,
            (start_x,start_y),
            length)
    image = (image>0).astype(int)

    np.save(f"{save_path}{name}.npy", image)
    plt.subplots()
    plt.imshow(image, cmap='binary_r')
    plt.axis('off')
    plt.savefig(
        save_path+name+'.svg',
        bbox_inches='tight')
    plt.close()
    tif_image = image*255
    tif_image = tif_image.astype(np.uint8)
    io.imsave(
        save_path+name+'.tif',
        tif_image,
        plugin='tifffile')


if __name__ == "__main__":
    import os
    from skimage import io
    from tqdm import tqdm
    from pathlib import Path

    save_path = str(Path(os.getcwd()).parent) +'\\example_set\\synth_straight\\'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    number_repeats = 2
    main_angle = pi/3
    "{0:.5f}".format(main_angle/pi)
    divergence_angles = [0, pi/6, pi/3]
    shape=100
    thickness=3
    number_of_lines=30

    for i in tqdm(range(number_repeats)):
        for divergence in divergence_angles:
            synth_random_angles_lines(
                save_path,
                shape,
                main_angle,
                thickness,
                number_of_lines,
                divergence,
                length_lower_bound=20,
                rep=i)

