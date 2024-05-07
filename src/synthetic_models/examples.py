import os
import numpy as np
from skimage import io
from math import tan, sin, cos, pi, sqrt
from matplotlib import pyplot as plt

def lines(shape, angle, distance, thickness):
    # set up blank array
    image = np.zeros((shape,shape))
    # set up coords
    x = np.linspace(0, shape, shape)
    y = np.linspace(0, shape, shape)
    xx, yy = np.meshgrid(x, y)
    # line gradient
    if (angle/pi == 0.5):
        x_intercepts = np.arange(0, shape, step=distance)
        for x_intercept in x_intercepts:
            l = ((xx>=x_intercept) & (xx<=x_intercept+thickness)).astype(int)
            image+=l
    elif (angle/pi== 0.) or (angle/pi==1.):
        y_intercepts = np.arange(0, shape, step=distance)
        for y_intercept in y_intercepts:
            l = ((yy>=y_intercept) & (yy<=y_intercept+thickness)).astype(int)
            image+=l
    elif angle/pi<0.5:
        gradient = -tan(angle)
        # work out constant step up y_axis so lines are distance apart
        c_step = abs(distance/sin(pi/2-angle))
        thickness_intercept = abs(thickness/sin(pi/2-angle))
        c_max = shape+shape*abs(gradient)
        intercepts = np.arange(0,c_max,step=c_step)
        intercepts_neg = np.arange(0,-c_max,step=-c_step)
        intercepts = np.unique(np.concatenate([intercepts_neg,intercepts]))
        # for each y intercept value (c) add a line to the image
        for c in intercepts:
            l = ((yy>=gradient*xx+c) & (yy<=gradient*xx+thickness_intercept+c)).astype(int)
            image+=l
    elif (angle/pi>0.5) and (angle/pi<1.0):
        gradient = -tan(angle)
        # work out constant step up y_axis so lines are distance apart
        c_step = abs(distance/sin(pi/2-angle))
        thickness_intercept = abs(thickness/sin(pi/2-angle))
        c_max = shape+shape*gradient
        intercepts = np.arange(0,c_max,step=c_step)
        intercepts_neg = np.arange(0,-c_max,step=-c_step)
        intercepts = np.unique(np.concatenate([intercepts_neg,intercepts]))
        # for each y intercept value (c) add a line to the image
        for c in intercepts:
            l = ((yy>=gradient*xx+c) & (yy<=gradient*xx+thickness_intercept+c)).astype(int)
            image+=l
    return image    

def single_spoke(
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
    if (angle/pi == 0.5) or (angle/pi==1.5):
        x_intercept = start_x
        if (angle/pi == 0.5):
            l = ((xx<=x_intercept) & (xx>=x_intercept-thickness)).astype(int)
        else:
            l = ((xx>=x_intercept) & (xx<=x_intercept+thickness)).astype(int)
        if start_y >= shape/2:
            l = l & (yy>=start_y) & (yy<=start_y+length)
        else:
            l = l & (yy<=start_y) & (yy>=start_y-length)
        image+=l
    elif (angle/pi == 0.) or (angle/pi==1.):
        y_intercept = start_y
        if (angle/pi == 0.):
            l = ((yy>=y_intercept) & (yy<=y_intercept+thickness)).astype(int)
        else:
            l = ((yy<=y_intercept) & (yy>=y_intercept-thickness)).astype(int)
        if start_x >= shape/2:
            l = l & (xx>=start_x) & (xx<=start_x+length)
        else:
            l = l & (xx<=start_x) & (xx>=start_x-length)
        image+=l
    else:
        gradient = tan(angle)
        intercept = start_y - (gradient*start_x)
        # work out constant step up y_axis so lines are distance apart
        thickness_intercept = thickness/sin(pi/2-angle)
        if thickness_intercept > 0:
            l = ((yy>=gradient*xx+intercept) & (yy<=gradient*xx+intercept+thickness_intercept)).astype(int)
        else:
            l = ((yy<=gradient*xx+intercept) & (yy>=gradient*xx+intercept+thickness_intercept)).astype(int)
        # line ends
        gradient_perp = tan(pi/2+angle)
        intercept_perp = start_y - (gradient_perp*start_x)
        if (start_x >= shape/2):
            end_x = start_x+np.abs(length*cos(angle))
        else:
            end_x = start_x-np.abs(length*(cos(angle)))
        if (start_y >= shape/2):
            end_y = start_y+np.abs(length*sin(angle))
        else:
            end_y = start_y-np.abs(length*sin(angle))
        intercept_perp_end = end_y - (gradient_perp*end_x)
        if angle/pi<1.:
            l_perp = ((yy>=gradient_perp*xx+intercept_perp)).astype(int)
            l_perp_end = ((yy<=gradient_perp*xx+intercept_perp_end)).astype(int)
        else:
            l_perp = ((yy<=gradient_perp*xx+intercept_perp)).astype(int)
            l_perp_end = ((yy>=gradient_perp*xx+intercept_perp_end)).astype(int)
        image+= l*l_perp*l_perp_end
    return image

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

def random_length_lines(
        shape,
        angle,
        thickness=10,
        number_of_lines=20
        ):
    max_len = round(sqrt(2*(shape**2)))
    start_points = np.random.randint(0,shape,(number_of_lines,2))
    lengths = np.random.randint(10,max_len,(number_of_lines))
    lines_to_plot = np.c_[start_points,lengths]
    images = []
    for start_x, start_y, length in lines_to_plot:
        images.append(single_line(
            shape,
            angle,
            thickness,
            (start_x,start_y),
            length))
    image = (np.sum(images,axis=0)>=1).astype(int)
    return image

def random_angles_lines(
        shape,
        angle,
        thickness=10,
        number_of_lines=20
        ):
    max_len = round(sqrt(2*(shape**2)))
    start_points = np.random.randint(0,shape,(number_of_lines,2))
    lengths = np.random.randint(10,max_len,(number_of_lines))
    line_angles = angle + (np.random.random(number_of_lines)*(pi/3) -(pi/6))
    lines_to_plot = np.c_[start_points,lengths,line_angles]
    images = []
    for start_x, start_y, length, line_angle in lines_to_plot:
        images.append(single_line(
            shape,
            line_angle,
            thickness,
            (start_x,start_y),
            length))
    image = (np.sum(images,axis=0)>=1).astype(int)
    return image


def grids(
        shape,
        angle1,
        distance1,
        angle2,
        distance2,
        thickness):
    image = lines(
        shape,
        angle1,
        distance1,
        thickness)
    image += lines(
        shape,
        angle2,
        distance2,
        thickness)
    image = (image>=1).astype(int)
    return image

def circles(shape, distance, thickness, min_radius=None):
    image = np.zeros((shape,shape))
    x = np.linspace(0, shape, shape)
    y = np.linspace(0, shape, shape)
    xx, yy = np.meshgrid(x, y)

    centre_coord = shape/2

    if not min_radius:
        min_radius = distance
    radii = np.arange(min_radius,shape,step=distance)

    for radius in radii:
        circle = (
            (np.sqrt((xx-centre_coord)**2 + (yy-centre_coord)**2) >=radius) 
            & (np.sqrt((xx-centre_coord)**2 + (yy-centre_coord)**2) <=radius+thickness)
            ).astype(int)
        image+=circle
    return image

def almost_loop(
        shape,
        radius,
        thickness=5,
        fraction=.25):
    image = np.zeros((shape,shape))
    x = np.linspace(0, shape, shape)
    y = np.linspace(0, shape, shape)
    xx, yy = np.meshgrid(x, y)

    centre_coord = shape/2

    circle = (
        (np.sqrt((xx-centre_coord)**2 + (yy-centre_coord)**2) >=radius) 
        & (np.sqrt((xx-centre_coord)**2 + (yy-centre_coord)**2) <=radius+thickness)
        ).astype(int)
    image+=circle
    takeout = int(shape*fraction)
    image[:takeout,:takeout] = 0    
    return image


def wheel(shape,thickness,num_lines,start_r,length):
    radius = 0.3*shape
    centre = shape/2
    x = np.linspace(0,shape,shape)
    y = np.linspace(0,shape,shape)
    xx, yy = np.meshgrid(x,y)
    circle = (((xx-centre)**2+(yy-centre)**2) <= radius**2) & (((xx-centre)**2+(yy-centre)**2) >= (radius-thickness)**2)
    line_angles = np.arange(0, 2*pi, step=2*pi/num_lines)
    start_points = np.array([[start_r*cos(theta)+centre, start_r*sin(theta)+centre] for theta in line_angles])
    line_images = [] 
    for line in range(num_lines):
        line_images.append(single_spoke(
                shape,
                angle=line_angles[line],
                thickness=thickness,
                start_point=start_points[line],
                length=length,
                ))

    lines = np.sum(line_images,axis=0)
    lines += circle
    lines = (lines > 0).astype(int)
    return lines

if __name__ == "__main__":
    from itertools import combinations
    from pathlib import Path

    #  run example
    save_path = str(Path(os.getcwd()).parent) +'\\example_set\\basic\\'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    shape = 500
    thickness = 10
    num_lines = 16
    distance=50
    ops = [(50,100,'in'),(148,100,'out'),(50,200,'both')]
    for start_radius, leng, name in ops:
        image = wheel(shape,thickness,num_lines,start_r=start_radius,length=leng)
        np.save(save_path+'wheel_'+name, image)
        io.imsave(save_path+'wheel_'+name+'.tif',image,plugin='tifffile')
        plt.subplots()
        plt.imshow(image, cmap='binary_r')
        plt.savefig(f"{save_path}wheel_{name}.svg", bbox_inches='tight')
        plt.close()

    #  generate lines images:
    angles = np.arange(0,pi+pi/6,step=pi/6)
    for angle in angles:
        image = lines(shape,angle,distance,thickness)
        name = f"lines_{shape}_{distance}_{thickness}_0p{str(round(angle/pi,5))[2:]}pi.npy"
        np.save(save_path+name, image)
        io.imsave(save_path+name[:-4]+'.tif',image,plugin='tifffile')
        plt.subplots()
        plt.imshow(image, cmap='binary_r')
        plt.savefig(save_path+name[:-3]+'svg', bbox_inches='tight')
        plt.close()

    # generate grids images
    angles_for_pairs = np.arange(0,4*pi/6,step=pi/6)
    angle_pairs = list(combinations(angles_for_pairs,2))
    reversed = [(v,u) for u,v in angle_pairs]
    angle_pairs = np.concatenate([angle_pairs, reversed])
    distance1, distance2 = distance, distance
    for angle1,angle2 in angle_pairs:
        image = grids(shape,angle1,distance1,angle2,distance2,thickness)
        name = f"grids_{shape}_{distance}_{thickness}_0p{str(round(angle1/pi,5))[2:]}pi_0p{str(round(angle2/pi,5))[2:]}pi.npy"
        np.save(save_path+name, image)
        plt.subplots()
        plt.imshow(image, cmap='binary_r')
        plt.savefig(save_path+name[:-3]+'svg', bbox_inches='tight')
        plt.close()

    # generate circles
    # radii = np.arange(50,350,step=50)
    radii = [100,200,250,300]
    for min_radius in radii:
        image = circles(shape, distance, thickness, min_radius=min_radius)
        name = f"circles_{shape}_{distance}_{thickness}_{min_radius}.npy"
        np.save(save_path+name, image)
        plt.subplots()
        plt.imshow(image, cmap='binary_r')
        plt.savefig(save_path+name[:-3]+'svg', bbox_inches='tight')
        plt.close()

    # generate random lines
    number_of_lines = 20
    angles = np.arange(pi/6,3*pi/6,step=pi/6)
    for angle in angles:
        image = random_length_lines(
            shape,
            angle,
            thickness,
            number_of_lines)
        name = f"lines_length_{shape}_{thickness}_{number_of_lines}_0p{str(round(angle/pi,5))[2:]}pi.npy"
        np.save(save_path+name, image)
        plt.subplots()
        plt.imshow(image, cmap='binary_r')
        plt.savefig(save_path+name[:-3]+'svg', bbox_inches='tight')
        plt.close()

    # angles = np.arange(pi/6,3*pi/6,step=pi/6)
    shape=500
    thickness=10
    number_of_lines=20
    angles = [pi/6, 2*pi/6]
    for angle in angles:
        image = random_angles_lines(
        shape,
        angle,
        thickness,
        number_of_lines
        )
        name = f"lines_angles_{shape}_{thickness}_{number_of_lines}_0p{str(round(angle/pi,5))[2:]}pi.npy"
        np.save(save_path+name, image)
        plt.subplots()
        plt.imshow(image, cmap='binary_r')
        plt.savefig(save_path+name[:-3]+'svg', bbox_inches='tight')
        plt.close()

    # approx circle
    shape = 20
    radius = 10
    thickness = 2
    for frac in [.25, .4, .6]:
        image = almost_loop(
            shape,
            radius,
            thickness,
            frac)
        np.save(f"{save_path}almost_loop_{shape}_{radius}_{thickness}_{frac}.npy", image)
        plt.imshow(image,cmap='binary')
        plt.savefig(
            f"{save_path}almost_loop_{shape}_{radius}_{thickness}_{frac}.svg",
            bbox_inches='tight')
        plt.close()
