import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import io
import itertools as it
from scipy import ndimage
from math import ceil, pi
from tqdm import tqdm
from math import sqrt

def quadratic_roots(a,b,c):
    discriminant = b**2-4*a*c 
    roots = []
    for id in range(discriminant.shape[0]):
        d=discriminant[id]
        if d < 0:
            print ("This equation has no real solution")
            print(a,b,c)
        elif d == 0:
            xa = (-b[id]+sqrt(b[id]**2-4*a[id]*c[id]))/2*a[id]
            roots.append([xa])
        else:
            xa = (-b[id]+sqrt((b[id]**2)-(4*(a[id]*c[id]))))/(2*a[id])
            xb = (-b[id]-sqrt((b[id]**2)-(4*(a[id]*c[id]))))/(2*a[id])
            roots.append([xa,xb])
    assert len(roots) == discriminant.shape[0], "incorrect output shape quadratic roots"
    return roots

def circle_arperp_intersect_points(x_1,y_1,x_2,y_2,r_1,r_2):
    assert (r_2 <= r_1).all(), "point (x2,y2) lies on or outside initial circle"
    # point halfway between (x1,y1) and (x2,y2)
    bisect_x, bisect_y = (x_1+x_2)/2,(y_1+y_2)/2

    # gradient of perpendicular line to the line passing through (x1,y1) (x2,y2)
    perp_grad = -1*(x_2-x_1)/(y_2-y_1)
    # intersect of perpendicular line, through bisect point
    perp_intersect = bisect_y - (perp_grad*bisect_x)

    # intersect perpendicular line with circle (x1,y1) lies on
    #  gives two points (x_alpha,y_alpha) and (x_beta,y_beta)
    d = (2*perp_intersect*perp_grad)**2 - 4*(1+perp_grad**2)*(perp_intersect**2-r_1**2)
    if (d<0).any():
        print(x_1[d<0],y_1[d<0],x_2[d<0],y_2[d<0],r_1[d<0],r_2[d<0])
        print(bisect_x[d<0],bisect_y[d<0],perp_grad[d<0],perp_intersect[d<0])
        print((1+perp_grad[d<0]**2),2*perp_intersect[d<0]*perp_grad[d<0],perp_intersect[d<0]**2-r_1**2)
    x_root = quadratic_roots(
        a=(1+perp_grad**2),
        b=2*perp_intersect*perp_grad,
        c=perp_intersect**2-r_1**2)

    x_alpha, x_beta = np.array(x_root).T
    y_alpha = perp_grad*x_alpha+perp_intersect
    y_beta = perp_grad*x_beta+perp_intersect
    # choose alpha or beta point farthest from bisect point 
    #  this will be third point on new circle
    d_alpha = ((bisect_x-x_alpha)**2+(bisect_y-y_alpha)**2)**.5
    d_beta = ((bisect_x-x_beta)**2+(bisect_y-y_beta)**2)**.5
    x_3 = np.where(d_alpha>d_beta,x_alpha,x_beta)
    y_3 = np.where(d_alpha>d_beta,y_alpha,y_beta)
    
    # find line perpendicular to line through (x1,y1) (x3,y3)
    perp_grad_2=-1*(x_1-x_3)/(y_1-y_3)
    perp_intersect_2 = (y_1+y_3)/2 - perp_grad_2*(x_1+x_3)/2

    # find centre of circle by intersecting 2 perp lines
    second_circle_cx = (perp_intersect-perp_intersect_2)/(perp_grad_2-perp_grad)
    second_circle_cy = perp_grad*second_circle_cx+perp_intersect
    # average two points original radii for new radius 
    # second_radius = ((x_1**2+y_1**2)**0.5+(x_2**2+y_2**2)**0.5)/2
    second_radius = ((x_1-second_circle_cx)**2 + (y_1-second_circle_cy)**2)**.5
    return second_circle_cx,second_circle_cy,second_radius, x_3, y_3

def rgrid_gen(
        shape,
        radius,
        centres=None):
    if not centres:
        if shape % 2 == 0:
            centres = [int(shape/2)-1, int(shape/2)]
        else:
            centres = [int(shape/2)]
        centres = list(it.product(centres,centres))

    image = np.zeros([shape,shape])
    for cx,cy in centres:
        image[cx,cy]=1

    image = np.logical_not(image)
    image = ndimage.distance_transform_edt(image)
    return image

def draw_fibre(
        thickness,
        xx,
        yy,
        theta_grid,
        c_x,
        c_y,
        new_radius,
        theta1,
        theta2,
        wrap=False):
    if not wrap:
        offset_circle = (xx-c_x)**2+(yy-c_y)**2 <= new_radius**2
        offset_circle = offset_circle & ((xx-c_x)**2+(yy-c_y)**2 >= (new_radius-thickness)**2)
        original_thetas = np.sort([theta1,theta2])
        theta_limits = (theta_grid >= original_thetas[0]) & (theta_grid <= original_thetas[1])
        fibre = offset_circle & theta_limits
    else:
        offset_circle = (xx-c_x)**2+(yy-c_y)**2 <= new_radius**2
        offset_circle = offset_circle & ((xx-c_x)**2+(yy-c_y)**2 >= (new_radius-thickness)**2)
        original_thetas = np.sort([theta1,theta2])
        assert (original_thetas[1]-(2*pi)) < original_thetas[0], "wrap completes circle"
        theta_limits = (theta_grid >= original_thetas[0]) & (theta_grid <= (2*pi))
        theta_limits_wrap = (theta_grid <= (original_thetas[1]-(2*pi)))
        theta_limits = (theta_limits).astype(int) + (theta_limits_wrap).astype(int)
        fibre = offset_circle & theta_limits
    return fibre

def intercept_func(m,theta,thickness,max_thickness):
    if m > 0:
        intercept = thickness/np.sin(theta)
    elif m < 0:
        intercept = thickness/np.sin(theta-(pi/2))
    elif m == 0:
        intercept = thickness
    else:
        intercept = np.nan
    if intercept > max_thickness:
        intercept = max_thickness
    if intercept < -max_thickness:
        intercept = -max_thickness
    return intercept

def generate_spokes(
        shape,
        radius,
        number_spokes,
        length_bounds,
        xx,
        yy,
        thickness,
        max_thickness):

    starting_thetas = 2*pi*np.random.random(
            size=number_spokes)
    starting_radius = radius * np.random.random(size=number_spokes)
    # lengths = (length_bounds[1]-length_bounds[0]) * np.random.random(size=number_spokes) + length_bounds[0]
    lengths = np.random.default_rng().exponential(
            scale=26.438318051575934,
            size=number_spokes)+20.028
    lengths = np.where(lengths<length_bounds[0], length_bounds[0],lengths)

    spoke_df = pd.DataFrame(
        np.vstack([starting_thetas,starting_radius,lengths]).T,
        columns=['theta1','r1','length'])
    spoke_df['poss_length'] = radius - spoke_df['r1']
    spoke_df['length'] = spoke_df.apply(lambda x: min(x['length'],x['poss_length']),axis=1)
    spoke_df['r2'] = spoke_df['r1']+spoke_df['length']
    spoke_df['x1'] = spoke_df.apply(lambda x: x['r1']*np.cos(x['theta1']), axis=1)
    spoke_df['y1'] = spoke_df.apply(lambda x: x['r1']*np.sin(x['theta1']), axis=1)
    spoke_df['x2'] = spoke_df.apply(lambda x: x['r2']*np.cos(x['theta1']), axis=1)
    spoke_df['y2'] = spoke_df.apply(lambda x: x['r2']*np.sin(x['theta1']), axis=1)
    spoke_df['theta2'] = spoke_df['theta1']
    spoke_df['m'] = (spoke_df['y2']-spoke_df['y1'])/(spoke_df['x2']-spoke_df['x1'])
    # spoke_df['intercept'] = spoke_df.apply(lambda x: intercept_func(x['m'],x['theta1'],thickness,max_thickness), axis=1)
    spoke_df['intercept'] = thickness
    spoke_df = spoke_df.reset_index(drop=True)
    spoke_df['ring'] = -1

    spoke_image = np.zeros([shape,shape])
    for i in range(number_spokes):
        terms = spoke_df.loc[i,:].to_dict()
        if terms['m']==0:
            # horizontal
            xsmall,xbig=np.sort([terms['x1'],terms['x2']])
            spoke = (xx>=xsmall) & (xx<=xbig) & (yy>=0) & (yy<=thickness)
        elif np.isnan(terms['m']):
            # vertical
            ysmall,ybig=np.sort([terms['y1'],terms['y2']])
            spoke = (yy>=ysmall) & (yy<=ybig) & (xx>=0) & (xx<=thickness)
        else:
            xsmall,xbig=np.sort([terms['x1'],terms['x2']])
            ysmall,ybig=np.sort([terms['y1'],terms['y2']])
            box = (yy >= ysmall) & (yy <= ybig) & (xx >= xsmall) & (xx <= xbig)
            if terms['intercept']<0:
                line = (yy <= terms['m']*xx) & (yy >= terms['m']*xx+terms['intercept'])
            elif terms['intercept']>0:
                line = (yy >= terms['m']*xx) & (yy <= terms['m']*xx+terms['intercept'])
            spoke = box & line
        spoke_image += spoke
    spoke_image = (spoke_image>=1)
    return spoke_df, spoke_image

def synth_image(
        save_path,
        sname,
        shape,
        radius,
        number_fibres=100,
        thickness=1,
        dist=[1,1,1,1],
        length_bounds=(2,20),
        spoke_percent=0.2,
        rad_dev=25):
    
    assert number_fibres > 0, "number_fibres must be positive"
    assert thickness >= 1, "fibre thickness must be >=1"
    assert len(length_bounds) == 2, "fibre length bounds must be 2 values"
    assert (np.array(length_bounds)>0).all(), "fibre length bounds must be > 0"
    assert len(dist) < radius, "radius less than number of rings in dist"
    str_dist = '_'.join([str(i) for i in dist])
    save_name = f"synthetic2D_{sname}_{number_fibres}_{str_dist}"

    number_spokes = int(spoke_percent*number_fibres)
    number_fibres = number_fibres - number_spokes
    fibres_per_ring = np.round(number_fibres*np.array(dist)/np.sum(dist)).astype(int)
    number_rings = len(dist)
    ring_bounds = [ceil(i) for i in np.arange(0,radius+radius/number_rings,step=radius/number_rings)]
    ring_bounds[0] = thickness
    ring_fibres = np.zeros([shape,shape])
    end_ring_bounds = ring_bounds[1:]
    end_ring_bounds.append(ring_bounds[-1])
    end_ring_bounds = list(np.array(end_ring_bounds)+radius/(2*number_rings))

    ring_df = []
    for i in range(number_rings):
        starting_radius = np.random.randint(
            low=ring_bounds[i],
            high=ring_bounds[i+1],
            size=fibres_per_ring[i])
        ending_radius = np.random.randint(
            low=np.where(starting_radius-rad_dev<thickness,thickness,starting_radius-rad_dev),
            high=starting_radius+rad_dev,
            size=fibres_per_ring[i])
        # ending radius always smaller so end point inside original circle
        ending_radius, starting_radius = np.sort(np.vstack([starting_radius,ending_radius]),axis=0)
        # starting_circumference = 2*pi*starting_radius
        smallest_circumference = 2*pi*ending_radius
        length_upper_bound = np.where(smallest_circumference>length_bounds[1], length_bounds[1], smallest_circumference)
        starting_theta = 2*pi*np.random.random(
            size=fibres_per_ring[i])
        gen_length = np.random.default_rng().exponential(
            scale=26.438318051575934,
            size=fibres_per_ring[i])+20.028
        gen_length = np.where(gen_length<length_bounds[0], length_bounds[0],gen_length)
        # checl what this does to length distribution on new run
        gen_length = np.where(gen_length>length_upper_bound, length_upper_bound, gen_length)
        # direction of point
        if np.random.random()>0.5:
            ending_theta = starting_theta + (gen_length / starting_radius)
        else:
            ending_theta = starting_theta - (gen_length / starting_radius)
        ending_theta[ending_theta>(2*pi)]

        starting_x = (starting_radius*np.cos(starting_theta))
        starting_y = (starting_radius*np.sin(starting_theta))
        ending_x = (ending_radius*np.cos(ending_theta))
        ending_y = (ending_radius*np.sin(ending_theta))
        df = pd.DataFrame(
            np.vstack([starting_x,starting_y,starting_theta,starting_radius,ending_theta,ending_radius,ending_x,ending_y,gen_length]).T,
            columns=['x1','y1','theta1','r1','theta2','r2','x2', 'y2','approx_l'])
        df['ring'] = i

        second_circle_cx,second_circle_cy,second_radius,x_3,y_3  = circle_arperp_intersect_points(
            np.array(df['x1']),
            np.array(df['y1']),
            np.array(df['x2']),
            np.array(df['y2']),
            np.array(df['r1']),
            np.array(df['r2']))
        df['new_centre_x'] =second_circle_cx
        df['new_centre_y'] =second_circle_cy
        df['new_radius'] = second_radius
        df['x3'] = x_3
        df['y3'] = y_3
        df['wrap'] = df['theta2']>2*pi
        ring_df.append(df)

    yy, xx = np.mgrid[int(-shape/2):round(shape/2), int(-shape/2):round(shape/2)]
    # theta_grid = np.arctan(yy/xx)+pi*(xx<0) + pi/2
    theta_grid = np.arctan2(yy,xx) + 2*pi*(yy<0)
    cols= ['new_centre_x', 'new_centre_y', 'new_radius','theta1','theta2','wrap']

    fibres_image = []
    for i in tqdm(range(number_rings)):
        # generate ring image
        ring_fibres = np.zeros([shape,shape])
        for fibre_no in range(fibres_per_ring[i]):
            c_x,c_y, new_radius, theta1,theta2,wrap = ring_df[i].loc[fibre_no,cols]
            fibre = draw_fibre(
                thickness,
                xx,
                yy,
                theta_grid,
                c_x,
                c_y,
                new_radius,
                theta1,
                theta2,
                wrap=wrap)
            ring_fibres += fibre
        fibres_image.append(ring_fibres)

    if number_spokes>0:
        spoke_df, spoke_image = generate_spokes(
            shape,
            radius,
            number_spokes,
            length_bounds,
            xx,
            yy,
            thickness,
            max_thickness=2*thickness)
    # save out fibre info
    all_fibres = pd.concat(ring_df)
    all_fibres.rename({'approx_l':'length'}, axis=1, inplace=True)
    if number_spokes>0:
        common_cols = [name for name in spoke_df.columns if name in all_fibres.columns]
        all_fibres = all_fibres.merge(spoke_df,on=common_cols,how='outer')
    all_fibres.to_csv(f"{save_path}fibre_info_{save_name}.csv")

    fibres_image = np.sum(fibres_image,axis=0)
    if number_spokes>0:
        fibres_image = ((fibres_image + spoke_image)>0).astype(int)
    np.save(f"{save_path}{save_name}_allrings.npy", fibres_image)

    fig,ax = plt.subplots()
    im = ax.imshow(fibres_image)
    plt.colorbar(im)
    fig.tight_layout()
    plt.savefig(f"{save_path}{save_name}_allrings.svg")
    plt.close()

def parse_config(config_path, config_name, save_path):
    assert save_path != config_path, "save path must not be config path"
    with open(config_path+config_name, 'r') as file:
        with open(save_path+config_name,'w') as outfile:
            config = {}
            for item in file:
                outfile.write(item)
                item = item.strip('\n') 
                if len(item.split(' ')) != 2:
                    print(f"No value for {item}")
                else:
                    key,val = item.split(' ')
                    config[key] = val
    types = [
        ('shape',int),
        ('number_fibres',int),
        ('min_length',int),
        ('max_length',int),
        ('width',float),
        ('rad_dev',float),
        ('spoke_percent',float),
        ('rep',int)
        ]
    keys = list(config.keys())
    for k, type in types:
        config[k] = type(config[k])
    if 'dist' in keys:
        config['dist'] = [float(i) for i in config['dist'].strip('[]').split(',')]
    if ('min_length' in keys) and ('max_length' in keys):
        config['length_bounds'] = [config['min_length'], config['max_length']]
    print(config)
    return config


if __name__ == "__main__":
    import os
    from pathlib import Path
    save_path = str(Path(os.getcwd()).parent) +'\\example_set\\synth_circular\\'
    if not os.path.exists(save_path):
        os.mkdir(save_path)


    config_path = os.getcwd()+'/'
    config_names = [
        'config_area_uniform.txt',
        'config_boundary_heavy.txt',
        'config_centre_heavy.txt']
    for conf in config_names:
        config = parse_config(config_path, conf, save_path)
        print(f"Started {conf}")
        i=0
        while i < config['rep']:
            try:
                synth_image(
                    save_path,
                    sname=f"{conf[:-4]}_rep_{i+1}",
                    shape=config['shape'],
                    radius=config['shape']/2,
                    number_fibres=config['number_fibres'],
                    thickness=config['width'],
                    dist=config['dist'],
                    length_bounds=config['length_bounds'],
                    spoke_percent=config['spoke_percent'],
                    rad_dev=config['rad_dev'])
                i+=1
            except:
                print('error')
        print(f"Completed {conf}")


