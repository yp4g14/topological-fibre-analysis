from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
import matplotlib
import psutil

def quad(b,d):
    """Return quadrant a point belongs to from 1,2,3,4.

    Args:
        b (float): x axis
        d (float): y axis

    Returns:
        int: quadrant in 1,2,3,4 
    """
    if b < 0:
        if d >= 0:
            q=2
        else:
            q=3
    elif b >= 0:
        if d >= 0:
            q=1
        else:
            q=4
    return q

def calculate_minimums(df):
    """Calculate the minimum birth and death value
     grouping by name, H dim, and quadrant.

    Args:
        df (pandas DataFrame): dataframe containing persistence diagrams 
        with name column id per diagram, and birth/death columns 'b'/'d'

    Returns:
        DataFrame: minimum birth and death per name H and quadrant keys.
    """
    df['quadrant'] = df.apply(lambda x: quad(x['b'],x['d']), axis=1)
    name = df['name'].drop_duplicates()[0]
    min_df = df.groupby(['name', 'H','quadrant'])[['b','d']].min().reset_index()
    min_df = min_df.rename({'b':'min_b','d':'min_d'},axis=1)
    return min_df

def shift_pd_slicemin_save(
        df,
        shift_path,
        name):
    """Align persistence diagram.

    Args:
        df (pandas dataframe): persistence diagram with columns 'b','d','H'.
        shift_path (string): path location to save shifted diagram
        name (string): filename to save as

    Returns:
        pandas dataframe: shifted persistence diagram
    """
    # per homology dimension, take the absolute of the minimums
    birth_shift = df.groupby('H')['b'].min().abs().to_dict()
    # make a column with that shift value per homology dimension
    df['shift'] = df['H'].apply(lambda x: birth_shift[x])
    # set that shift value to 0 for H1 Quadrant 1.
    df.loc[(df['quadrant']==1) & (df['H']==1),'shift'] = 0
    # align the birth values
    df['b_aligned'] = df['shift']+df['b']
    # align the death values for Quadrant 3
    df['d_q3_aligned'] = df[df['quadrant']==3].apply(lambda x: x['d']+x['shift'], axis=1)
    df.to_csv(f"{shift_path}aligned_{name[:-4]}.csv",index=False)
    return df

def acf(
        H_acf,
        name,
        col,
        quadrants,
        save_path,
        save_string,
        step):
    """Standardise data into a curve by interpolating with values step apart.
    Sort and count instances of birth (death)
    Standardise the birth (death) values to be step apart 
    and fill in their estimated count by forming a line 
    between the next and last real value. 
    Where the birth (death) has (one or more) entries of the standard value, 
    take the max index for that standard value.

    Args:
        H_acf (pandas dataframe): data
        name (string): data filename
        col (string): name of H_acf column to standardise
        quadrants (list): list of quadrants integers the data is from
        save_path (string): path to save acfs to
        save_string (string): string to save standardised data as
        step (float): standardised points will be this far apart
    """
    
    # remove infinity values in column if they exist
    H_acf = H_acf[H_acf[col]<np.inf].reset_index(drop=True)
    # only process if still data in DataFrame
    if H_acf.shape[0]>=1:
        # max and min vals
        min_val, max_val = H_acf[col].min(), H_acf[col].max()
        # create standard steps 
        steps = np.arange(int(min_val/step)*step, max_val+step,step=step)
        steps = steps[steps>=min_val]
        step_df = pd.DataFrame(steps,columns=['step'])
        # label as standard points
        step_df['standard'] = True
        # fill in H and name columns as they are single value id cols
        if H_acf['H'].nunique() == 1:
            step_df['H']=H_acf['H'].drop_duplicates().to_list()[0]
        step_df['name']=name
        # set index of standard points to nan
        step_df['index'] = np.nan

        # sort data byh column value and reset the index
        H_acf = H_acf.sort_values(col).reset_index(drop=True)
        # reset the new index as a column, this is the sorted order of real data
        H_acf = H_acf.reset_index()
        H_acf['index']+=1
        # join the standard steps and the real data
        H_acf['step'] = H_acf[col]
        H_acf = H_acf[list(step_df.columns)+[col]]
        H_acf = pd.concat([H_acf,step_df])
        # sort the joined data so the standard points are between the real points
        H_acf = H_acf.sort_values('step').reset_index(drop=True)
        # line equation between last real and next real points
        # to interpolate value for new standard point
        H_acf['y1'] = H_acf['index'].ffill()
        H_acf['y2'] = H_acf['index'].bfill()
        H_acf['x1'] = H_acf[col].ffill()
        H_acf['x2'] = H_acf[col].bfill()
        H_acf['m'] =  (H_acf['y2']-H_acf['y1'])/(H_acf['x2']-H_acf['x1'])
        H_acf['c'] = H_acf['y1'] - H_acf['m']*H_acf['x1']
        H_acf['acf'] = H_acf['m']*H_acf['step'] + H_acf['c']
        # for a given step, if the last real and next real point have same col value need to replace
        # with max index for that step
        replace_steps = H_acf[H_acf['standard'] & (H_acf['x1']==H_acf['x2'])]['step'].to_list()
        replacement = H_acf.groupby('step')['index'].max().reset_index()
        vals= dict.fromkeys(replace_steps)
        for s in replace_steps:
            vals[s] = replacement[replacement['step']==s]['index'].to_list()[0]
        H_acf.loc[H_acf['standard'] & (H_acf['x1']==H_acf['x2']),'acf'] = list(vals.values())

        # format and save
        cols = ['H', 'name', 'step','acf']
        standard = H_acf[H_acf['standard']][cols]
        standard['quadrants'] = [quadrants]*standard.shape[0]
        standard = standard.reset_index(drop=True)
        standard['acf_diff'] = standard['acf'].diff()
        spline_df = standard[['step','acf']].dropna()
        steps = np.array(spline_df['step'])
        if steps.shape[0] > 3:
            acf = np.array(spline_df['acf'])
            m = steps.shape[0]
            smooth = (m+sqrt((2*m)))
            tck = interpolate.splrep(steps,acf,s=0.1*smooth)
            spline = interpolate.splev(steps,tck)
            spline_gradient = interpolate.splev(steps,tck,der=1)
            spline_df['spline'] = spline
            spline_df['spline_gradient'] = spline_gradient
            spline_df['smooth'] = 0.2*smooth
            spline_df = spline_df[['step','spline','spline_gradient','smooth']]
            standard = standard.merge(spline_df,on='step',how='left')
        standard.to_csv(f"{save_path}{save_string}.csv")


def standardise_points_per_slice(
        df,
        name,
        acf_path,
        step=1,
        inwards=True,
        acf_cols=['b','d']):
    """
    Take birth and death acf by Homology type (0,1) and quadrant combos.
    If inwards, will use the aligned persistence values.


    Args:
        df (pandas dataframe): persistence diagram with name, quadrant, rep details plus shift if aligned
        name (string): name of file for persistence diagram
        acf_path (string): path to save acf curves to
        step (int, optional): Interval for standard values. Defaults to 1.
        inwards (bool, optional): If boundary filtration was taken inwards. Defaults to True.
        acf_cols (list, optional): list of strings with the column names to take acfs of. Defaults to ['b','d'].
    """
    df['standard']=False
    H0 = df[df['H']==0]
    H1 = df[df['H']==1]

    for H_df,hom in [[H0,0],[H1,1]]:
        if H_df.shape[0]>0:
            quadrants = H_df['quadrant'].drop_duplicates().to_list()
            quad_str = f"Quadrant_{'n'.join([str(i) for i in quadrants])}"
            path_str = f"H{hom}_{quad_str}/"
            if not os.path.isdir(acf_path+path_str):
                os.mkdir(acf_path+path_str)
            for col in acf_cols:
                save_string = f"H{hom}_{col}_{quad_str}_{name[:-4]}"
                acf(
                    H_df,
                    name,
                    col,
                    quadrants,
                    acf_path+path_str,
                    save_string,
                    step)

            if len(quadrants)>1:
                for q in quadrants:
                    quad_str = f"H{hom}_Quadrant_{q}/"
                    if not os.path.isdir(acf_path+quad_str):
                        os.mkdir(acf_path+quad_str)
                    if inwards:
                        cols = ['b', 'd', 'H', 'name','quadrant','rep',
                                'shift', 'b_aligned', 'd_q3_aligned', 'standard']
                    else:
                        cols = ['b', 'd', 'H', 'name','quadrant','rep','standard']
                    quad_df = H_df[H_df['quadrant']==q][cols].reset_index(drop=True)
                    if quad_df.shape[0]>0:
                        for col in acf_cols:
                            save_string = f"H{hom}_{col}_Quadrant_{q}_{name[:-4]}"
                            acf(
                                quad_df,
                                name,
                                col,
                                [q],
                                acf_path+quad_str,
                                save_string,
                                step)

def process_pd(
        path,
        names,
        align=True,
        step=1):
    """Reads in persistence diagram.
    Standardises to get an acf curve.

    Args:
        path (string): location of persistence diagrams
        step (int, optional): Interval between standard curve points. Defaults to 1.
    """
    acf_path = path+'acf/'
    if not os.path.exists(acf_path):
        os.mkdir(acf_path)
    if align:
        shift_path = path+'shifted/'
        if not os.path.exists(shift_path):
            os.mkdir(shift_path)

    # shift each PD and save:
    for name in names:
        df = pd.read_csv(path+name, names=['b','d','H'],header=0)
        print("Warning omitting persistence less than sqrt 2")
        df = df[df['d']>df['b']+sqrt(2)]
        df.loc[:,'name'] = name
        df['quadrant'] = df.apply(lambda x: quad(x['b'],x['d']),axis=1)
        if align:
            df = shift_pd_slicemin_save(
                df,
                shift_path,
                name)
            acf_cols=['b_aligned','d_q3_aligned','d']
        else:
            acf_cols=['b','d']

            df = df.reset_index(drop=True)
            combos = np.array(df[['H','quadrant','name']].groupby(['H','quadrant']).count().reset_index())
            for hom,q,count in combos:
                hom,q=int(hom),int(q)
                for col in acf_cols:
                    save_string = f"H{hom}_{col}_Q{q}_{name[:-4]}"
                    acf(
                        df[(df['H']==hom)&(df['quadrant']==q)],
                        name,
                        col,
                        [q],
                        acf_path,
                        save_string,
                        step)
                

if __name__ == "__main__":
    import os
    from tqdm import tqdm
    import pandas as pd
    import numpy as np
    import seaborn as sns
    from pathlib import Path

    path = str(Path(os.getcwd()).parent) +'\\example\\'
    filt_path = path+'direction_thickening\\'
    save_path = path+'acf\\'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    name = [name for name in os.listdir(filt_path) if name.split('_')[0]=='ph' and name.split('.')[-1]=='csv'][0]
    process_pd(path=filt_path,names=[name],align=True,step=1)
    name_dict = {
        'b_aligned':'aligned birth',
        'd':'death',
        'd_q3_aligned':'aligned death Q3'}

    acf_path = filt_path+'acf//'
    names = [name for name in os.listdir(acf_path) if name.split('.')[-1]=='csv']
    for name in names:
        acf_df = pd.read_csv(acf_path+name,index_col=0)
        if acf_df.shape[0]>1:
            h = int(name[1])
            var = name_dict['_'.join(name.split('_Q')[0].split('_')[1:])]
            q = int(name.split('_Q')[1].split('_')[0])
            fig,ax=plt.subplots()
            sns.lineplot(x='step',y='acf', data=acf_df)
            plt.xlabel(var)
            plt.ylabel('ACF')
            plt.title(f"H{h} Quadrant {q}")
            plt.savefig(acf_path+name.split('.')[0]+'.png')
            plt.close()


