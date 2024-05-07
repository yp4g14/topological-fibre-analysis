import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable 
from math import sqrt, floor, ceil


def two_phase_filtration_plot(
        filtration_image,
        save_path,
        name,
        centre):
    """Plots the two phases of the filtration in side-by-side subplots.

    Args:
        filtration_image (np array): filtration valued image
        save_path (string): string location to save
        name (string): string filename for saving
        centre (tuple): integer tuple pixel location for radial distances
    """
    mpl.rcParams.update({'font.size': 17})
    neg = np.where(filtration_image<=0., filtration_image, 0.)
    pos = np.where(filtration_image>0., filtration_image, 0.)
    neg_ticks = [int(np.min(neg)), int(np.max(neg[neg<0.]))]
    pos_ticks = [int(np.min(pos)), int(np.max(pos[pos<np.inf]))]
    fig, ax = plt.subplots(nrows=1,ncols=2)
    for axes in ax:
        axes.set_xticks([])
        axes.set_yticks([])
        axes.xaxis.set_ticklabels([])
        axes.yaxis.set_ticklabels([])
    neg_im = ax[0].imshow(neg, cmap='magma', vmax=0.)
    ax[0].scatter(centre[1],centre[0],marker='o',c='b',s=50)
    pos_im = ax[1].imshow(pos, cmap='magma', vmin=0)
    divider_neg = make_axes_locatable(ax[0])
    divider_pos = make_axes_locatable(ax[1])
    cax_neg = divider_neg.append_axes("right", size="5%", pad=0.05)
    cax_pos = divider_pos.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(neg_im, cax=cax_neg, ticks=neg_ticks)
    plt.colorbar(pos_im, cax=cax_pos, ticks=pos_ticks)
    fig.tight_layout(pad=.6)
    plt.savefig(
        f"{save_path}/filtration_dual_{name}.svg",
        bbox_inches='tight')
    plt.close()


def plot_persistence_diagrams(
        df,
        save_path,
        name):
    weight = df.reset_index().groupby(['b','d','H'])[['index']].count().reset_index().rename(columns={'index':'count'})
    sizes = (80,300)
    mpl.rcParams.update({'font.size': 18})
    weight['p']=weight['d']-weight['b']
    weight = weight[weight['p']>sqrt(2)].reset_index(drop=True)
    # print('WARNING: ommitting persistence < squareroot(2)')
    min_birth = df['b'].min()
    if min_birth < 0.:
        diagmax = np.max([weight['b'].abs().max(), weight[weight['d']!=np.inf]['d'].abs().max()])+1
        posmax = np.max([weight['b'].max(), weight[weight['d']!=np.inf]['d'].max()])+20
        negmax = weight['b'].min()-1
        weight['d']= weight['d'].replace(np.inf, posmax)
        weight = weight.sort_values('H', ascending=False)
        fig,ax = plt.subplots()
        graph = sns.scatterplot(
            data=weight,
            x='b',
            y='d',
            hue='H',
            size='count',
            sizes=sizes,
            ax=ax)
        plt.legend(
            loc='upper left',
            frameon=False,
            markerfirst=False,
            bbox_to_anchor=(1.05, 1),
            borderaxespad=0)
        graph.axhline(posmax, linestyle='--', c='tab:gray')
        graph.axhline(0., linestyle='--', c='tab:gray')
        graph.axvline(0., linestyle='--', c='tab:gray')
        ax.set_ylim(negmax-1,posmax+2)
        ax.set_xlim(negmax-1,posmax+2)
        ax.fill_between([min_birth,0,posmax], [min_birth,0,posmax],[min_birth,min_birth,min_birth],color='tab:blue',alpha=0.2)
        ax.set_xticks([round(min_birth), 0, round(posmax)])
        ax.set_yticks([round(min_birth), 0, round(posmax)])
        # ax.set_xticks(np.insert(np.arange(0,posmax+2,step=np.round(posmax/2)).astype(int),0,min_birth))
        # ax.set_yticks(np.insert(np.arange(0,posmax+2,step=np.round(posmax/2)).astype(int),0,min_birth))
        plt.savefig(f"{save_path}{name}.svg",bbox_inches='tight')
        plt.close()
    elif all(weight['d']==np.inf):
        bmax = df['b'].max()
        weight['d'] = bmax+1
        fig,ax = plt.subplots()
        graph = sns.scatterplot(
            data=weight,
            x='b',
            y='d',
            hue='H',
            size='count',
            sizes=sizes,
            ax=ax)
        plt.legend(
            loc='lower right',
            frameon=False,
            markerfirst=False)
        graph.axhline(bmax+1, linestyle='--', c='tab:gray')
        ax.set_ylim(0,bmax+2)
        ax.set_xlim(0,bmax+1)
        ax.fill_between([0,bmax+1], [0,bmax+1],color='tab:blue',alpha=0.2)
        ax.set_xticks(np.arange(0,bmax+2,step=4).astype(int))
        ax.set_yticks([0,bmax])
        plt.savefig(f"{save_path}{name}.svg",bbox_inches='tight')
        plt.close()
    else:
        diagmax = np.max([weight['b'].max(), weight[weight['d']!=np.inf]['d'].max()])+1
        weight['d']= weight['d'].replace(np.inf, diagmax)
        fig,ax = plt.subplots()
        graph = sns.scatterplot(
            data=weight,
            x='b',
            y='d',
            hue='H',
            size='count',
            sizes=sizes,
            ax=ax)
        plt.legend(
            loc='lower right',
            frameon=False,
            markerfirst=False)
        graph.axhline(diagmax, linestyle='--', c='tab:gray')
        ax.set_ylim(0,diagmax+1)
        ax.set_xlim(0,diagmax)
        ax.fill_between([0,diagmax], [0,diagmax],color='tab:blue',alpha=0.2)
        ax.set_xticks(np.arange(0,diagmax+1,step=4).astype(int))
        ax.set_yticks(np.arange(0,diagmax+1,step=4).astype(int))
        plt.savefig(f"{save_path}{name}.svg",bbox_inches='tight')
        plt.close()

