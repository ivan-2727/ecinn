import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm
import numpy as np 
import const 
from copy import deepcopy
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

def toDimensionalPotential(x):
    return x / (96485/(8.314*298))
def toDimensionlessPotential(x):
    return x * (96485/(8.314*298))

def draw_comparison(epoch, predictions, experiments):
    fig,axs_all = plt.subplots(figsize=(24,9),nrows=1,ncols=len(predictions))
    colors = cm.viridis(np.linspace(0,1,len(predictions)))
    scan_rates = [10,20,50]
    for i in range(len(predictions)):
        ax = axs_all[i] if len(predictions) > 1 else axs_all
        predictions[i].plot(x='Potential',y='Flux',ax=ax,color='b',lw=3,alpha=0.7,label='PINN')
        experiments[i].pot_flux.plot(x=0,y=1,ax=ax,color='r',ls='--',lw=3,alpha=0.7,label='Experiment')
        ax.annotate("", xy=(2.5, -0.5), xytext=(10, -0.5),arrowprops=dict(facecolor='black', shrink=0.05))
        ax.set_xlabel(r'Potential,$\theta$')
        ax.set_ylabel(r'Flux, $J$')
        ax.set_title(f'$\\nu={scan_rates[i]:.2E}$V/s,$\sigma={experiments[i].sigma:.2E}$')
        sec_ax = ax.secondary_xaxis(-0.15,functions=(toDimensionalPotential,toDimensionlessPotential))
        sec_ax.set_xlabel(f'$E-E^0_f$, V')
        ax.legend(fontsize=15)
    fig.tight_layout()
    fig.savefig(f'images/{epoch}.png',dpi=250,bbox_inches='tight')

    plt.close(fig)