# This module creates a GIF of the particle filter evolution.

import io
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def animate(numSteps,objHistory,pctsHistory,obsHistory,estHistory,\
            xLimits,yLimits):
    """ Save a GIF of the particle filter. """
    def buffer_plot_and_get():
        """ Convert a matplotlib figure to an image. """
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        return Image.open(buf)

    fontSize = 14
    plt.rcParams.update({'font.size': fontSize})
    plt.rcParams["font.family"] = "serif"
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',\
              'tab:purple','tab:brown']
    mSize = 50 # Markersize
    outImgs = []
    
    print('Animating...')
    for idx in tqdm(range(numSteps)):
        fig, ax1 = plt.subplots(1,1, figsize=(5, 5), dpi = 600,\
                           constrained_layout = True)
        
        # Plot
        ax1.scatter(objHistory[idx,0], objHistory[idx,1],\
                    s=mSize, label = 'Object') # Object
        ax1.scatter(pctsHistory[idx,:,0], pctsHistory[idx,:,1], marker='o',
                    edgecolor = 'none', color=colors[3], s=10, alpha = .25,\
                    label = 'Particle') # Particles
        ax1.set_xlim(xLimits), ax1.set_ylim(yLimits)
        ax1.set_xticks([]), ax1.set_yticks([])
        
        # Inset
        ax2 = inset_axes(ax1, width="37.5%", height="37.5%", loc="lower right")
        ax2.scatter(objHistory[idx,0], objHistory[idx,1],\
                    s=mSize) # Object
        ax2.scatter(obsHistory[idx,0], obsHistory[idx,1], marker='o',\
                    edgecolor=colors[3], facecolors = 'none',\
                    s=mSize, label = 'Observed') # Observation
        ax2.scatter(estHistory[idx,0], estHistory[idx,1], marker='^',\
                    edgecolor='black', facecolors = 'none',\
                    s=mSize, label = 'Estimated') # Estimation
        ax2.set_xlim(xLimits), ax2.set_ylim(yLimits)    
        ax2.set_xticks([]), ax2.set_yticks([])
        
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2,
           framealpha=0.95, loc='upper left')
        
        ax1.set_xlabel('Feature 1')
        ax1.set_ylabel('Feature 2')
        plt.suptitle('Particle Filter')
        plt.close()
        outImgs.append(buffer_plot_and_get())
    
    outImgs[0].save("output/evolution.gif", save_all = True, duration = 100,\
                    append_images=outImgs[1:], loop = 1) # Save GIF