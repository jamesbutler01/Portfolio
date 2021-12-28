'''
    Helper plot function for main.py
'''

import matplotlib.pyplot as plt
import numpy as np
import scipy
import warnings; warnings.filterwarnings('ignore') 


def colors(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    import matplotlib as mpl
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

def plot1(averages, sems):
    color_range = [colors('green', 'blue', i/5) for i in range(5)]
    f, axes = plt.subplots(2,2)
    plt.tight_layout()
    neurons = 137, 91, 69, 42
    for avgs, sems, ax, neur in zip(averages[[neurons]], sems[[neurons]], axes.flatten(), neurons):
        for i, (avg, sem) in enumerate(zip(avgs, sems)):  # For each value
            ax.plot(time, avg, c=color_range[i], label=f'Value {i+1}')
            ax.fill_between(time, avg-sem, avg+sem, alpha=0.3, color=color_range[i])
        ax.set_xlim(0, 1000); 
        ax.text(990, ax.get_ylim()[1]-0.05, 'Neuron '+str(neur), ha='right', va='top' )
    axes[0,1].legend(loc='upper left',bbox_to_anchor=[1,1])
    axes[0,0].set_ylabel('Firing rate'); axes[1,0].set_ylabel('Firing rate'); 
    axes[1,0].set_xlabel('Time (ms) post cue onset'); axes[1,1].set_xlabel('Time (ms) post cue onset');
    
    plt.suptitle(f'Neuron response to value',y=1.1);

def plot2(coeffs):
    f, ax = plt.subplots(1)
    tp = 25
    ax.stem(range(len(coeffs[:,tp])), coeffs[:,tp], 'k')
    ax.set_ylabel('Coefficients'); ax.set_xlabel('Neurons')
    ax.set_title(f'Coefficients across neurons at {tp*10} ms')

def plot3(corrMatrix,tit=0):
    f,ax=plt.subplots(1)
    im=plt.imshow(corrMatrix)
    cbar = f.colorbar(im, ax=ax)
    cbar.set_label('Pearson r', rotation=270, fontsize='medium', x=0.5)
    ax.set_xticks(range(numTimepoints)[::25]);ax.set_yticks(range(numTimepoints)[::25])
    ax.set_xticklabels(time[::25]);ax.set_yticklabels(time[::25])
    ax.set_xlabel('Time (ms) post cue onset'); ax.set_ylabel('Time (ms) post cue onset');
    tits = 'Correlation matrix of coefficients over time', 'Significant portions of correlation matrix', 'Correlation between different trial types'
    if tit==2:
        ax.axvline(85,c='k',ls='--',lw=0.5);ax.axhline(85,c='k',ls='--',lw=0.5)
        ax.set_xlabel('Time (ms) (second cue additive)'); ax.set_ylabel('Time (ms)  (second cue subtractive)');
    plt.title(tits[tit])


def plot4(dotProducts):
    from mpl_toolkits.mplot3d import Axes3D
    plt.rcParams['figure.figsize'] = [10, 5]
    f=plt.figure()
    gs2 = plt.GridSpec(1,1)
    ax3d = f.add_subplot(111, projection='3d')
    start, stop = 0, 80
    for i in range(10):
        a, b = dotProducts[i]
        if i in [0, 5]:
            ax3d.plot(a[start:stop], b[start:stop], np.arange(start*10, stop*10, 10), ls=ls[i//5], color=f'C{i%5}', lw=2, zorder=1, label=f'{labs[i<5]}')
        else:
            ax3d.plot(a[start:stop], b[start:stop], np.arange(start*10, stop*10, 10), ls=ls[i//5], color=f'C{i%5}', lw=2, zorder=1)
    plt.legend(loc='upper right', bbox_to_anchor=[1, 1])
    ax3d.zaxis.set_rotate_label(False)
    ax3d.set_zlabel('Time post cue 1 (ms)', rotation=90)
    ax3d.set_xlabel('PC1')
    ax3d.set_ylabel('PC2')
    ax3d.set_xticks(np.arange(-2, 5, 2))
    ax3d.set_yticks(np.arange(-1.5, 2.51, 2))
    ax3d.set_yticklabels(np.arange(-1.5, 2.51), x=-0.5)
    ax3d.set_xlim(-2, 3.5)
    ax3d.set_ylim(-1.5, 2.5)
    ax3d.view_init(28, 50.138)
    plt.gca().patch.set_facecolor('white')
    ax3d.w_xaxis.set_pane_color((0.9, 0.9, 0.9, 1.0))
    ax3d.w_yaxis.set_pane_color((0.9, 0.9, 0.9, 1.0))
    ax3d.w_zaxis.set_pane_color((0.9, 0.9, 0.9, 1.0))
    # Label values
    ax3d.text(-1, 0.5, 800, '1', c='C0', weight='bold')
    ax3d.text(0, 0.95, 800, '2', c='C1', weight='bold')
    ax3d.text(0.75, -0.1, 800, '3', c='C2', weight='bold')
    ax3d.text(2, 0.5, 800, '4', c='C3', weight='bold')
    ax3d.text(4, 1.5, 850, '5', c='C4', weight='bold')
    # Join up pairs
    ttps = [800]; pnts =  [80]
    for t, p in zip(ttps, pnts):
        for i in range(5):
                ax3d.plot(dotProducts[[i, i+5], 0, p], dotProducts[[i, i+5], 1, p], [t]*2, c=f'C{i%5}', lw=1)
                ax3d.plot([dotProducts[i, 0, p]], [dotProducts[i, 1, p]], [t], c=f'C{i%5}', lw=0, marker=ms[0])
                ax3d.plot([dotProducts[i+5, 0, p]], [dotProducts[i+5, 1, p]], [t], c=f'C{i%5}', lw=0, marker=ms[1])
    

def plot5(timepoints):
    plt.rcParams['figure.figsize'] = 6.4, 4.8
    f, axes = plt.subplots(2); axes = axes.flatten()
    [a.spines['right'].set_visible(False) for a in axes]
    [a.spines['top'].set_visible(False) for a in axes]
    
    # For each timepoint
    for time, tp, ax in zip([300, 800], timepoints.T, axes):
        ax.set_title(str(time)+' ms', y=0.8, weight='bold')
    
        # Plot individual points in proper colours
        for i in range(5):
            ax.plot(tp[0, i], tp[1, i], c=f'C{i}', lw=0.5, marker=ms[0], ls='--')
            ax.plot(tp[0, i+5], tp[1, i+5], c=f'C{i}', lw=0.5, marker=ms[1])
        
        # Join up values
        ax.plot(tp[0, :5], tp[1, :5], c='k', lw=1, ls='--', zorder=-1, label=labs[0])
        ax.plot(tp[0, 5:], tp[1, 5:], c='k', lw=1, zorder=-1, label=labs[1])
    
    axes[0].legend(loc='upper left', bbox_to_anchor=[1,1])
    
    # Label values and axes
    axes[0].text(-1.6, 0.1, '1', c='C0', weight='bold')
    axes[0].text(-0.9,-0.2, '2', c='C1', weight='bold')
    axes[0].text(-0.6, 0.2, '3', c='C2', weight='bold')
    axes[0].text(1, 0.6, '4', c='C3', weight='bold')
    axes[0].text(1.3, 0.3, '5', c='C4', weight='bold')    
    axes[1].text(-0.75, 0.5, '1', c='C0', weight='bold')
    axes[1].text(-0.5,0.5, '2', c='C1', weight='bold')
    axes[1].text(0.6, -0.2, '3', c='C2', weight='bold')
    axes[1].text(2.25, 0.4, '4', c='C3', weight='bold')
    axes[1].text(2.5, 0.4, '5', c='C4', weight='bold')    
    axes[1].set_xlabel('PC1');
    axes[0].set_ylabel('PC2');axes[1].set_ylabel('PC2');

time = range(00, 1800, 10)  # Data is binned into 10 ms windows
numTimepoints = 180
color_range = [colors('gray', 'black', i/5) for i in range(5)]
color_range = [colors('green', 'blue', i/5) for i in range(5)]
ms = ['o', 'v']; ls = ['-', '--'];
labs = ['Cue 2 adding', 'Cue 2 subtracting']