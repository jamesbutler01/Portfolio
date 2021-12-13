import matplotlib.pyplot as plt
import numpy as np

def colors(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    import matplotlib as mpl
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

def plot1(avgs, sems=None, f=None, ax=None):
    if ax == None: f, ax = plt.subplots(1)
    if sems == None: sems = [None]*len(avgs)
    for i, (avg, sem) in enumerate(zip(avgs, sems)):  # For each value
        ax.plot(time, avg, c=color_range[i], label=f'Value {i+1}')
        
        if sem != None:
            ax.fill_between(time, avg-sem, avg+sem, alpha=0.3, color=color_range[i])
    ax.set_xlim(-100, 1000); ax.legend()
    ax.set_ylabel('Firing rate'); ax.set_xlabel('Time (ms) post cue onset');
    return f, ax


def avgsem(rsqrs, ax=None):
    if ax == None: f, ax= plt.subplots(1)
    for iparam, rsqr in enumerate(rsqrs):
        avg, sem = np.mean(rsqrs[iparam],axis=0), np.std(rsqrs[iparam],axis=0) / np.sqrt(len(rsqrs[iparam]))
        ax.plot(time, avg, c=f'C{iparam}', label=labels[iparam]); ax.fill_between(time, avg-sem, avg+sem, color=f'C{iparam}', alpha=0.4)
    plt.ylabel('CPD');plt.xlabel('Time (ms) post cue on'); plt.legend(); ax.set_xlim(-100,1000)
    return f, ax

def hist(avgNullDist, thresholds, rsqrs, tp):
    f, axes = plt.subplots(2,2)
    plt.subplots_adjust(hspace=0.5)
    
    for iparam, (dist, thresh, obs) in enumerate(zip(avgNullDist, thresholds, rsqrs)):
        ax = axes[iparam//2, iparam%2]
        ax.hist(dist[tp], bins=30); ax.axvline(thresh[tp],c='k',label='p=0.05');
        ax.axvline(obs[tp],c='red',label='Observed')
        ax.set_title(labels[iparam])
    axes[0, 0].legend(loc='upper right'); 
    axes[0, 0].set_ylabel('Count');axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_xlabel('CPD (null distribution)');axes[1, 1].set_xlabel('CPD (null distribution)')

labels=['Cue looked at', 'Neighbouring cue', 'Far cue #1', 'Far cue #2']
time = range(-100, 1000, 10)  # Data is binned into 10 ms windows
color_range = [colors('gray', 'black', i/5) for i in range(5)]
color_range = [colors('green', 'blue', i/5) for i in range(5)]
