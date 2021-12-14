'''
    Helper plot function for main.py
'''
import matplotlib.pyplot as plt
import numpy as np
import scipy

labels = 'Cue looked at', 'Neighbouring cue', 'Far cue #1', 'Far cue #2'

def plot1(avg, sem, t, p):
    lab = f't={np.round(t,2)}\np={np.round(p,4)}'
    plt.errorbar(range(len(avg)), avg,sem, label=lab, fmt="s", c='C0', capsize=4);
    plt.title('Model accuracy (%)'); plt.xlabel('Session #'); plt.ylabel('Accuracy');
    plt.axhline(50, c='k',label='Chance', ls='--'); plt.legend(); plt.ylim(48, 100);

def plot2(bestScores, solvers):
    plt.boxplot(bestScores.T, labels=solvers); plt.axhline(0, c='k')
    plt.ylabel('Accuracy'); plt.title('Score across sessions for each different Log. Reg. solver')
    plt.ylim(50,100);
    
def plot3(coeffs,ps):
    plt.boxplot(coeffs, labels=labels); plt.axhline(0, c='k')
    plt.ylabel('Coefficient'); plt.title('Coefficient loading for predicting direction of the second saccade')
    [plt.text(i+1, np.max(coeffs,axis=0)[i]+0.02,f'p={np.round(ps[i],4)}',ha='center',va='bottom') for i in range(len(ps))];
    plt.text(0.51,-0.01,'Towards far cue',va='top');plt.text(0.51,0.01,'Towards neighbour cue',va='bottom');
    
def plot4(scoresReducNorm, ps):
    avgScoresReducNorm = np.mean(scoresReducNorm,axis=1)
    plt.boxplot(scoresReducNorm.T, labels=labels); plt.axhline(0, c='k')
    plt.ylabel('Accuracy improvement (% change)'); plt.title('Effect of each parameter on model accuracy')
    [plt.text(i+1, np.max(scoresReducNorm,axis=1)[i]+0.2,f'p={np.round(ps[i],4)}',ha='center',va='bottom') for i in range(len(ps))];
    plt.ylim(-3, 18)

def plot5(pseudoRsqrNorm, ps):
    plt.boxplot(pseudoRsqrNorm.T, labels=labels); plt.axhline(0, c='k')
    plt.ylabel('Change in psuedo R-squared'); plt.title('Effect of each parameter on the pseudo R-squared score')
    [plt.text(i+1, np.max(pseudoRsqrNorm,axis=1)[i]+0.01,f'p={np.round(ps[i],4)}',ha='center',va='bottom') for i in range(len(ps))]
    plt.ylim(0,0.4);

def plot6(bics,aics):
    f,axes = plt.subplots(2)
    for arr, lab, ax in zip((bics, aics), ('BIC','AIC'), axes):
        arrNorm = arr[1:] - arr[0]
        ts, ps = scipy.stats.ttest_1samp(arrNorm.T, 0)
        
        # Plot
        ax.boxplot(arrNorm.T, labels=labels); ax.axhline(0, c='k')
        ax.set_ylabel('Change in '+lab); ax.set_ylim(-10,110)
        [ax.text(i+1, np.max(arrNorm,axis=1)[i]+4,f'p={np.round(ps[i],4)}',ha='center',va='bottom') for i in range(len(ps))]
    axes[0].set_title('Effect of each parameter on model\'s BIC/AIC score');