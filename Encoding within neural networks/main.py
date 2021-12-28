import numpy as np
import scipy
import plot  # Custom plot function for this script
import matplotlib.pyplot as plt
import maths
import sklearn.decomposition
from mpl_toolkits.mplot3d import Axes3D

x = np.load('x.npy')[0] # The value of the cue [numFeatures x numSamples]
y = np.load('y.npy')  # The firing rates of our neurons over time [numFeatures x numSamples x numTimepoints]

# Data parameters
time = range(0, 1800, 10)  # Data is in 10 ms steps
numNeurons, numSamples, numTimepoints = y.shape
numValues = 5

#%%

averages,sems = np.empty((numNeurons,numValues,numTimepoints)),np.empty((numNeurons,numValues,numTimepoints))

for i, (yy, xx) in enumerate(zip(y, x)):
        
    # Calculate mean and SEM for response to each value
    averages[i] = [np.mean(yy[xx==val], axis=0) for val in np.unique(xx)]
    sems[i] = [np.std(yy[xx==val], axis=0) / np.sqrt(len(yy[xx==val])) for val in np.unique(xx)]

plot.plot1(averages, sems);
#%%
coeffs = np.empty((numNeurons, numTimepoints))

for i, (yy, xx) in enumerate(zip(y, x)):
        
    # Fit model and log coefficients
    coeffs[i] = maths.reg(xx, yy)[1]

plot.plot2(coeffs)

#%%
coeffs = np.empty((2, numNeurons, numTimepoints))
for i, (yy, xx) in enumerate(zip(y, x)):
        
    # Random 50:50 split
    randInds = np.random.permutation(numSamples)
    firstsplit,secondsplit = randInds[::2],randInds[1::2]

    # Fit model and log coefficients
    coeffs[0, i] = maths.reg(xx[firstsplit], yy[firstsplit])[1]
    coeffs[1, i] = maths.reg(xx[secondsplit], yy[secondsplit])[1]

# Correlate our coefficients over time
corrMatrix = np.corrcoef(coeffs[0].T, coeffs[1].T)
corrMatrix = corrMatrix[numTimepoints:,:numTimepoints]  # Just want quarter of the corrMatrix

plot.plot3(corrMatrix)

asd
#%%
pvalues = np.empty(corrMatrix.shape)
for i in range(numTimepoints):
    for j in range(numTimepoints):
        pvalues[i,j] = scipy.stats.pearsonr(coeffs[0,:,i],coeffs[1,:,j])[1]
sigCorr = np.copy(corrMatrix)
sigCorr[pvalues>0.05] = np.nan

plot.plot3(corrMatrix, 1)
#%%


pca = sklearn.decomposition.PCA()

# Average over time dimension
avgAverages = np.mean(averages[:, :, 20:100],axis=-1)

# Do PCA
model = pca.fit(avgAverages.T)

plt.plot(model.explained_variance_)
plt.xlabel('Principal component'); plt.ylabel('% Variance Explained')
plt.xticks(range(5)); plt.title('Variance explained by each principal component')

#%%
f = plt.figure()
ax3d = f.add_subplot(111, projection='3d')
ms = ['o', 'v']
ls = ['-', '--']
labs = ['AA', 'AB']
for value in range(numValues):

    a, b = [np.dot(model.components_[ii], averages[:, value]) for ii in range(2)]
    if i in [0, 5]:
        ax3d.plot(a, b, time, ls=ls[value//5], color=f'C{value%5}', lw=2, label=f'{labs[i<5]}', zorder=1)
    else:
        ax3d.plot(a, b, time, ls=ls[value//5], color=f'C{value%5}', lw=2, zorder=1)


#%%

trialtype = np.load('x.npy')[1]

coeffs = np.empty((2, numNeurons, numTimepoints))
for i, (yy, xx, t) in enumerate(zip(y, x, trialtype)):

    # Fit model and log coefficients
    coeffs[0, i] = maths.reg(xx[t==0], yy[t==0])[1]
    coeffs[1, i] = maths.reg(xx[t==1], yy[t==1])[1]

# Correlate our coefficients over time
corrMatrix = np.corrcoef(coeffs[0].T, coeffs[1].T)
corrMatrix = corrMatrix[numTimepoints:,:numTimepoints]  # Just want quarter of the corrMatrix

#%
pvalues = np.empty(corrMatrix.shape)
for i in range(numTimepoints):
    for j in range(numTimepoints):
        pvalues[i,j] = scipy.stats.pearsonr(coeffs[0,:,i],coeffs[1,:,j])[1]
sigCorr = np.copy(corrMatrix)
sigCorr[pvalues>0.05] = np.nan

f,ax=plt.subplots(1)
im=plt.imshow(sigCorr)
cbar = f.colorbar(im, ax=ax)
cbar.set_label('Pearson r', rotation=270, fontsize='medium', x=0.5)
ax.set_xticks(range(numTimepoints)[::10]);ax.set_yticks(range(numTimepoints)[::10])
ax.set_xticklabels(time[::10]);ax.set_yticklabels(time[::10])
ax.set_xlabel('Time (ms) post cue onset'); ax.set_ylabel('Time (ms) post cue onset');
plt.axhline(85, ls='--', c='k'); plt.axvline(85, ls='--', c='k')
plt.title('Correlation matrix of coefficients over time')


#%%

averages = np.empty((10, numNeurons, numTimepoints))

for i, (yy, xx, trialtype) in enumerate(zip(y, x, t)):
        
    # Calculate mean and SEM for response to each value
    averages[:5, i] = [np.mean(yy[(xx==val) & (t==0)], axis=0) for val in np.unique(xx)]
    averages[5:, i] = [np.mean(yy[(xx==val) & (t==1)], axis=0) for val in np.unique(xx)]

# Average over time dimension
avgAverages = np.mean(averages[:, :, 30:80],axis=-1)

# Do PCA
model = pca.fit(avgAverages)

plt.plot(model.explained_variance_)
plt.xlabel('Principal component'); plt.ylabel('% Variance Explained')
plt.xticks(range(5)); plt.title('Variance explained by each principal component')

#%%
from mpl_toolkits.mplot3d import Axes3D
f=plt.figure()
gs2 = plt.GridSpec(1,1)
ax3d = f.add_subplot(111, projection='3d')
ms = ['o', 'v']
ls = ['-', '--']
labs = ['AA', 'AB']
start, stop = 5, 140
for i in range(10):

    a, b = [np.dot(model.components_[ii], averages[i]) for ii in range(2)]
    if i in [0, 5]:
        ax3d.plot(a[start:stop], b[start:stop], np.arange(start*10, stop*10, 10), ls=ls[i//5], color=f'C{i%5}', lw=2, label=f'{labs[i<5]}', zorder=1)
    else:
        ax3d.plot(a[start:stop], b[start:stop], np.arange(start*10, stop*10, 10), ls=ls[i//5], color=f'C{i%5}', lw=2, zorder=1)


#%%
ms = ['o', 'v']
ls = ['-', '--']
labs = ['AA', 'AB']

for ii, j in enumerate(np.arange(0,150,10)):
    
    f=plt.figure()
    maxes = f.add_subplot(111)
    maxes.set_title(str(j*10)+' ms', y=0.85)
    
    l = np.empty((3, 10))
    
    # For each value
    for i in range(10):
    
        a, b, c = [np.dot(model.components_[iii], averages[i]) for iii in range(3)]
        l[:, i] = a[j], b[j], c[j]
    
    # Plot individual points in proper colours
    for i in range(5):
        maxes.plot(l[0, i], l[1, i], c=f'C{i}', lw=0.5, marker=ms[0], ls='--')
        maxes.plot(l[0, i+5], l[1, i+5], c=f'C{i}', lw=0.5, marker=ms[1])
        
    # Plot the line
    maxes.plot(l[0, :5], l[1, :5], c='k', lw=1, ls='--', zorder=-1)
    maxes.plot(l[0, 5:], l[1, 5:], c='k', lw=1, zorder=-1)
    
maxes[0].text(-2, 0.5, '1', c='C0', weight='bold')
maxes[0].text(-1.4,0, '2', c='C1', weight='bold')
maxes[0].text(-0.6, 0.2, '3', c='C2', weight='bold')
maxes[0].text(1.6, 0.5, '4', c='C3', weight='bold')
maxes[0].text(2.6, 0.6, '5', c='C4', weight='bold')    
























