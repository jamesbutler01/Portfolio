import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, GridSearchCV
import statsmodels.api as sm

x = np.load('x.npy')  # The value of each of the four images [numMonkeys x numSessions x numImages x numTrials]
y= np.load('y.npy')  # Whether they look down or right on the next trial [numSessions x numTrials]
labels = 'Cue looked at', 'Neighbouring cue', 'Far cue #1', 'Far cue #2'

numSessions = x.shape[0]

#%%
logReg = LogisticRegression()

#% We use stratified sampling to ensure an even distribution of labels in our test set
SplitGenerator = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)

scoresFull = np.empty((numSessions, 50))
for iSess, (xx, yy) in enumerate(zip(x, y)):  # For each session
    
    # We have to remove the trailing NaNs from our jagged arrays
    xx, yy = xx.T[~np.isnan(yy)], yy[~np.isnan(yy)]

    scoresFull[iSess] = cross_val_score(logReg, xx, yy, cv=SplitGenerator) * 100

# Stats
avgScores = np.mean(scoresFull, axis=1)
t, p = scipy.stats.ttest_1samp(avgScores, 0.5)

#% Plot
avg, sem = np.mean(scoresFull,axis=1), np.std(scoresFull,axis=1)/np.sqrt(numSessions)
lab = f't={np.round(t,2)}\np={np.round(p,4)}'
plt.plot(avg, label=lab); plt.fill_between(range(len(avg)), avg-sem,avg+sem,alpha=0.5)
plt.title('Model accuracy (%)'); plt.xlabel('Session #'); plt.ylabel('Accuracy');
plt.axhline(50, c='k',label='Chance', ls='--'); plt.legend(); plt.ylim(48, 100);

#%% Hyperparameter tuning
bestscores,bestparams = np.empty((5, numSessions)), np.empty((5, numSessions, 3), dtype=object)
solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
penalties = [['l2', 'none'], ['l2', 'none'], ['l1', 'l2'], ['l2', 'none'],['elasticnet', 'l1', 'l2', 'none']]
c_values = [100, 10, 1.0, 0.1, 0.01]

for  iSolve, (solver,penalty) in enumerate(zip(solvers, penalties)):  # For each solver
    
    # Create grid search
    grid = dict(solver=[solver],penalty=penalty,C=c_values)
    grid_search = GridSearchCV(estimator=logReg, param_grid=grid, cv=SplitGenerator, scoring='accuracy');
    
    for iSess, (xx, yy) in enumerate(zip(x, y)):  # For each session
        
        # We have to remove the trailing NaNs from our jagged data
        xx, yy = xx.T[~np.isnan(yy)], yy[~np.isnan(yy)]
        
        # Perform grid search and log the results
        grid_result = grid_search.fit(xx, yy)
        bestscores[iSolve,iSess] = grid_result.best_score_ * 100
        bestparams[iSolve, iSess] = list(grid_result.best_params_.values())
    
#%% Pairwise stats 
ts, ps = np.ones((len(solvers), len(solvers))), np.ones((len(solvers), len(solvers)))
for i in range(len(solvers)):
    for j in range(len(solvers)):
        ts[i, j],ps[i,j] = scipy.stats.ttest_rel(bestscores[i],bestscores[j])
print('pairwise p-values\n', np.round(ps, 4))
#% Plot
plt.boxplot(bestscores.T, labels=solvers); plt.axhline(0, c='k')
plt.ylabel('Accuracy'); plt.title('Score across sessions for each different Log. Reg. solver')
plt.ylim(50,100)

# Extract our best scoring parameters
iBestSolver = np.argmax(np.mean(bestscores,axis=1))
iBestPenalty= np.argmax([np.sum(bestparams[iBestSolver, :, 1]==p) for p in penalties[iBestSolver]])
iBestC= np.argmax([np.sum(bestparams[iBestSolver, :, 0]==c) for c in c_values])

solver, penalty, c =solvers[iBestSolver], penalties[iBestSolver][iBestPenalty], c_values[iBestC]
logReg = LogisticRegression(solver=solver, penalty=penalty,C=c)
#%% Average coeffs.
coeffs = np.empty((numSessions, 4))
for iSess, (xx, yy) in enumerate(zip(x, y)):  # For each session
    
    # We have to remove the trailing NaNs from our jagged arrays
    xx, yy = xx.T[~np.isnan(yy)], yy[~np.isnan(yy)]

    # Get coefficients of model using all available data
    coeffs[iSess] = logReg.fit(xx, yy).coef_

#Plot
ts, ps = scipy.stats.ttest_1samp(coeffs, 0)
plt.boxplot(coeffs, labels=labels); plt.axhline(0, c='k')
plt.ylabel('Coefficient'); plt.title('Coefficient loading for predicting whether the saccade will be towards the neighbouring cue')
[plt.text(i+1, np.max(coeffs,axis=0)[i]+0.02,f'p={np.round(ps[i],4)}',ha='center',va='bottom') for i in range(len(ps))]
#%%
scoresReduc = np.empty((4, numSessions, 50))
for iSess, (xx, yy) in enumerate(zip(x, y)):  # For each session
    
    # We have to remove the trailing NaNs from our jagged arrays
    xx, yy = xx.T[~np.isnan(yy)], yy[~np.isnan(yy)]

    for iCoeff in range(4):
        
        #Accuracy of the model with feature removed
        xxx = np.delete(xx, iCoeff, axis=1)
        scoresReduc[iCoeff, iSess] = cross_val_score(logReg, xxx, yy, cv=SplitGenerator) * 100


#% Stats
avgScoresReduc = np.mean(scoresReduc, axis=2)
scoresReducNorm = avgScores - avgScoresReduc
ts, ps = scipy.stats.ttest_1samp(scoresReducNorm.T, 0)

# Plot
avgScoresReducNorm = np.mean(scoresReducNorm,axis=1)
plt.boxplot(scoresReducNorm.T, labels=labels); plt.axhline(0, c='k')
plt.ylabel('Accuracy improvement (%)'); plt.title('Effect of each parameter on model accuracy')
[plt.text(i+1, np.max(scoresReducNorm,axis=1)[i]+0.2,f'p={np.round(ps[i],4)}',ha='center',va='bottom') for i in range(len(ps))]

#%% Pseudo R squared

pseudoR = np.empty((5, numSessions))
for iSess, (xx, yy) in enumerate(zip(x, y)):  # For each session
    
    # We have to remove the trailing NaNs from our jagged arrays
    xx, yy = xx.T[~np.isnan(yy)], yy[~np.isnan(yy)]
    
    # SM doesnt add intercept by default
    xx = np.vstack((xx.T, np.ones(len(xx)))).T
    
    # Fit full model and get BIC
    full = sm.Logit(yy, xx).fit(disp=0)
    pseudoR[0, iSess] = full.prsquared
    
    for iCoeff in range(4):
        
        # Remove coefficient
        xxx = np.delete(xx, iCoeff, axis=1)

        # Fit reduced model and log BIC
        red = sm.Logit(yy, xxx).fit(disp=0)
        pseudoR[iCoeff+1, iSess] =  red.prsquared

# Stats
psuedoRsqrNorm = pseudoR[0]  - pseudoR[1:]  # Subtract full model score
ts, ps = scipy.stats.ttest_1samp(psuedoRsqrNorm.T, 0)  # T-test of differences against 0

# Plot
plt.boxplot(psuedoRsqrNorm.T, labels=labels); plt.axhline(0, c='k')
plt.ylabel('Change in psuedo R-squared'); plt.title('Effect of each parameter on model\'s pseudo R-squared score')
[plt.text(i+1, np.max(psuedoRsqrNorm,axis=1)[i]+0.01,f'p={np.round(ps[i],4)}',ha='center',va='bottom') for i in range(len(ps))]
plt.ylim(0,0.4)
#%% BIC and AIC

bics, aics = np.empty((5, numSessions)), np.empty((5, numSessions))
for iSess, (xx, yy) in enumerate(zip(x, y)):  # For each session
    
    # We have to remove the trailing NaNs from our jagged arrays
    xx, yy = xx.T[~np.isnan(yy)], yy[~np.isnan(yy)]
    
    # SM doesnt add intercept by default
    xx = np.vstack((xx.T, np.ones(len(xx)))).T
    
    # Fit full model and get BIC
    full = sm.Logit(yy, xx).fit(disp=0)
    bics[0, iSess] = full.bic
    aics[0, iSess] = full.aic
    
    for iCoeff in range(4):
        
        # Remove coefficient
        xxx = np.delete(xx, iCoeff, axis=1)

        # Fit reduced model and log BIC
        red = sm.Logit(yy, xxx).fit(disp=0)
        bics[iCoeff+1, iSess] =  red.bic
        aics[iCoeff+1, iSess] =  red.aic

# Stats
f,axes = plt.subplots(2)
for arr, lab, ax in zip((bics, aics), ('BIC','AIC'), axes):
    arrNorm = arr[1:] - arr[0]
    ts, ps = scipy.stats.ttest_1samp(arrNorm.T, 0)
    
    # Plot
    ax.boxplot(arrNorm.T, labels=labels); ax.axhline(0, c='k')
    ax.set_ylabel('Change in '+lab); ax.set_ylim(-10,110)
    [ax.text(i+1, np.max(arrNorm,axis=1)[i]+4,f'p={np.round(ps[i],4)}',ha='center',va='bottom') for i in range(len(ps))]
axes[0].set_title('Effect of each parameter on model\'s BIC/AIC score')