'''
    Helper maths function for main.py
'''
import matplotlib.pyplot as plt
import numpy as np
import scipy

def reg(x, y):
    xx = np.vstack((np.ones(len(x)), x)).T
    return np.dot(np.linalg.pinv(xx), y)
