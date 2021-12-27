import numpy as np
import os
import matplotlib.pyplot as plt
from print_values import *
from plot_data_all_phonemes import *
from plot_data import *
import random
from sklearn.preprocessing import normalize
from get_predictions import *
from plot_gaussians import *

# File that contains the data
data_npy_file = 'data/PB_data.npy'

# Loading data from .npy file
data = np.load(data_npy_file, allow_pickle=True)
data = np.ndarray.tolist(data)

# Make a folder to save the figures
figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# Array that contains the phoneme ID (1-10) of each sample
phoneme_id = data['phoneme_id']
# frequencies f1 and f2
f1 = data['f1']
f2 = data['f2']

# Initialize array containing f1 & f2, of all phonemes.
X_full = np.zeros((len(f1), 2))
#########################################
# Write your code here
# Store f1 in the first column of X_full, and f2 in the second column of X_full
########################################/
X_full[:,0] = f1
X_full[:,1] = f2

X_full = X_full.astype(np.float32)

# number of GMM components
k = 6

#########################################
# Write your code here

# Create an array named "X_phonemes_1_2", containing only samples that belong to phoneme 1 and samples that belong to phoneme 2.
# The shape of X_phonemes_1_2 will be two-dimensional. Each row will represent a sample of the dataset, and each column will represent a feature (e.g. f1 or f2)
# Fill X_phonemes_1_2 with the samples of X_full that belong to the chosen phonemes
# To fill X_phonemes_1_2, you can leverage the phoneme_id array, that contains the ID of each sample of X_full
# X_phonemes_1_2 = ...
########################################/
phonemes1_index = np.where(phoneme_id == 1)
phonemes1_index_size = phonemes1_index[0].size
phonemes_1 = np.take(X_full,phonemes1_index,axis=0).reshape(phonemes1_index_size,2)
print(len(phonemes_1))

phonemes2_index = np.where(phoneme_id == 2)
phonemes2_index_size = phonemes2_index[0].size
phonemes_2 = np.take(X_full,phonemes2_index,axis=0).reshape(phonemes2_index_size,2)
print(len(phonemes_2))
# print(phonemes_2)

X_phonemes_1_2 = np.vstack((phonemes_1,phonemes_2))
print(len(X_phonemes_1_2))
# print(X_phonemes_1_2)

# as dataset X, we will use only the samples of phoneme 1 and 2
X = X_phonemes_1_2.copy()

min_f1 = int(np.min(X[:,0]))
max_f1 = int(np.max(X[:,0]))
min_f2 = int(np.min(X[:,1]))
max_f2 = int(np.max(X[:,1]))
N_f1 = max_f1 - min_f1
N_f2 = max_f2 - min_f2
print('f1 range: {}-{} | {} points'.format(min_f1, max_f1, N_f1))
print('f2 range: {}-{} | {} points'.format(min_f2, max_f2, N_f2))

#########################################
# Write your code here

# Create a custom grid of shape N_f1 x N_f2
# The grid will span all the values of (f1, f2) pairs, between [min_f1, max_f1] on f1 axis, and between [min_f2, max_f2] on f2 axis
# Then, classify each point [i.e., each (f1, f2) pair] of that grid, to either phoneme 1, or phoneme 2, using the two trained GMMs
# Do predictions, using GMM trained on phoneme 1, on custom grid
# Do predictions, using GMM trained on phoneme 2, on custom grid
# Compare these predictions, to classify each point of the grid
# Store these prediction in a 2D numpy array named "M", of shape N_f2 x N_f1 (the first dimension is f2 so that we keep f2 in the vertical axis of the plot)
# M should contain "0.0" in the points that belong to phoneme 1 and "1.0" in the points that belong to phoneme 2
########################################/

#get f1 values between [minf1, maxf1]
f1_values = np.linspace(min_f1,max_f1,num=N_f1)
#get f2 values between [minf2, maxf2]
f2_values = np.linspace(min_f2,max_f2,num=N_f2)
#get every possible combination of values
newf1,newf2=np.meshgrid(f2_values,f1_values)

# print(newf1.shape)
# print(newf2.shape)

# generate the 2 arrays for every possible combination between f1 and f2
custom_grid = np.array([newf2,newf1]).T

# File path of phoneme 1 & phoneme 2
if k == 3:
    phoneme1_path = 'data/GMM_params_phoneme_01_k_03.npy'
    phoneme2_path = 'data/GMM_params_phoneme_02_k_03.npy'
else:
    phoneme1_path = 'data/GMM_params_phoneme_01_k_06.npy'
    phoneme2_path = 'data/GMM_params_phoneme_02_k_06.npy'

# Load data for phoneme1
data_phoneme1 = np.load(phoneme1_path, allow_pickle=True)
data_phoneme1 = np.ndarray.tolist(data_phoneme1)
# load data for phoneme2
data_phoneme2 = np.load(phoneme2_path, allow_pickle=True)
data_phoneme2 = np.ndarray.tolist(data_phoneme2)

# predict each row
predict_row = lambda X, p: np.sum(get_predictions(p["mu"], p["s"], p["p"], X), axis=1)
predict_data_phoneme_1 = lambda X: predict_row(X, dict(data_phoneme1))
predict_data_phoneme_2 = lambda X: predict_row(X, dict(data_phoneme2))

P_1 = np.array([predict_data_phoneme_1(F_n) for F_n in custom_grid])
P_2 = np.array([predict_data_phoneme_2(F_n) for F_n in custom_grid])

# Assign 0 if False, 1 if True
M = (P_1 < P_2).astype(np.float32)
print(M)

################################################
# Visualize predictions on custom grid

# Create a figure
#fig = plt.figure()
fig, ax = plt.subplots()

# use aspect='auto' (default is 'equal'), to force the plotted image to be square, when dimensions are unequal
plt.imshow(M, aspect='auto')

# set label of x axis
ax.set_xlabel('f1')
# set label of y axis
ax.set_ylabel('f2')

# set limits of axes
plt.xlim((0, N_f1))
plt.ylim((0, N_f2))

# set range and strings of ticks on axes
x_range = np.arange(0, N_f1, step=50)
x_strings = [str(x+min_f1) for x in x_range]
plt.xticks(x_range, x_strings)
y_range = np.arange(0, N_f2, step=200)
y_strings = [str(y+min_f2) for y in y_range]
plt.yticks(y_range, y_strings)

# set title of figure
title_string = 'Predictions on custom grid'
plt.title(title_string)

# add a colorbar
plt.colorbar()

N_samples = int(X.shape[0]/2)
plt.scatter(X[:N_samples, 0] - min_f1, X[:N_samples, 1] - min_f2, marker='.', color='red', label='Phoneme 1')
plt.scatter(X[N_samples:, 0] - min_f1, X[N_samples:, 1] - min_f2, marker='.', color='green', label='Phoneme 2')

# add legend to the subplot
plt.legend()

# save the plotted points of the chosen phoneme, as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'GMM_predictions_on_grid.png')
plt.savefig(plot_filename)

################################################
# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()