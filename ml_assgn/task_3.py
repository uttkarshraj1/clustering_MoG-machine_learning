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

X_phonemes_1_2 = np.zeros((np.sum(phoneme_id == 1) + np.sum(phoneme_id == 2), 2))
index = 0
phoneme_label = np.zeros((np.sum(phoneme_id == 1) + np.sum(phoneme_id == 2), 1))
groundtruth_index = 0
for i in range(len(phoneme_id)):
    if phoneme_id[i] == 1 or phoneme_id[i] == 2:
        X_phonemes_1_2[index] = X_full[i]
        index += 1
        if phoneme_id[i] == 1:
            phoneme_label[groundtruth_index] = 1
        elif phoneme_id[i] == 2:
            phoneme_label[groundtruth_index] = 2
        groundtruth_index += 1

# Plot array containing the chosen phonemes

# Create a figure and a subplot
fig, ax1 = plt.subplots()

title_string = 'Phoneme 1 & 2'
# plot the samples of the dataset, belonging to the chosen phoneme (f1 & f2, phoneme 1 & 2)
plot_data(X=X_phonemes_1_2, title_string=title_string, ax=ax1)
# save the plotted points of phoneme 1 as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'dataset_phonemes_1_2.png')
plt.savefig(plot_filename)


#########################################
# Write your code here
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 1
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 2
# Compare these predictions for each sample of the dataset, and calculate the accuracy, and store it in a scalar variable named "accuracy"
########################################/

X=X_phonemes_1_2.copy()

# File path of phoneme 1 & phoneme 2
if k==3:
    phoneme1_path = 'data/GMM_params_phoneme_01_k_03.npy'
    phoneme2_path = 'data/GMM_params_phoneme_02_k_03.npy'
else:
    phoneme1_path = 'data/GMM_params_phoneme_01_k_06.npy'
    phoneme2_path = 'data/GMM_params_phoneme_02_k_06.npy'

# Load data for phoneme1

data_phoneme1 = np.load(phoneme1_path, allow_pickle=True)
data_phoneme1 = np.ndarray.tolist(data_phoneme1)
means1 = data_phoneme1['mu']
weights1 = data_phoneme1['p']
covariance1 = data_phoneme1['s']
predictions1 = get_predictions(means1, covariance1, weights1, X)

# load data for phoneme2

data_phoneme2 = np.load(phoneme2_path, allow_pickle=True)
data_phoneme2 = np.ndarray.tolist(data_phoneme2)
means2 = data_phoneme2['mu']
weights2 = data_phoneme2['p']
covariance2 = data_phoneme2['s']
predictions2 = get_predictions(means2, covariance2, weights2, X)

# print(predictions2)
# print(phoneme_id == 1)

# get sum value from each row for predictions 1 & 2
sumValue_predictions1 = np.zeros((np.sum(phoneme_id == 1) + np.sum(phoneme_id == 2), 1))
sumValue_predictions2 = np.zeros((np.sum(phoneme_id == 1) + np.sum(phoneme_id == 2), 1))
for i in range(len(X)):
    sumValue_predictions1[i] = np.sum(predictions1[i])
    sumValue_predictions2[i] = np.sum(predictions2[i])
print(sumValue_predictions1)

# compare sum values from each array
predicted_label = np.zeros((np.sum(phoneme_id == 1) + np.sum(phoneme_id == 2), 1))
predicted_index = 0
for i in range(len(X)):
    if sumValue_predictions1[i] > sumValue_predictions2[i]:
        predicted_label[predicted_index] = 1
    elif sumValue_predictions1[i] < sumValue_predictions2[i]:
        predicted_label[predicted_index] = 2
    predicted_index += 1

# calculate accuracy
correct_samples = 0
total_samples = 0
for i in range(len(predicted_label)):
    if predicted_label[i] == phoneme_label[i]:
        correct_samples += 1
    total_samples += 1

accuracy = correct_samples/total_samples
print('Accuracy using GMMs with {} components: {:.2f}%'.format(k, accuracy))

################################################
# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()