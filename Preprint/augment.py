import pandas as pd
import numpy as np
from cnn_main import *
import sys

sys.path.insert(1, 'VAE/')
sys.path.insert(1, 'GAN/')

from augment_vae import *
from augment_gan import *

# Fetch our data
# ----------------------------------------------------------
# df_all = pd.read_csv('../../CVSs/preprocess_test/data.csv',header=None)
# df_all = df_all.clip(lower=0)
# labels = pd.read_csv('../../CVSs/preprocess_test/labels.csv',header=None)
# df_all['labels']=labels[0]

path_to_file = '../../CSVs/celllines.csv'
df_all = pd.read_csv(path_to_file)
print(df_all.head())

splits = np.arange(0.80, 0.10, -0.05)
print(splits)

splits = np.array([0.5])

res_array = np.zeros((len(splits), 4))
num_repeats = 1

for i, split in enumerate(splits):
    acci = np.zeros(num_repeats)
    VAE_acci = np.zeros(num_repeats)
    GAN_acci = np.zeros(num_repeats)
    for j in range(num_repeats):
        num_augment = int(split * len(df_all))
        VAEaug_set, train_set, test_set = augment_vae(num_augment, df_all, split, 5, True)
        VAE_acci[j] = ConvolutionalNeuralNetwork(VAEaug_set, test_set)
        GANaug_set = augment_gan(num_augment, train_set, test_set, True, split)
        GAN_acci[j] = ConvolutionalNeuralNetwork(GANaug_set, test_set)
        acci[j] = ConvolutionalNeuralNetwork(train_set, test_set)
        
        # RELEASE MEMORY
        del VAEaug_set
        del GANaug_set
        del train_set
        del test_set

    acc_med = np.median(acci)
    VAE_acc_med = np.median(VAE_acci)
    GAN_acc_med = np.median(GAN_acci)

    acc_mean = np.mean(acci)
    VAE_acc_mean = np.mean(VAE_acci)
    GAN_acc_mean = np.mean(GAN_acci)

    acc_var = np.var(acci)
    VAE_acc_var = np.var(VAE_acci)
    GAN_acc_var = np.var(GAN_acci)

    print(acc_med, VAE_acc_med, GAN_acc_med)
    res_array[i, :] = np.array([split, acc_med, VAE_acc_med, GAN_acc_med, 
                                acc_mean, VAE_acc_mean, GAN_acc_mean,
                                acc_var, VAE_acc_var, GAN_acc_var])

res_df = pd.DataFrame(res_array, columns=['test_percentage', 'median acc', 'median VAE_acc', 'median GAN_acc',
                                          'mean acc', 'mean VAE_acc', 'mean GAN_acc',
                                          'var acc', 'var VAE_acc', 'var GAN_acc'])
res_df.to_csv('results_splits_celllines2.csv')
