import pandas as pd
import numpy as np
from nn_main import *
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

path_to_file = '../../CSVs/preprocess_test/wheatdata.csv'
df_all = pd.read_csv(path_to_file)
df_all = df_all.rename(columns = {'label':'labels'})

splits = np.arange(0.05, 1, 0.05)

res_array = np.zeros((len(splits), 4))

for i, split in enumerate(splits):
    num_augment = int(split * len(df_all))
    VAEaug_set, train_set, test_set = augment_vae(num_augment, df_all, split, 1, True)
    GANaug_set = augment_gan(num_augment, train_set, test_set, True)
    acc = NeuralNetwork(train_set, test_set)
    VAE_acc = NeuralNetwork(VAEaug_set, test_set)
    GAN_acc = NeuralNetwork(GANaug_set, test_set)
    res_array[i, :] = np.array([split, acc, VAE_acc, GAN_acc])

res_df = pd.DataFrame(res_array, columns=['test_percentage', 'acc', 'VAE_acc', 'GAN_acc'])
res_df.to_csv('results_splits_wheat.csv')
