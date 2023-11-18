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

path_to_file = '../../CSVs/diabetes.csv'
df_all = pd.read_csv(path_to_file)

splits = np.arange(0.15, 0.9, 0.05)

res_array = np.zeros((len(splits), 4))

for i, split in enumerate(splits):
    acci = np.zeros(4)
    VAE_acci = np.zeros(4)
    GAN_acci = np.zeros(4)
    for j in range(4):
        num_augment = int(2 * split * len(df_all))
        VAEaug_set, train_set, test_set = augment_vae(num_augment, df_all, split, 20, True)
        GANaug_set = augment_gan(num_augment, train_set, test_set, True, split)
        acci[j] = ConvolutionalNeuralNetwork(train_set, test_set)
        VAE_acci[j] = ConvolutionalNeuralNetwork(VAEaug_set, test_set)
        GAN_acci[j] = ConvolutionalNeuralNetwork(GANaug_set, test_set)
    acc = np.mean(acci)
    VAE_acc = np.mean(VAE_acci)
    GAN_acc = np.mean(GAN_acci)
    print(acc, VAE_acc, GAN_acc)
    res_array[i, :] = np.array([split, acc, VAE_acc, GAN_acc])

res_df = pd.DataFrame(res_array, columns=['test_percentage', 'acc', 'VAE_acc', 'GAN_acc'])
res_df.to_csv('results_splits_diabetes.csv')
