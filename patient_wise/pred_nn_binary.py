from nn import *
from utils import *

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from itertools import combinations

# IMPORT DATA
path_to_file = '../../processed_data_untrimmed.parquet'

df_all = pd.read_parquet(path_to_file)

df_all = df_all.rename(columns = {'label':'labels'})

# Specify each patient number for each label
df_all['patient#'] = df_all['labels'].apply(lambda x: x[0]) + df_all['patient#'].astype(str)

sublists = [list(t) for t in list(combinations(['HC', 'GBM', 'MNG'], 2))]
binary_dict = {i:(sublists[i], df_all[df_all['labels'].isin(sublists[i])]) for i in range(3)}

predictions = []
test_datas = []
for i in range(3):
  sublist, df = binary_dict[i]
  print(f'Starting binary classification for {sublist[0]} & {sublist[1]}')

  train_data, test_data = ttsplit(df_all, 0.25)
  test_datas.append(test_data)
  nn_model, nn_probs, nn_predictions = NeuralNetwork(train_data.drop('patient#', axis=1),
               test_data.drop('patient#', axis=1),
               f'ConfMat_NN_Patient_{sublist[0]}&{sublist[1]}')
  predictions.append(nn_predictions)

  y_test = list(test_data['labels'])
  lb = LabelBinarizer()
  y_test = lb.fit_transform(y_test)

  nn_roc_auc = roc_auc_score(y_test, nn_probs, multi_class='ovr')
  print('ROC AUC Score :',nn_roc_auc)

  print('Done! \n')
