from nn import *
from utils import *

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

# IMPORT DATA
path_to_file = '../../processed_data_untrimmed.parquet'

df_all = pd.read_parquet(path_to_file)

df_all = df_all.rename(columns = {'label':'labels'})

# Specify each patient number for each label
df_all['patient#'] = df_all['labels'].apply(lambda x: x[0]) + df_all['patient#'].astype(str)

train_data, test_data = ttsplit(df_all, 0.25)

nn_model, nn_probs = NeuralNetwork(train_data.iloc[:, :-1], test_data.iloc[:, :-1])

y_test = list(test_data['labels'])
lb = LabelBinarizer()
y_test = lb.fit_transform(y_test)

nn_roc_auc = roc_auc_score(y_test, nn_probs, multi_class='ovr')
print('ROC AUC Score :',nn_roc_auc)
