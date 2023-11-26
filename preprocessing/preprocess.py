from pipeline import *

df = pd.read_csv('../../CSVs/celllines_raw.csv')
print(df.head())
print(df.info())

df.dropna(inplace=True)

pixel_num = len(df.columns) - 1
df_processed = preprocess(train_set, 150, 1000, pixel_num, smoothmeth='SavGol', bcmeth='ARPLS', normmeth='AUC')

print(df_processed.head())
print(df_processed.info())