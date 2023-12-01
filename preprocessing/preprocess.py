from pipeline import *
import pandas as pd

df = pd.read_csv('../../CSVs/celllines_raw.csv')
print(df.head())
print(df.info())
print(len(df))
df.dropna(inplace=True)

pixel_num = len(df.columns) - 1
df_processed = preprocess(df, 150, 1000, pixel_num, smoothmeth='SavGol', bcmeth='ARPLS', normmeth='AUC')

print(df_processed.head())
print(df_processed.info())

df_processed.to_csv('../../CSVs/celllines.csv', index=False)
