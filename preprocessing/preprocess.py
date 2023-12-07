from pipeline import *
import pandas as pd

df = pd.read_csv('../../CSVs/diabetes_raw.csv')
df.drop('patient#', axis=1, inplace=True)
print(df.head())
print(df.info())
print(len(df))
df.dropna(inplace=True)

pixel_num = len(df.columns) - 1
df_processed = preprocess(df, 150, 1000, pixel_num, smoothmeth=None, bcmeth='IASLS', normmeth='Vector')

print(df_processed.head())
print(df_processed.info())

df_processed.to_csv('../../CSVs/diabetes_none.csv', index=False)
