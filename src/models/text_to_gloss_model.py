import pandas as pd
df = pd.read_parquet('data/raw/aslg_pc12/data/train-00000-of-00001.parquet')
print(df.size)
