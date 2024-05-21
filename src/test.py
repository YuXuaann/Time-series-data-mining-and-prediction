import pandas as pd
import matplotlib.pyplot as plt

data_path = r'../data/G_Data0002.csv'
data = pd.read_csv(data_path)
data = data.fillna(0)

data.to_csv(r'./Autoformer-main/data/G_Data0002.csv', index=False)