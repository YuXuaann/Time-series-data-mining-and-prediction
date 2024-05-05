import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data_path = r'../../data/G_Data0001.csv'
data = pd.read_csv(data_path)
data['CreateDate'] = pd.to_datetime(data['CreateDate'])
data.set_index('CreateDate', inplace=True)
data.drop(columns=['Err', 'Alarm'], inplace=True)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
pca = PCA(n_components=4)  # 降到四维
pca.fit(scaled_data)
reduced_data = pca.transform(scaled_data)

print(reduced_data)
