import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

data = pd.read_csv(r'../../data/G_Data0001.csv')

# 选择要训练的特征列
features = ['GTotal', 'BTotal', 'GFlow', 'BFlow', 'T', 'Pa']
train_data = data[features].values

# 数据标准化
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)

# 定义自编码器模型ae
input_dim = len(features)
encoding_dim = 3  # 编码器输出维度

input_layer = tf.keras.layers.Input(shape=(input_dim,))
encoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
decoder = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoder)  # sigmoid函数
autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoder)

# 编译和训练模型，均方误差（MSE）作为损失函数，Adam优化器进行优化
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(train_data_scaled, train_data_scaled, epochs=100, batch_size=32, shuffle=True)

# 使用训练好的自编码器模型进行预测
reconstructed_data = autoencoder.predict(train_data_scaled)

# 计算重构误差（即原始数据与重构数据之间的差异）
reconstruction_errors = np.mean(np.square(train_data_scaled - reconstructed_data), axis=1)

# 设置阈值(平均平方误差 + 3倍标准差)来判断哪些数据点是异常的
threshold = np.mean(reconstruction_errors) + 3 * np.std(reconstruction_errors)
anomalies = data[reconstruction_errors > threshold]

print("异常数据点:")
print(anomalies)
