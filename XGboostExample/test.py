import pandas as pd
# finalize model and make a prediction for monthly births with xgboost
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from xgboost import XGBRegressor

# forecast monthly births with xgboost
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from matplotlib import pyplot

# 将时间序列数据转换为监督学习数据集
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols = list()
	# 输入序列 (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# 预测序列 (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# 整合在一起
	agg = concat(cols, axis=1)
	# 删除带有 NaN 值的行
	if dropnan:
		agg.dropna(inplace=True)
	return agg.values

# 将单变量数据集拆分为训练集/测试集
def train_test_split(data, n_test):
	return data[:-n_test, :], data[-n_test:, :]

# 拟合 xgboost 模型并进行单步预测
def xgboost_forecast(train, testX):
	# 将列表转换为数组
	train = asarray(train)
	# 拆分为输入和输出列
	trainX, trainy = train[:, :-1], train[:, -1]
	# 拟合模型
	model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
	model.fit(trainX, trainy)
	# 进行一步预测
	yhat = model.predict(asarray([testX]))
	return yhat[0]

# 针对单变量数据的逐步前向验证
def walk_forward_validation(data, n_test):
	predictions = list()
	# 拆分数据集
	train, test = train_test_split(data, n_test)
	# 使用训练数据集初始化历史数据
	history = [x for x in train]
	# 遍历测试集中的每个时间步
	for i in range(len(test)):
		# 将测试行拆分为输入和输出列
		testX, testy = test[i, :-1], test[i, -1]
		# 在历史数据上拟合模型并进行预测
		yhat = xgboost_forecast(history, testX)
		# 将预测结果存储在预测列表中
		predictions.append(yhat)
		# 将实际观察结果添加到历史数据中以备下一次循环使用
		history.append(test[i])
		# 汇总进度
		print('>预期=%.1f, 预测=%.1f' % (testy, yhat))
	# 估计预测误差
	error = mean_absolute_error(test[:, -1], predictions)
	return error, test[:, -1], predictions

# 加载数据集
series = read_csv('./data/G_Data0001.csv', header=0, index_col=0)
print(series.columns)
series['Err'] = series['Err'].fillna(0)
series['Alarm'] = series['Alarm'].fillna(0)
values = series[['GTotal', 'BTotal', 'GFlow', 'BFlow', 'T', 'Pa','Alarm']].values
# 将时间序列数据转换为监督学习数据
data = series_to_supervised(values, n_in=6)
# 评估
mae, y, yhat = walk_forward_validation(data, 12)
print('MAE: %.3f' % mae)
# 绘制预期 vs 预测图
pyplot.plot(y, label='预期')
pyplot.plot(yhat, label='预测')
pyplot.legend()
pyplot.show()