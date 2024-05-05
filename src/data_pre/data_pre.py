# 读入data数据
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

data_path = r'../../data/G_Data0001.csv'
data = pd.read_csv(data_path)
data['CreateDate'] = pd.to_datetime(data['CreateDate'])
data['BDiff'] = data['BTotal'].diff()
data.loc[0, 'BDiff'] = 0
data.set_index('CreateDate', inplace=True)
# time = data['CreateDate'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp() / 1000)
alarm_points = data[data['Alarm'] == 400]


def show(target, name):
    assert target in data.columns
    plt.plot(data.index, data[target], color='green')
    if target in alarm_points.columns:
        plt.scatter(alarm_points.index, alarm_points[target], color='red', label='alarm points', s=1)
    plt.title(name)
    plt.xlabel('time')
    plt.ylabel(name)
    plt.show()


def resample(target):
    print(f'before resampled, length of {target} is', len(data[target]))
    resampled_data = data.resample('d').mean().rolling(window=3).mean()  # 使用移动平均重采样到天，减少异常值的影响
    resampled_data.interpolate(method='linear', inplace=True)  # 使用插值法填充缺失值
    print(f'after resampled,  length of {target} is', len(resampled_data[target]))
    plt.plot(resampled_data.index, resampled_data[target], color='green')
    plt.xlabel('time')
    plt.ylabel(target)
    plt.title(f'Resampled {target}')
    plt.show()


# BTotal: 标况累计流量
show('BTotal', 'BTotal')
resample('BTotal')

# BFlow: 标况瞬时流量
show('BFlow', 'BFlow')
resample('BFlow')

# BDiff: 标况差分后的单位时间间隔流量
show('BDiff', 'BDiff')
resample('BDiff')
