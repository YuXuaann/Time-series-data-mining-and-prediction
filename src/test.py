import pandas as pd
import matplotlib.pyplot as plt

# 示例数据
data = pd.DataFrame({'X': [1, 2, 3, 4, 5], 'Y': [10, 15, 20, 15, 30]})

# 筛选出满足条件的特征点
special_points = data[data['Y'] == 15]

# 绘制折线图，并将特征点以红色标注出来
plt.plot(data['X'], data['Y'], color='blue')  # 折线图
plt.scatter(special_points['X'], special_points['Y'], color='red', label='Special Points')  # 特征点

# 显示图例和标题
plt.legend()
plt.title('Line Plot with Special Points Highlighted')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 显示图表
plt.show()
