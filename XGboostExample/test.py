import matplotlib
import pandas as pd
# finalize model and make a prediction for monthly births with xgboost
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
# 加载数据集
data = pd.read_csv('./data/G_Data0001.csv')
data['CreateDate'] = pd.to_datetime(data['CreateDate'])
data.set_index('CreateDate', inplace=True)

# 生成正样本
L = 25
alarm_points = data[data['Alarm'] == 400]
positive_samples = pd.DataFrame()
for index, row in alarm_points.iterrows():
    start_index = data.index.get_loc(index) - L
    if start_index >= 0:
        sample = data.iloc[start_index:data.index.get_loc(index)].copy()
        sample['Label'] = 1
        positive_samples = positive_samples._append(sample, ignore_index=True)

# 生成负样本
negative_samples = pd.DataFrame()
for _ in range(len(positive_samples) // 10):
    random_index = np.random.randint(0, len(data))
    sample = data.iloc[random_index:random_index + L].copy()
    if sample['Alarm'].sum() == 0:
        sample['Label'] = 0  # 标记为正常
        negative_samples = negative_samples._append(sample, ignore_index=True)

# 合并正负样本
samples = pd.concat([positive_samples, negative_samples])

# 分离特征和目标变量
X = samples.drop('Label', axis=1)
y = samples['Label']

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# 训练XGBoost分类器
model = XGBClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'准确率: {accuracy}')
print(f'混淆矩阵:\n{conf_matrix}')

# 打印测试集的预测结果和实际结果
for i in range(len(X_test)):
    print(f'测试数据 {i+1}:')
    print(f'特征: {X_test.iloc[i].values}')
    print(f'预测的Alarm状态: {"故障" if y_pred[i] == 1 else "正常"}')
    print(f'实际的Alarm状态: {"故障" if y_test.iloc[i] == 1 else "正常"}\n')
labels = ['预测故障', '预测正常', '实际故障', '实际正常']
sizes = [
    (y_pred == 1).sum(),  # 预测故障
    (y_pred == 0).sum(),  # 预测正常
    (y_test == 1).sum(),  # 实际故障
    (y_test == 0).sum()  # 实际正常
]
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
explode = (0.1, 0, 0.1, 0)  # 突出显示“故障”

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # 确保饼状图是圆形的
plt.title('预测与实际故障状态分布')
plt.show()

# 进行多次迭代，生成准确率折线图
iterations = 20  # 迭代次数
accuracies = []

for i in range(iterations):
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # 训练XGBoost分类器
    model = XGBClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

plt.figure(figsize=(20, 5))
sns.lineplot(x=range(1, iterations + 1), y=accuracies)
plt.title('准确率变化趋势')
plt.xlabel('迭代次数')
plt.ylabel('准确率')
plt.show()