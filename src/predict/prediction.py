import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path = r'../Autoformer-main/data/resample.csv'
predict_path = r'../Autoformer-main/results/test_Autoformer_custom_ftMS_sl16_ll10_pl6_dm64_nh8_el2_dl1_df512_fc1_ebtimeF_dtTrue_test_1/'
# predict_path = r'../Autoformer-main/results/test_Autoformer_custom_ftMS_sl10_ll10_pl10_dm64_nh8_el2_dl1_df512_fc1_ebtimeF_dtTrue_test_1/'
predict_data_path = predict_path + '/' + 'pred.npy'
true_data_path = predict_path + '/' + 'true.npy'
paths = [predict_data_path, true_data_path]

data = pd.read_csv(data_path)
predict_data = np.load(predict_data_path)
true_data = np.load(true_data_path)

predict_data_sum = []
for x in predict_data:
    sum = 0
    for y in x:
        sum += y[0]
    predict_data_sum.append(sum)

true_data_sum = []
for x in true_data:
    sum = 0
    for y in x:
        sum += y[0]
    true_data_sum.append(sum)


print(predict_data)
print(true_data_sum)
print(predict_data_sum)

# plt.plot(data['date'][:154], data['BFlow'][:154])
plt.plot(data['date'], data['BFlow'])
plt.xlabel('time')
plt.ylabel('BTotal')
plt.title('BTotal')
plt.show()

plt.plot(data['date'][:154], true_data_sum, label='true')
plt.plot(data['date'][:154], predict_data_sum, label='predict')
plt.xlabel('time')
plt.ylabel('BTotal')
plt.title('BTotal')
plt.legend(loc='lower right')
plt.show()

# cnt = 0
# for path in paths:
#     result_data = np.load(path)
#     plt.imshow(result_data, cmap='viridis')
#     plt.colorbar()
#     plt.title(str(cnt))
#     cnt += 1
#     plt.show()
