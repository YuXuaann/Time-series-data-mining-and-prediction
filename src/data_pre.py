# 读入data数据
import csv

with open(r'../data/G_Data0001.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        print(row)
