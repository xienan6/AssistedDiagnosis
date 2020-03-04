# -*- coding: utf-8 -*-
import time
from sklearn.ensemble import RandomForestClassifier
import function
'''
作者 Nan
版本 1.0
日期 2016/5/15
功能 基于随机森林多标记预测
更新 excel格式优化 叶子节点索引优化 prox矩阵优化 svd优化
'''

start = time.clock()

# 测试使用
# X = [[0, 0, 0.1], [0, 0, 0], [0, 0.1, 1], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
# Y = [0, 0, 1, 1, 2, 2, 3, 3, 3, 3]
# Z = [0.1, 0.1, 10]
# num_of_label = 4

'''
历史结果记录
n_of_tree time
	eta scores n_of_test
400 7.5s
	0.95 0.99 1000
	0.96 1.00 100
	0.97 0.93 100
	0.98 0.76 200
	0.99 0.72 735
300 6.4s
	0.95 1.00 100
	0.96 0.98 100
	0.97 0.77 100
	0.98 0.74 100
	0.99 0.70 100
200 5.2s
	0.95 0.98 100
	0.96 0.98 100
	0.97 0.82 100
	0.98 0.71 100
	0.99 0.59 100
100 4s
	0.95 0.98 100
	0.96 0.90 100
	0.97 0.89 100
'''

'''
X训练集 Y标签集
eta置信度
K奇异值计算参数
n_tree 随机森林大小
n_sample 单标签的训练样本数
target 单个多标签的测试目标
n_test 测试集大小
str_pickle pickle所在目录
n_label 标签种类数
n_multi 多标签样本数
默认pickel读入 X,Y,model,feature_table,label_table加速
'''
eta = 0.95
K = 5
n_tree = 100
n_sample = 1377
target = 735
n_test = 100
str_pickle = "data{}_{}.pkl".format(n_sample, n_tree)

try:
    X, Y, model, feature_table, label_table = function.load_pickle(str_pickle)
except IOError:
	# print(u"无存档！建立随机森林...")
	feature_table, label_table = function.load_excel()

	model = RandomForestClassifier(n_estimators=n_tree, n_jobs=-1)

	X, Y = function.multi_label_data(feature_table, label_table)

	model.fit(X, Y)

	function.create_pickle(str_pickle, [X, Y, model, feature_table, label_table])

n_label = len(label_table[0])
n_multi = len(feature_table)
Z = feature_table[target]
# function.random_test(X, Y, model, feature_table, label_table, n_test, n_label, n_tree, K, eta)

# function.indp_test(X, Y, feature_table, label_table, model, target, n_tree, n_label, K, eta, n_multi)

labels = function.test(Z)
for label in labels:
	print(str(label)),

# 计时
# end = time.clock()
# print(u"\n总耗时%ss" % (end - start))

