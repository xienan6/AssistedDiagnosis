# -*- coding: utf-8 -*-
import sys
import numpy as np
import random
import pickle
import xlrd
import time
from sklearn.ensemble import RandomForestClassifier


def top_n(lists, n):
	topn = []
	for i in range(n):
			if len(lists) == 0:
				return topn, i
			m = max(lists)
			topn.append(m)
			lists.remove(m)
	return topn, n


def make_group(labels, n_of_labels):
	groups = []
	for i in range(n_of_labels):
		groups.append([])
	for i in range(len(labels)):
		groups[labels[i]].append(i)
	return groups


def make_prox(leaf_ids, n_of_tree):
	proxes = np.zeros((len(leaf_ids), len(leaf_ids)))
	for index in range(len(leaf_ids[0])):
		col = [(leaf_ids[i][index], i) for i in range(len(leaf_ids))]
		col = sorted(col, key=lambda x: x[0])
		firstsample = 0
		for secondsample in range(1, len(leaf_ids)):
			if col[firstsample][0] == col[secondsample][0]:
				for temp in range(firstsample, secondsample):
					proxes[col[temp][1]][col[secondsample][1]] += 1
			else:
				firstsample = secondsample
	proxes /= n_of_tree
	for d in range(len(leaf_ids)):
			proxes[d][d] = 1
	for firstsample in range(len(leaf_ids) - 1):
			for secondsample in range(firstsample + 1, len(leaf_ids)):
				proxes[secondsample][firstsample] = proxes[firstsample][secondsample]
	return proxes


def prox_to_svd(proxes, groups, labels, k=5):
	svds = np.zeros(len(proxes))
	for i in range(len(proxes)):
		sames = groups[labels[i]][:]
		sames.remove(i)
		notsames = []
		for j in range(len(groups)):
			if j != labels[i]:
				notsames += groups[j]
		same_prox = []
		for x in sames:
			same_prox.append(proxes[i][x])
		notsame_prox = []
		for x in notsames:
			notsame_prox.append(proxes[i][x])
		same_topn, tn = top_n(same_prox, k)
		notsame_topn, tn = top_n(notsame_prox, tn)
		if sum(same_topn) != 0:
			svds[i] = sum(notsame_topn)/sum(same_topn)
		else:
			svds[i] = sys.maxint
	return svds


def make_p(svds, num_of_labels):
	pyn = np.zeros(num_of_labels)
	for i in range(len(pyn)):
		pyn[i] += 1
		for j in range(len(svds) - num_of_labels):
			if svds[j] >= svds[len(svds) - num_of_labels + i]:
				pyn[i] += 1
	pyn /= (len(svds) - num_of_labels + 1)
	return pyn


def multi_to_one(features, labels):
	x = []
	y = []
	flag = False
	for i in range(len(labels)):
		if int(labels[i]) == 1:
			x.append(features)
			y.append(i)
			flag = True
	if flag is False:
		x.append(features)
		y.append(len(labels))
	return x, y


def multi_label_data(feature_table, label_table):
	x = []
	y = []
	for i in range(len(feature_table)):
		xx, yy = multi_to_one(feature_table[i], label_table[i])
		x += xx
		y += yy
	return x, y


def cmp_list(list1, list2):
	if len(list1) != len(list2):
		return False
	for x, y in zip(list1, list2):
		if x != y:
			return False
	return True


def random_test_score(test_features, test_labels, features, labels, model, n_of_labels, n_of_trees, k, eta):
	score = 0
	for sample in range(len(test_features)):
		list1 = test_labels[sample][:]
		list2 = []
		x = features[:]
		y = labels[:]
		for i in range(n_of_labels):
			x.append(test_features[sample])
			y.append(i)
		leaf_ids = model.apply(x)
		group = make_group(y, n_of_labels)
		prox = make_prox(leaf_ids, n_of_trees)
		svd = prox_to_svd(prox, group, y, k)
		p = make_p(svd, n_of_labels)
		for label in range(len(p)):
			if 1 - eta < p[label]:
				list2.append(label)
		if cmp_list(list1, list2) is True:
			score += 1
	return score * 1.0 / (len(test_features))


def random_choose(feature_table, label_table, n_of_test):
	features = [feature_table[i] for i in range(len(feature_table))]
	temp = [label_table[i] for i in range(len(label_table))]
	labels = []
	for sample in temp:
		l = []
		for i in range(len(sample)):
			if int(sample[i]) == 1:
				l.append(i)
		labels.append(l)
	h1 = [[x, y] for x, y in zip(features, labels)]
	h = []
	for i in range(n_of_test):
		h.append(h1[random.randint(0, len(feature_table) - 1)])
	x = []
	y = []
	for i in range(len(h)):
		x.append(h[i][0])
		y.append(h[i][1])
	return x, y


def indp_test(x, y, feature_table, label_table, model, target, n_tree, n_label, k, eta, n_multi):
	print(u"测试单个样本...")
	if target >= n_multi:
		print(u"多标签样本%s不存在，无法测试！" % target)
		return
	print(u"通过{}个单标签样本生成{}棵树验证多标签样本{}，其标签为：".format(len(x), n_tree, target)),
	z = feature_table[target]
	for i in range(n_label):
		if int(label_table[target][i]) == 1:
			print("%s " % i),
	print("")
	for i in range(n_label):
		x.append(z)
		y.append(i)
	leaf_id = model.apply(x)
	group = make_group(y, n_label)

	prox = make_prox(leaf_id, n_tree)

	svd = prox_to_svd(prox, group, y, k)

	p = make_p(svd, n_label)
	print(u"样本的各个标签的随机性水平为\n %s" % p)
	print(u"样本的标签在风险%s下为：" % (1 - eta)),
	for label in range(len(p)):
		if 1 - eta < p[label]:
			print(label),


def random_test(x, y, model, feature_table, label_table, n_test, n_label, n_tree, k, eta):
	print(u"大小为%s的测试集验证开始..." % n_test)
	test_feature, test_label = random_choose(feature_table, label_table, n_test)
	scores = random_test_score(test_feature, test_label, x, y, model, n_label, n_tree, k, eta)
	print(u"测试集在风险{}下正确率为{}".format(1 - eta, scores))


def create_pickle(str_pickle, data):
	pickle_wt = file(str_pickle, 'wb')
	pickle_data = data
	pickle.dump(pickle_data, pickle_wt, True)
	pickle_wt.close()


def load_pickle(str_pickle):
	pickle_rd = file(str_pickle, 'rb')
	pickle_data = pickle.load(pickle_rd)
	pickle_rd.close()
	return pickle_data[0], pickle_data[1], pickle_data[2], pickle_data[3], pickle_data[4]


def load_excel():
	feature_data = xlrd.open_workbook(r"features736_95.xlsx")
	feature_table = feature_data.sheet_by_index(0)
	label_data = xlrd.open_workbook(r"labels736_4.xlsx")
	label_table = label_data.sheet_by_index(0)
	feature_table = [feature_table.row_values(i) for i in range(feature_table.nrows)]
	label_table = [label_table.row_values(i) for i in range(label_table.nrows)]
	return feature_table, label_table


def test(Z):
	eta = 0.95
	K = 5
	n_tree = 100
	n_sample = 1377
	str_pickle = "data{}_{}.pkl".format(n_sample, n_tree)
	try:
		X, Y, model, feature_table, label_table = load_pickle(str_pickle)
	except IOError:
		feature_table, label_table = load_excel()
		model = RandomForestClassifier(n_estimators=n_tree, n_jobs=-1)
		X, Y = multi_label_data(feature_table, label_table)
		model.fit(X, Y)
		create_pickle(str_pickle, [X, Y, model, feature_table, label_table])
		n_label = len(label_table[0])
		n_multi = len(feature_table)
	n_label = len(label_table[0])
	return temp_indp_test(X, Y, model, Z, n_tree, n_label, K, eta)


def temp_indp_test(x, y, model, z, n_tree, n_label, k, eta):
	print len(y)
	for i in range(n_label):
		x.append(z)
		y.append(i)
	leaf_id = model.apply(x)
	group = make_group(y, n_label)
	prox = make_prox(leaf_id, n_tree)
	print(prox)
	svd = prox_to_svd(prox, group, y, k)
	print(len(svd))
	p = make_p(svd, n_label)
	labels = []
	for label in range(len(p)):
		if 1 - eta < p[label]:
			labels.append(label)
	return labels
