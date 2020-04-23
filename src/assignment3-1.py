#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import random


# In[18]:


def z_score(dataset):
    means = dataset.mean(axis = 0, skipna = True)
    stds = dataset.std(axis = 0, skipna = True) 
    mean_list = list(means)[0:-1]
    std_list = list(stds)[0:-1]
    length = len(dataset.columns) - 1
    for i in range(length):
        dataset.iloc[:,i] = (dataset.iloc[:,i] - mean_list[i]) / std_list[i]
    return list(means)[0:-1], list(stds)[0:-1], dataset
    
def z_score_with_paras(dataset, mean_list, std_list):
    length = len(dataset.columns) - 1
    for i in range(length):
        dataset.iloc[:,i] = (dataset.iloc[:,i] - mean_list[i]) / std_list[i]
    return dataset


# In[19]:


def k_folds_split(k, dataset):
    length = len(dataset)
    piece_len = int(length / k)
    mylist = list(range(length))
    random.shuffle(mylist)
    result = []
    for i in range(k):
        test_index = mylist[i*piece_len:(i+1)*piece_len]
        train_index = mylist[0:i*piece_len] + mylist[(i+1)*piece_len:]
        result.append((train_index, test_index))
    return result


# In[20]:


# Sigmoid
def sigmoid(z):
    sigmoid = 1.0 / (1.0 + np.exp(-z))
    return sigmoid
 
# log loss 
def loss(h, y):
    loss = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    return loss
 
# gradient direction
def gradient(X, h, y):
    gradient = np.dot(X.T, (h - y)) / len(y)
    return gradient
 
# logistic regression using gradient descent
def logistic_regression(x, y, tol, is_plot, lr=0.05, count=200):
    intercept = np.ones((x.shape[0], 1)) # set initial intercepts to 1
    x = np.concatenate((intercept, x), axis=1)
    w = np.zeros(x.shape[1]) # set parameters to 0
 
    x_axis = []
    y_axis = []

    old_l = 10
    l = 2
    
    i = 0
    for i in range(count): # gradient loop
        z = np.dot(x, w) # liner function
        h = sigmoid(z)
 
        g = gradient(x, h, y) # calculate gradient
        w -= lr * g # calculate step length and do gradient descent
        
        old_l = l
        l = loss(h, y) # between 0-1
        x_axis.append(i + 1)
        y_axis.append(l)
        if np.isnan(l) or np.abs(old_l - l) < tol:
            break
#         print(i + 1, l)
    
    if is_plot:
        plt.plot(x_axis, y_axis)
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.show()
 
    return old_l, w # return final loss value and estimated parameters


# In[21]:


def predict(x, y, w):
    intercept = np.ones((x.shape[0], 1)) # set initial intercepts to 1
    x = np.concatenate((intercept, x), axis=1)
    z = np.dot(x, w) # liner function
    h = sigmoid(z)
    err = 0
    predict_res = []
    for i in range(len(h)):
        if h[i] >= 0.5:
            predict_label = 1
        else:
            predict_label = 0
        predict_res.append(predict_label)
        if y[i] != predict_label:
            err += 1
    return 1 - err/len(y), predict_res


# In[22]:


def cal_recall(predict_res, actual_labels):
    all_relevant = 0
    for i in range(len(actual_labels)):
        if actual_labels[i] == 1:
            all_relevant += 1
    all_irrelavant = len(actual_labels) - all_relevant
    
    tp1 = 0
    tp0 = 0
    for i in range(len(predict_res)):
        if predict_res[i] == 1 and actual_labels[i] == 1:
            tp1 += 1
        if predict_res[i] == 0 and actual_labels[i] == 0:
            tp0 += 1
    avg_recall = np.mean((tp1/all_relevant, tp0/all_irrelavant))
    return avg_recall
        
def cal_precision(predict_res, actual_labels):
    tp1 = 0
    all1 = 0
    tp0 = 0
    all0 = 0
    for i in range(len(predict_res)):
        if predict_res[i] == 1:
            all1 += 1
            if actual_labels[i] == 1:
                tp1 += 1
        if predict_res[i] == 0:
            all0 += 1
            if actual_labels[i] == 0:
                tp0 += 1
    avg_precision = np.mean((tp1/all1, tp0/all0))
    return avg_precision


# In[23]:


def k_folds(dataset, tol, count, lr):
    result_table = {}
    result_table2 = {}
    result_table3 = {}
    for i in range(1, 11):
        result_table[str(i)] = {}
    for i in range(1, 11):
        result_table2[str(i)] = {}
    for i in range(1, 11):
        result_table3[str(i)] = {}
    result_table['mean accuracy'] = {}
    result_table['std accuracy'] = {}
    result_table2['mean recall'] = {}
    result_table2['std recall'] = {}
    result_table3['mean precision'] = {}
    result_table3['std precision'] = {}
    train_accuracys = []
    test_accuracys = []
    train_recalls = []
    test_recalls = []
    train_precisions = []
    test_precisions = []
    
    losses = []
    
    i = 1
    for train_index, test_index in k_folds_split(10, dataset):
        train_data = dataset.iloc[train_index]
        means, stds, train_data = z_score(train_data)
        
        x = train_data.iloc[:,0:-1]
        y = train_data.iloc[:,-1]
        y = y.to_numpy()
        theta = []
        if i <= 1:
            l, theta = logistic_regression(x, y, tol, is_plot=True, count=count, lr=lr)
        else:
            l, theta = logistic_regression(x, y, tol, is_plot=False, count=count, lr=lr)
        losses.append(l)
        accuracy_train, predict_res_train = predict(x, y, theta)
        recall_train = cal_recall(predict_res_train, y)
        precision_train = cal_precision(predict_res_train, y)
        train_accuracys.append(accuracy_train)
        train_recalls.append(recall_train)
        train_precisions.append(precision_train)
        
        
        test_data = dataset.iloc[test_index]
        test_data = z_score_with_paras(test_data, means, stds)

        x = test_data.iloc[:,0:-1]
        y = test_data.iloc[:,-1]
        y = y.to_numpy()
        accuracy_test, predict_res_test = predict(x, y, theta)
        recall_test = cal_recall(predict_res_test, y)
        precision_test = cal_precision(predict_res_test, y)
        test_accuracys.append(accuracy_test)
        test_recalls.append(recall_test)
        test_precisions.append(precision_test)
  
        result_table[str(i)]['train'] = accuracy_train
        result_table[str(i)]['test'] = accuracy_test
        result_table2[str(i)]['train'] = recall_train
        result_table2[str(i)]['test'] = recall_test
        result_table3[str(i)]['train'] = precision_train
        result_table3[str(i)]['test'] = precision_test
        i += 1
    # end for
    
    mean_train_accuracy = np.mean(train_accuracys)
    std_train_accuracy = np.std(train_accuracys, ddof=1)
    mean_test_accuracy = np.mean(test_accuracys)
    std_test_accuracy = np.std(test_accuracys, ddof=1)
    result_table['mean accuracy']['train'] = mean_train_accuracy
    result_table['std accuracy']['train'] = std_train_accuracy
    result_table['mean accuracy']['test'] = mean_test_accuracy
    result_table['std accuracy']['test'] = std_test_accuracy
    
    mean_train_recall = np.mean(train_recalls)
    std_train_recall = np.std(train_recalls, ddof=1)
    mean_test_recall = np.mean(test_recalls)
    std_test_recall = np.std(test_recalls, ddof=1)
    result_table2['mean recall']['train'] = mean_train_recall
    result_table2['std recall']['train'] = std_train_recall
    result_table2['mean recall']['test'] = mean_test_recall
    result_table2['std recall']['test'] = std_test_recall
    
    mean_train_precision = np.mean(train_precisions)
    std_train_precision = np.std(train_precisions, ddof=1)
    mean_test_precision = np.mean(test_precisions)
    std_test_precision = np.std(test_precisions, ddof=1)
    result_table3['mean precision']['train'] = mean_train_precision
    result_table3['std precision']['train'] = std_train_precision
    result_table3['mean precision']['test'] = mean_test_precision
    result_table3['std precision']['test'] = std_test_precision
    
    columns = list(range(1, 11)) + ['mean accuracy', 'std accuracy']
    columns = [ str(x) for x in columns ]
    result_table_df = pd.DataFrame(result_table, index = ['train', 'test'], columns = columns )
    print(result_table_df)
    
    columns = list(range(1, 11)) + ['mean recall', 'std recall']
    columns = [ str(x) for x in columns ]
    result_table_df = pd.DataFrame(result_table2, index = ['train', 'test'], columns = columns )
    print(result_table_df)
    
    columns = list(range(1, 11)) + ['mean precision', 'std precision']
    columns = [ str(x) for x in columns ]
    result_table_df = pd.DataFrame(result_table3, index = ['train', 'test'], columns = columns )
    print(result_table_df)
    
    return np.mean(losses)


# In[ ]:


# Spambase dataset
dataset = pd.read_csv('Assignment3/spambase.csv', header = None)
prob_name = 'Spambase'

count = 1000
lr = 0.08
tols = np.arange(1e-8, 1e-4, 1e-5)

x_axis = tols
y_axis = []
for i in range(len(tols)):
    tol = tols[i]
    l = k_folds(dataset, tol, count, lr)
#     print(i, l)
    y_axis.append(l)
#     break

# print(x_axis)
# print(y_axis)
plt.plot(x_axis, y_axis)
plt.xlabel('tolerance')
plt.ylabel('loss')
plt.show()


# In[25]:


# Breast Cancer dataset
dataset = pd.read_csv('Assignment3/breastcancer.csv', header = None)
prob_name = 'BreastCancer'

tol = 1e-8
count = 200
lr = 0.05
k_folds(dataset, tol, count, lr)


# In[24]:


# Pima Indian Diabetes dataset
dataset = pd.read_csv('Assignment3/diabetes.csv', header = None)
prob_name = 'diabetes'

tol = 1e-20
count = 1000
lr = 0.5
k_folds(dataset, tol, count, lr)


# In[ ]:




