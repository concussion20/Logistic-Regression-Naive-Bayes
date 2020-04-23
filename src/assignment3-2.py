#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import math
from decimal import Decimal


# In[58]:


def load_data(data_file, label_file, map_file):
    data = pd.read_csv(f'Assignment3/20NG_data/{data_file}' , header = None)
    label = pd.read_csv(f'Assignment3/20NG_data/{label_file}' , header = None)
    
    x = {}
    word_cnt_dict = {}
    doc_labels = {}
    doc_len = {}
    
    for i in range(len(label)):
        doc_labels[str(i+1)] = str(label[0][i])
        
    for i in range(len(data)):
        freq_str = data[0][i]
        doc_id, word_id, freq = freq_str.split()
        if doc_id not in x:
            x[doc_id] = {}
        if word_id not in x[doc_id]:
            x[doc_id][word_id] = 0
        x[doc_id][word_id] += int(freq)
        
        if word_id not in word_cnt_dict:
            word_cnt_dict[word_id] = 0
        word_cnt_dict[word_id] += int(freq)
        
        if doc_id not in doc_len:
            doc_len[doc_id] = 0
        doc_len[doc_id] += int(freq)
    
    sorted_word = sorted(word_cnt_dict, key=word_cnt_dict.get, reverse=True)
    return x, word_cnt_dict, sorted_word, doc_labels, doc_len


# In[59]:


def cal_model_paras(x, word_cnt_dict, sorted_word, doc_labels, is_bernoulli):
    y_paras = cal_y_paras(doc_labels)
    if is_bernoulli:
        x_paras = cal_x_paras_bernoulli(x, word_cnt_dict, sorted_word, doc_labels, y_paras)
    else:
        x_paras = cal_x_paras_multinomial(x, word_cnt_dict, sorted_word, doc_labels, y_paras)
    return x_paras, y_paras

def cal_y_paras(doc_labels):
    y_paras = {}
    
    for label in doc_labels.values():
        if label not in y_paras:
            y_paras[label] = 0
        y_paras[label] += 1
        
    for label in y_paras:
        y_paras[label] = y_paras[label]/len(doc_labels)
        
    return y_paras


# In[60]:


def cal_x_paras_bernoulli(x, word_cnt_dict, sorted_word, doc_labels, y_paras):
    x_paras = {}
    
    for label in y_paras:
        x_paras[label] = {}
    
        label_index = []
        for doc_id in x:
            if doc_labels[doc_id] == label:
                label_index.append(doc_id)
    
        for word_id in sorted_word:
            word_cnt = 0
            for doc_id in label_index:
                if word_id in x[doc_id]:
                    word_cnt += 1
            x_paras[label][word_id] = (word_cnt + 1) / (y_paras[label] * len(doc_labels) + 2)
            
    return x_paras
    
def cal_x_paras_multinomial(x, word_cnt_dict, sorted_word, doc_labels, y_paras):
    x_paras = {}
    
    for label in y_paras:
        x_paras[label] = {}
        
        label_index = []
        total_len = 0
        for doc_id in x:
            if doc_labels[doc_id] == label:
                label_index.append(doc_id)
                for word_id in sorted_word:
                    if word_id in x[doc_id]:
                        total_len += x[doc_id][word_id]
                
        for word_id in sorted_word:
            word_cnt = 0
            for doc_id in label_index:
                if word_id in x[doc_id]:
                    word_cnt += x[doc_id][word_id]
            x_paras[label][word_id] = (word_cnt + 1) / (total_len + len(sorted_word))
            
    return x_paras


# In[61]:


def predict(test_data, test_label, sorted_word, x_paras, y_paras, is_bernoulli):
    predict_res = {}
    
    for label in y_paras:
        predict_res[label] = []
    
    for doc_id in test_data:
        if int(doc_id) % 20 == 0:
            print('doc id is ', doc_id)
        max_prob = Decimal(0)
        predict_label = '0'
        for label in y_paras:
            cur_prob = Decimal(str(y_paras[label]))
            if is_bernoulli:
                for word_id in sorted_word:
                    if word_id in test_data[doc_id]:
                        cur_prob *= Decimal(str(x_paras[label][word_id]))
                    else:
                        cur_prob *= Decimal(str(1 - x_paras[label][word_id]))
            else:
                word_set = set(sorted_word)
                sub_doc_len = 0
                for word_id in test_data[doc_id]:
                    if word_id in word_set:
                        sub_doc_len += test_data[doc_id][word_id]
                        tmp_decimal = Decimal(str(x_paras[label][word_id])) ** test_data[doc_id][word_id]
                        cur_prob *= tmp_decimal / math.factorial(test_data[doc_id][word_id])
                cur_prob *= math.factorial(sub_doc_len)
            if cur_prob == 0:
                print('cur_prob is too small!')
            if cur_prob > max_prob:
                max_prob = cur_prob
                predict_label = label
        # end for
        predict_res[predict_label].append(doc_id)
    
    return predict_res


# In[62]:


def cal_accuracy_for_each_class(predict_res, actual_labels):
    accuracy_dict = {}
    all_ids = set(actual_labels.keys())
    
    for label in predict_res:
        err = 0
        doc_ids = predict_res[label]
        for doc_id in doc_ids:
            if actual_labels[doc_id] != label:
                err += 1
        rest_ids = all_ids - set(doc_ids)
        for doc_id in rest_ids:
            if actual_labels[doc_id] == label:
                err += 1
        accuracy_dict[label] = 1 - err / len(actual_labels)
        
    return accuracy_dict

def cal_accuracy(predict_res, actual_labels):
    err = 0
    for label in predict_res:
        doc_ids = predict_res[label]
        for doc_id in doc_ids:
            if actual_labels[doc_id] != label:
                err += 1
    return 1 - err/len(actual_labels)
    
def cal_recall(predict_res, actual_labels):
    y_paras = cal_y_paras(actual_labels)
    recall_dict = {}
    for label in predict_res:
        doc_ids = predict_res[label]
        tp = 0
        for doc_id in doc_ids:
            if label == actual_labels[doc_id]:
                tp += 1
        recall_dict[label] = tp / (len(actual_labels) * y_paras[label])
    avg_recall = np.mean(list(recall_dict.values()))
    return recall_dict, avg_recall
        
def cal_precision(predict_res, actual_labels):
    precision_dict ={}
    for label in predict_res:
        doc_ids = predict_res[label]
        tp = 0
        for doc_id in doc_ids:
            if label == actual_labels[doc_id]:
                tp += 1
        precision_dict[label] = tp / len(doc_ids)
    avg_precision = np.mean(list(precision_dict.values()))
    return precision_dict, avg_precision


# In[65]:


def naive_bayes():
    x, word_cnt_dict, sorted_word, doc_labels, doc_len_train = load_data('train_data.csv', 'train_label.csv', 'train_map.csv')
    test_data, useless_word_cnt_dict, useless_sorted_word, test_label, doc_len_test = load_data('test_data.csv', 'test_label.csv', 'test_map.csv')
    
    voc_sizes = [100, 500, 1000, 2500, 5000, 7500, 10000, 12500, 25000, 50000, len(sorted_word)]

    accuracies_bernoulli = []
    accuracies_multinomial = []
    avg_recalls_bernoulli = []
    avg_recalls_multinomial = []
    avg_precisions_bernoulli = []
    avg_precisions_multinomial = []
    for voc_size in voc_sizes:
        sub_sorted_word = sorted_word[0:voc_size]
        
        x_paras_bernoulli, y_paras_bernoulli = cal_model_paras(x, word_cnt_dict, sub_sorted_word, doc_labels, True)
        x_paras_multinomial, y_paras_multinomial = cal_model_paras(x, word_cnt_dict, sub_sorted_word, doc_labels, False)
        
        predict_res_bernoulli = predict(test_data, test_label, sub_sorted_word, x_paras_bernoulli, y_paras_bernoulli, True)
        predict_res_multinomial = predict(test_data, test_label, sub_sorted_word, x_paras_multinomial, y_paras_multinomial, False)
        
        accuracies_bernoulli.append(cal_accuracy(predict_res_bernoulli, test_label))
        accuracies_multinomial.append(cal_accuracy(predict_res_multinomial, test_label))
        
        recall_dict_bernoulli, avg_recall_bernoulli = cal_recall(predict_res_bernoulli, test_label)
        recall_dict_multinomial, avg_recall_multinomial = cal_recall(predict_res_multinomial, test_label)
        avg_recalls_bernoulli.append(avg_recall_bernoulli)
        avg_recalls_multinomial.append(avg_recall_multinomial)
        
        precision_dict_bernoulli, avg_precision_bernoulli = cal_precision(predict_res_bernoulli, test_label)
        precision_dict_multinomial, avg_precision_multinomial = cal_precision(predict_res_multinomial, test_label)
        avg_precisions_bernoulli.append(avg_precision_bernoulli)
        avg_precisions_multinomial.append(avg_precision_multinomial)
        
#         print(accuracies_bernoulli)
#         print(accuracies_multinomial)
#         print(avg_recalls_bernoulli)
#         print(avg_recalls_multinomial)
#         print(avg_precisions_bernoulli)
#         print(avg_precisions_multinomial)
        
        if voc_size == 5000:
            index = np.arange(1, 21)
            bar_width = 0.35
            opacity = 0.8

            accuracy_dict_bernoulli = cal_accuracy_for_each_class(predict_res_bernoulli, test_label)
            accuracy_dict_multinomial = cal_accuracy_for_each_class(predict_res_multinomial, test_label)
            accuracy_list_bernoulli = []
            accuracy_list_multinomial = []
            for label in sorted(accuracy_dict_bernoulli.keys()):
                accuracy_list_bernoulli.append(accuracy_dict_bernoulli[label])
            for label in sorted(accuracy_dict_multinomial.keys()):
                accuracy_list_multinomial.append(accuracy_dict_multinomial[label])
                
            rects1 = plt.bar(index, accuracy_list_bernoulli, bar_width,
            alpha=opacity,
            label='bernoulli')
            rects2 = plt.bar(index + bar_width, accuracy_list_multinomial, bar_width,
            alpha=opacity,
            label='multinomial')
            plt.xlabel('class')
            plt.ylabel('accuracy')
            plt.xticks(index + bar_width / 2, index)
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            
            recall_list_bernoulli = []
            recall_list_multinomial = []
            for label in sorted(recall_dict_bernoulli.keys()):
                recall_list_bernoulli.append(recall_dict_bernoulli[label])
            for label in sorted(recall_dict_multinomial.keys()):
                recall_list_multinomial.append(recall_dict_multinomial[label])
                
            rects1 = plt.bar(index, recall_list_bernoulli, bar_width,
            alpha=opacity,
            label='bernoulli')
            rects2 = plt.bar(index + bar_width, recall_list_multinomial, bar_width,
            alpha=opacity,
            label='multinomial')
            plt.xlabel('class')
            plt.ylabel('recall')
            plt.xticks(index + bar_width / 2, index)
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            
            precision_list_bernoulli = []
            precision_list_multinomial = []
            for label in sorted(precision_dict_bernoulli.keys()):
                precision_list_bernoulli.append(precision_dict_bernoulli[label])
            for label in sorted(precision_dict_multinomial.keys()):
                precision_list_multinomial.append(precision_dict_multinomial[label])
                
            rects1 = plt.bar(index, precision_list_bernoulli, bar_width,
            alpha=opacity,
            label='bernoulli')
            rects2 = plt.bar(index + bar_width, precision_list_multinomial, bar_width,
            alpha=opacity,
            label='multinomial')
            plt.xlabel('class')
            plt.ylabel('precision')
            plt.xticks(index + bar_width / 2, index)
            plt.legend()
            plt.tight_layout()
            plt.show()
            
    index = range(len(voc_sizes))
    
    plt.plot(index, accuracies_bernoulli, index, accuracies_multinomial)
    plt.xticks(range(len(voc_sizes)), [100, 500, 1000, 2500, 5000, 7500, 10000, 12500, 25000, 50000, 'All'])
    plt.xlabel('voc size')
    plt.ylabel('accuracy')
    plt.legend(['bernoulli', 'multinomial'])
    plt.title('accuracies versus the vocabulary size')
    plt.show()
    
    plt.plot(index, avg_recalls_bernoulli, index, avg_recalls_multinomial)
    plt.xticks(range(len(voc_sizes)), [100, 500, 1000, 2500, 5000, 7500, 10000, 12500, 25000, 50000, 'All'])
    plt.xlabel('voc size')
    plt.ylabel('recall')
    plt.legend(['bernoulli', 'multinomial'])
    plt.title('recalls versus the vocabulary size')
    plt.show()
    
    plt.plot(index, avg_precisions_bernoulli, index, avg_precisions_multinomial)
    plt.xticks(range(len(voc_sizes)), [100, 500, 1000, 2500, 5000, 7500, 10000, 12500, 25000, 50000, 'All'])
    plt.xlabel('voc size')
    plt.ylabel('precision')
    plt.legend(['bernoulli', 'multinomial'])
    plt.title('precisions versus the vocabulary size')
    plt.show()
     
naive_bayes()


# In[ ]:




