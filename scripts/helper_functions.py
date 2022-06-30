import json
import joblib
import os

import numpy as np
import pandas as pd

from sklearn import metrics


#functions for feature engineering
def add_sum_of(df, columns, key):
    ret = df.copy()
    ret[key] = df[columns].sum(axis = 1)
    return ret

def add_mean_of(df, to_transform):
    ret = df.copy()
    ret['Mean'] = df[to_transform].mean(axis = 1)
    return ret

def add_ratio_of(df, feature1, feature2, key):
    ret = df.copy()
    ret[key] = df[feature1] / (df[feature2] + 0.01)
    return ret

#returns a list of values which are engineered from existing features
def get_eng_values(df, feat_list):
    ret = []
    for entry in feat_list:
        if entry in df.columns: ret.append(df[entry].values[0])
        elif entry[:3] == 'Sum':
            df_tmp = add_sum_of(df, get_sums_of(entry[3:]), 'tmp')
            ret.append(df_tmp['tmp'].values[0])
        elif entry[:5] == 'ratio':
            f0, f1 = get_ratios_of(entry[5:])
            df_tmp = add_ratio_of(df, f0, f1, 'tmp')
            ret.append(df_tmp['tmp'].values[0])
        elif entry[:8] == 'bl_ratio':
            df_bl = df.copy()
            if entry[9:9+3] == 'Sum':
                #print('adding:', entry[9:-1])
                tmp_list = get_sums_of(entry[9+3:])
                #print(tmp_list)
                df_bl = add_sum_of(df_bl, get_sums_of(entry[9+3:]), entry[9:-1])
            elif entry[9:9+5] == 'ratio':
                f0, f1 = get_ratios_of(entry[9+5:])
                #print('adding:', entry[9:-1])
                df_bl = add_ratio_of(df_bl, f0, f1, entry[9:-1])
            f0 = get_bl_ratios_of(entry[9:-1])
            f1 = df.columns[1]
            df_tmp = add_ratio_of(df_bl, f0, f1, 'tmp')
            # print('df_tmp comumns')
            # print(df_tmp.columns)
            ret.append(df_tmp['tmp'].values[0])
    return ret
    
#return dataframe with feat engineered columns
def get_feat_eng_df(df, feat_list):
    df_tmp = df.copy()
    for entry in feat_list: 
        if entry[:3] == 'Sum':
            df_tmp = add_sum_of(df_tmp, get_sums_of(entry[3:]), entry)
        elif entry[:5] == 'ratio':
            f0, f1 = get_ratios_of(entry[5:])
            df_tmp = add_ratio_of(df_tmp, f0, f1, entry)
        elif entry[:8] == 'bl_ratio':
            # print('checking for Sum:', entry[9:9+3])
            # print('checking for ratio:', entry[9:9+5])
            if entry[9:9+3] == 'Sum':
                #print('adding:', entry[9:-1])
                tmp_list = get_sums_of(entry[9+3:])
                #print(tmp_list)
                df_tmp = add_sum_of(df_tmp, get_sums_of(entry[9+3:]), entry[9:-1])
            elif entry[9:9+5] == 'ratio':
                f0, f1 = get_ratios_of(entry[9+5:])
                #print('adding:', entry[9:-1])
                df_tmp = add_ratio_of(df_tmp, f0, f1, entry[9:-1])
            f0 = get_bl_ratios_of(entry[9:-1])
            f1 = df_tmp.columns[1]
            # print('bl ratio of:')
            # print(f0)
            # print(f1)
            df_tmp = add_ratio_of(df_tmp, f0, f1, entry)
    # print('columns df_tmp')
    # print(df_tmp.columns)
    # print('feat_list')
    # print(feat_list)
    return df_tmp[feat_list]


def get_sums_of(stri):
    ret = []
    split = stri.split('+')
    ret.append(split[0][1:])

    for entry in split[1:-2]:
        ret.append(entry)
    ret.append(split[-2])
    return ret

def get_ratios_of(stri):
    split = stri.split('/')
    ret0 = split[0][1:]
    ret1 = split[1][:-1]
    return ret0, ret1

def get_bl_ratios_of(stri):
    # ret = []
    # if stri[:3] == 'Sum':
    #     f0, f1 = get_sums_of(stri[3:])
    # elif stri[:5] == 'ratio':
    #     f0, f1 = get_ratios_of(stri[5:])
    # else: ret.append(stri)
    # print('ret get bl ratios of')
    # print('[', stri)
    # print('EF baseline (in %, numeric) ]')
    return stri
    
def get_models(path = 'assets/data/datasets/virus_pos_no_rep/df0/best_models/'):
    ret = {}
    #model_label_list = []
    for entry in os.listdir(path):
        if entry[0] != '.':
            #model_label_list.append(entry)
            #ret[entry] = joblib.load(path + entry)
            ret[entry] = 'tmp'
    return ret


def get_datasets(path = 'assets/data/datasets/'):
    ret = {}
    for entry in os.listdir(path):
        if entry[0] != '.':
            ret[entry] = 'tmp'
    return ret

def get_eng_dfs(path = 'assets/data/datasets/virus_pos_no_rep/'):
    ret = {}
    for entry in os.listdir(path):
        if entry[0] == 'd':
            #model_label_list.append(entry)
            tmp_data = read_from_json(path + entry + '/features.json')
            desc = 'label not found [' + entry + ']'
            if 'description' in tmp_data:
                desc = read_from_json(path + entry + '/features.json')['description']
            print(desc)
            #ret[entry] = desc
            ret[desc] = entry
    print('eng_dfs:')
    print(ret)
    return ret

def get_y_pred_thresholded(y_pred, th, reverse = False):
    y_pred_thresholded = []
    for y_ in y_pred:
        if reverse:
            if y_ <= th: y_pred_thresholded.append(1)
            else: y_pred_thresholded.append(0)
        else:
            if y_ >= th: y_pred_thresholded.append(1)
            else: y_pred_thresholded.append(0)
    return y_pred_thresholded

def get_bin_data(df_orig, min_val, max_val, nbins, bin_letter = ''):
    range_ = max_val - min_val
    i_last = min_val
    bin_nr = 0
    percentages = {}
    #print(df_orig.columns[-1])
    print('from:', min_val + range_/nbins)
    print('to:', max_val + range_/nbins)
    print('in', range_/nbins, 'steps')
    for i in np.arange(min_val + range_/nbins, max_val-0.01 + range_/nbins, step = range_/nbins):
        df = df_orig.copy()
        #print('bin number', bin_nr)
        range_min, range_max = i_last, i
        # print('r_min:', range_min)
        # print('r_max:', range_max)
        # print('samples:', df_orig.shape[0])
        df['in_range'] = df.iloc[:,-1].between(left=range_min, right=range_max)
        i_last = i
        df = df[df['in_range'] == True]
        #print(df)

        counts = df['Group'].value_counts()
        if 0 in counts: n_0 = df['Group'].value_counts()[0]
        else: n_0 = 0
        if 1 in counts: n_1 = df['Group'].value_counts()[1]
        else: n_1 = 0
        #total = df.shape[0]
        total = n_0 + n_1
        total_ = total
        if total == 0: total_ = 1

        percentages['bin'+str(bin_nr) + bin_letter] = [range_min, range_max, n_0/total_, n_1/total_, n_0, n_1, total]
        # print('Total:', total)
        # print(n_0, 'class 0:', n_0/total_, '%')
        # print(n_1, 'class 1:', n_1/total_, '%')
        bin_nr += 1
    return percentages

def make_bin_list(n_bins, min_val, max_val, sep):
    # print('%:', ((sep - min_val) / (max_val - min_val)))
    # print('n_binsa:', round(n_bins * ((sep - min_val) / (max_val - min_val))))
    y = round(n_bins * ((sep - min_val) / (max_val - min_val)))
    if y == 0: y = 1
    return y

def bin_df(df_orig, nbins, sep = None):
    #print(df_orig)
    min_val = df_orig.iloc[:,-1].min()
    max_val = df_orig.iloc[:,-1].max()

    if sep != None:
        if sep < min_val or sep > max_val: sep = None
    # print('min:', min_val)
    # print('max:', max_val)
    percentages = {}
    print(df_orig.columns[-1])
    if sep == None: percentages = get_bin_data(df_orig, min_val, max_val, nbins)
    elif sep != None:
        #bin_list = make_bin_list(nbins, min_val, max_val, sep)
        percentages = get_bin_data(df_orig, min_val, sep, make_bin_list(nbins, min_val, max_val, sep), 'a')
        # for entry in percentages:
        #     percentages[entry][2] = percentages[entry][2]/2
        #     percentages[entry][3] = percentages[entry][3]/2
        percentages.update(get_bin_data(df_orig, sep, max_val, nbins - make_bin_list(nbins, min_val, max_val, sep), 'b'))
        #delete double bin
        start = percentages['bin0b'][0]
        for entry in percentages:
            if percentages[entry][0] == start and entry != 'bin0b':
                del percentages[entry]
                break
    #print('sep:', sep)
    return pd.DataFrame.from_dict(percentages, columns = ['range_min', 'range_max', '%Group0', '%Group1', 'absGroup0', 'absGroup1', 'total'], orient = 'index')

def get_proba(values, model, threshold):
    custom = np.array([values]).reshape(1, -1)
    #print('predict_proba:', model.predict_proba(custom))
    confidences = model.predict_proba(custom)[0]
    print('confidences')
    print(confidences)
    #id = np.argmax(confidences)
    id = 0
    if confidences[1] > threshold: id = 1
    return id, confidences[id]

#functions for I/O
#json
def write_to_json(data, filename):
    with open(filename + '.json', 'w') as f:
        json.dump(data, f)
        
def read_from_json(filename):
    with open(filename) as f:
        return json.load(f)
