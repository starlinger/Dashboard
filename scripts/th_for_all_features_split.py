import numpy as np
import pandas as pd
from sklearn import metrics
import os

import argparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from get_dataframes import get_no_rep_all, get_pos_no_rep, get_pos_rep, get_virus_negative
from helper_functions import *

os.chdir('..')
cwd = os.getcwd()
paths = read_from_json(cwd + '/paths.json')
path_to_csv = paths['path_to_csv']
path_to_datasets = paths['path_to_datasets']

def make_df(dff, file_name, random_state = None):

    data = {}
    return_df = pd.DataFrame.from_dict(data)
    #print(dff['Group'])
    scaler = MinMaxScaler()
    #dff.iloc[:,1] = 100 - dff.iloc[:,1]         #baseline = 100 - baseline, so it does not behave inversely

    df_scaled = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
    #df_scaled['Group'] = dff['Group']
    #df_scaled.insert(0, 'Group', dff['Group'])
    #print(df_scaled['Group'])

    # print('getting ths for scaled df')
    # print('[................................................................]')
    # print('[', end = '')
    # dot_step = int(len(df_scaled.columns[1:])/64)
    # f_index = 0
    # for feature in df_scaled.columns[1:]:
    #     accuracys = []
    #     tprs = []
    #     fprs = []
    #     aucs = []
    #     ths = []
    #     #print('sum of:', slct)
    #     accuracy_th = 0
    #     best_accuracy = 0
    #     f1_th = 0
    #     best_f1 = 0
    #     cohens_th = 0
    #     best_cohens = 0
    #     mcc_th = 0
    #     best_mcc = 0
    #     youden_th = 0
    #     best_youden = 0
    #     for th in np.arange(0, np.max(df_scaled[feature]) + 0.01, step = np.max(df_scaled[feature])/100):
    #         #print(th)
    #         df_tmp = df_scaled.copy()
    #         y = df_tmp['Group'].to_numpy()
    #         y_pred_proba = df_tmp[feature].to_numpy()
    #         #fpr, tpr, thresholds = metrics.roc_curve(y, y_pred_proba)
    #         df_tmp.loc[df_tmp[feature] < th, feature] = 1
    #         df_tmp.loc[df_tmp[feature] != 1, feature] = 0
    #         #print(df_tmp.head())
    #         y_pred = df_tmp[feature].to_numpy()
    #         y.astype(int)
    #         y_pred.astype(int)
            
    #         # print('y_pred')
    #         # print(y_pred)
    #         # print('y')
    #         # print(y)
    #         tmp0 = metrics.accuracy_score(y, y_pred)
    #         #print('acc:', tmp0, 'at', th)
    #         if tmp0 > best_accuracy:
    #             best_accuracy = tmp0
    #             accuracy_th = th
    #         #f1
    #         tmp1 = metrics.f1_score(y, y_pred)
    #         if tmp1 > best_f1:
    #             best_f1 = tmp1
    #             f1_th = th
    #         #cohens kappa
    #         tmp2 = metrics.cohen_kappa_score(y, y_pred)
    #         if tmp2 > best_cohens:
    #             best_cohens = tmp2
    #             cohens_th = th
    #         #mcc
    #         # tmp3 = metrics.matthews_corrcoef(y, y_pred)
    #         # if tmp3 > best_mcc:
    #         #     best_mcc = tmp3
    #         #     mcc_th = th
    #         #youden
    #         index = 0
    #         tpr_count = 0
    #         fpr_count = 0
    #         for sample in y:
    #             if sample == 1 and y_pred[index] == 1: tpr_count += 1
    #             if sample == 0 and y_pred[index] == 1: fpr_count += 1
    #             index += 1
    #         tpr = tpr_count/np.count_nonzero(y == 1)
    #         fpr = fpr_count/np.count_nonzero(y == 0)
    #         tmp4 = metrics.balanced_accuracy_score(y, y_pred, adjusted = True)
    #         #tmp4 = tpr - fpr
    #         if tmp4 > best_youden:
    #             best_youden = tmp4
    #             youden_th = th
    #         # #roc_auc = metrics.auc(fpr, tpr)
    #         # #aucs.append(roc_auc)
    #         # tprs.append(tpr)
    #         # fprs.append(fpr)
    #         # ths.append(th)
    #     fpr, tpr, thresholds = metrics.roc_curve(y, y_pred_proba)
    #     roc_auc = metrics.auc(fpr, tpr)
    #     # print('feature:', feature)
    #     # print('metric Roc AUC:', roc_auc)
    #     # print('best_acc:', best_accuracy, 'at', accuracy_th)
    #     # print('best_f1:', best_f1, 'at', f1_th)
    #     # print('best_cohens:', best_cohens, 'at', cohens_th)
    #     # #print('best_mcc:', best_mcc, 'at', mcc_th)
    #     # print('best_youden:', best_youden, 'at', youden_th)
    #     data = {'Feature' : [feature], 'scaled_accuracy_th' : [accuracy_th], 'scaled_f1_th' : [f1_th], 'scaled_cohens_th' : [cohens_th], 'scaled_youden_th' : [youden_th], 'scaled_auc' : [roc_auc]}
    #     last = pd.DataFrame.from_dict(data)
    #     return_df = pd.concat([return_df, last])
    #     if f_index%dot_step == 0: print('.', end = '', flush=True)
    #     f_index += 1
    # print(']')

    #for original

    # X = dff.iloc[:,1:].to_numpy()
    # y = dff['Group'].to_numpy()

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify = y)
    # print('X_train.shape:', X_train.shape)
    # print('len(X_train[:]):', len(X_train[:]))

    accuracy_ths = []
    f1_ths = []
    cohens_ths = []
    mcc_ths = []
    youden_ths = []
    train_aucs = []
    test_aucs = []
    print('[................................................................]')
    print('[', end = '')
    dot_step = int(len(dff.columns[1:])/64)
    f_index = 0
    for feature in dff.columns[1:]:
        accuracys = []
        train_tprs = []
        train_fprs = []
        test_tprs = []
        test_fprs = []
        ths = []
        #print('sum of:', slct)
        accuracy_th = 0
        best_accuracy = 0
        f1_th = 0
        best_f1 = 0
        cohens_th = 0
        best_cohens = 0
        mcc_th = 0
        best_mcc = 0
        youden_th = 0
        best_youden = 0
        from_ = np.min(dff[feature])
        step_ = (np.max(dff[feature] - np.min(dff[feature])))/500
        till_ = np.max(dff[feature]) + step_ + (step_ * 0.1)
        # print('min:', np.min(dff[feature]))
        # print('max:', np.max(dff[feature]))
        # print('from:', from_)
        # print('till:', till_)
        # print('step:',step_)
        for th in np.arange(from_, till_, step = step_):
            #print(th)
            df_tmp = dff.copy()
            y = df_tmp['Group'].to_numpy()
            y_pred_proba = df_tmp.iloc[:,-1].to_numpy()

            #fpr, tpr, thresholds = metrics.roc_curve(y, y_pred_proba)
            #print('before')
            #print(df_tmp[feature].to_numpy())
            df_tmp.loc[df_tmp[feature] < th, feature] = 1
            df_tmp.loc[df_tmp[feature] != 1, feature] = 0
            #print('after')
            #print(df_tmp[feature].to_numpy())
            #print(df_tmp.head())
            y_pred = df_tmp[feature].to_numpy()
            y.astype(int)
            y_pred.astype(int)
            
            y_pred_train, y_pred_test, y_train, y_test = train_test_split(y_pred, y, test_size=0.2, random_state=random_state, stratify = y)
            #print('y_pred')
            #print(y_pred)
            #print('y')
            #print(y)
            tmp0 = metrics.accuracy_score(y, y_pred)
            #print('acc:', tmp0, 'at', th)
            if tmp0 > best_accuracy:
                best_accuracy = tmp0
                accuracy_th = th
            #f1
            tmp1 = metrics.f1_score(y, y_pred)
            if tmp1 > best_f1:
                best_f1 = tmp1
                f1_th = th
            #cohens kappa
            tmp2 = metrics.cohen_kappa_score(y, y_pred)
            if tmp2 > best_cohens:
                best_cohens = tmp2
                cohens_th = th
            #mcc
            # tmp3 = metrics.matthews_corrcoef(y, y_pred)
            # if tmp3 > best_mcc:
            #     best_mcc = tmp3
            #     mcc_th = th
            #youden
            index = 0
            tp_count = 0
            fp_count = 0
            for sample in y_train:
                if sample == 1 and y_pred_train[index] == 1: tp_count += 1
                if sample == 0 and y_pred_train[index] == 1: fp_count += 1
                index += 1
            tpr = tp_count/np.count_nonzero(y_train == 1)
            fpr = fp_count/np.count_nonzero(y_train == 0)
            train_tprs.append(tpr)
            train_fprs.append(fpr)

            index = 0
            tp_count = 0
            fp_count = 0
            for sample in y_test:
                if sample == 1 and y_pred_test[index] == 1: tp_count += 1
                if sample == 0 and y_pred_test[index] == 1: fp_count += 1
                index += 1
            tpr = tp_count/np.count_nonzero(y_test == 1)
            fpr = fp_count/np.count_nonzero(y_test == 0)
            test_tprs.append(tpr)
            test_fprs.append(fpr)
            #tmp4 = metrics.balanced_accuracy_score(y, y_pred, adjusted = True)
            tmp4 = tpr - fpr
            # print('th:', th)
            # print('tpr:', tpr, '- fpr:', fpr)
            # print('youden:', tmp4)
            # if th > 40:
            #     print('stop')
            if tmp4 > best_youden:
                best_youden = tmp4
                youden_th = th
        if f_index%dot_step == 0: print('.', end = '', flush=True)
        f_index += 1
        train_roc_auc = metrics.auc(train_fprs, train_tprs)
        test_roc_auc = metrics.auc(test_fprs, test_tprs)
        # print('feature:', feature)
        # print('manual Roc AUC:', roc_auc)
        # print('best_acc:', best_accuracy, 'at', accuracy_th)
        # print('best_f1:', best_f1, 'at', f1_th)
        # print('best_cohens:', best_cohens, 'at', cohens_th)
        # #print('best_mcc:', best_mcc, 'at', mcc_th)
        # print('best_youden:', best_youden, 'at', youden_th)
        accuracy_ths.append(accuracy_th)
        f1_ths.append(f1_th)
        cohens_ths.append(cohens_th)
        #mcc_ths.append(mcc_th)
        youden_ths.append(youden_th)
        train_aucs.append(train_roc_auc)
        test_aucs.append(test_roc_auc)
    print(']')

    return_df['features'] = dff.columns[1:]
    return_df['accuracy_th'] = accuracy_ths
    return_df['f1_th'] = f1_ths
    return_df['cohens_th'] = cohens_ths
    #return_df['mcc_th'] = mcc_ths
    return_df['youden_th'] = youden_ths
    return_df['train_roc_auc'] = train_aucs
    return_df['test_roc_auc'] = test_aucs
    return_df.to_csv(file_name + '_' + str(random_state) + '.csv', sep = ';', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--split_random_state',
        type=int,
        help='random state of split function',
        required = False
    )

    args = parser.parse_args()

    random_state = None
    if args.split_random_state != None: random_state = args.split_random_state

    #virus_negative
    df = get_virus_negative(path = cwd + '/assets/data/2022-03-25_virusneg.csv')
    df_all = make_feature_engineered_df(df)
    make_df(df_all, cwd + path_to_datasets + '/thresholds_all_split', random_state = random_state)

    df_reduced = get_virus_negative(path = cwd + '/assets/data/2022-03-25_virusneg.csv', reduced =True)
    df_all_reduced = make_feature_engineered_df(df)
    make_df(df_all_reduced, cwd + path_to_datasets + '/thresholds_all_reduced_split', random_state = random_state)

