import numpy as np
import pandas as pd
from sklearn import metrics
import os

from sklearn.preprocessing import MinMaxScaler
from get_dataframes import get_no_rep_all, get_pos_no_rep, get_pos_rep, get_virus_negative


cwd = os.getcwd()

def make_df(dff, file_name):

    data = {}
    return_df = pd.DataFrame.from_dict(data)
    #print(dff['Group'])
    scaler = MinMaxScaler()
    dff.iloc[:,1] = 100 - dff.iloc[:,1]         #baseline = 100 - baseline, so it does not behave inversely

    df_scaled = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
    #df_scaled['Group'] = dff['Group']
    #df_scaled.insert(0, 'Group', dff['Group'])
    #print(df_scaled['Group'])

    for feature in df_scaled.columns[1:]:
        accuracys = []
        tprs = []
        fprs = []
        aucs = []
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
        for th in np.arange(0, np.max(df_scaled[feature]) + 0.01, step = np.max(df_scaled[feature])/100):
            #print(th)
            df_tmp = df_scaled.copy()
            y = df_tmp['Group'].to_numpy()
            y_pred_proba = df_tmp.iloc[:,-1].to_numpy()
            #fpr, tpr, thresholds = metrics.roc_curve(y, y_pred_proba)
            df_tmp.loc[df_tmp[feature] < th, feature] = 1
            df_tmp.loc[df_tmp[feature] != 1, feature] = 0
            #print(df_tmp.head())
            y_pred = df_tmp[feature].to_numpy()
            y.astype(int)
            y_pred.astype(int)
            
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
            tmp3 = metrics.matthews_corrcoef(y, y_pred)
            if tmp3 > best_mcc:
                best_mcc = tmp3
                mcc_th = th
            #youden
            # index = 0
            # tpr_count = 0
            # fpr_count = 0
            # for sample in y:
            #     if sample == 1 and y_pred[index] == 1: tpr_count += 1
            #     if sample == 0 and y_pred[index] == 1: fpr_count += 1
            #     index += 1
            # tpr = tpr_count/np.count_nonzero(y == 1)
            # fpr = fpr_count/np.count_nonzero(y == 0)
            tmp4 = metrics.balanced_accuracy_score(y, y_pred, adjusted = True)
            #tmp4 = tpr - fpr
            if tmp4 > best_youden:
                best_youden = tmp4
                youden_th = th
            # #roc_auc = metrics.auc(fpr, tpr)
            # #aucs.append(roc_auc)
            # tprs.append(tpr)
            # fprs.append(fpr)
            # ths.append(th)
        print('feature:', feature)
        print('best_acc:', best_accuracy, 'at', accuracy_th)
        print('best_cf1:', best_f1, 'at', f1_th)
        print('best_cohens:', best_cohens, 'at', cohens_th)
        print('best_mcc:', best_mcc, 'at', mcc_th)
        print('best_youden:', best_youden, 'at', youden_th)
        data = {'Feature' : [feature], 'accuracy_th' : [accuracy_th], 'f1_th' : [f1_th], 'cohens_th' : [cohens_th], 'mcc_th' : [mcc_th], 'youden_th' : [youden_th]}
        last = pd.DataFrame.from_dict(data)
        return_df = pd.concat([return_df, last])
    return_df.to_csv(file_name, index=False)

df = get_pos_no_rep(path = cwd + '/assets/data/2022-03-25_norep_all.csv')
df_reduced = get_pos_no_rep(path = cwd + '/assets/data/2022-03-25_norep_all.csv', reduced =True)

df_neg = get_virus_negative(path = cwd + '/assets/data/2022-03-25_virusneg.csv')
df_neg_reduced = get_virus_negative(path = cwd + '/assets/data/2022-03-25_virusneg.csv', reduced =True)

df_no_rep_all = get_no_rep_all(path = cwd + '/assets/data/2022-03-25_norep_all.csv')
df_no_rep_all_reduced = get_no_rep_all(path = cwd + '/assets/data/2022-03-25_norep_all.csv', reduced =True)

df_pos_rep = get_pos_rep(path = cwd + '/assets/data/2022-05-30-B19_pos.csv')
df_pos_rep_reduced = get_pos_rep(path = cwd + '/assets/data/2022-05-30-B19_pos.csv', reduced =True)

make_df(df, cwd + '/assets/data/datasets/virus_pos_no_rep/thresholds.csv')
make_df(df_reduced, cwd + '/assets/data/datasets/virus_pos_no_rep/thresholds_reduced.csv')
#make_df(df_neg, cwd + '/assets/data/datasets/thresholds_virus_negative.csv')
#make_df(df_neg_reduced, cwd + '/assets/data/datasets/thresholds_reduced_virus_negative.csv')
#make_df(df_no_rep_all, cwd + '/assets/data/datasets/thresholds_no_rep_all.csv')
#make_df(df_no_rep_all_reduced,cwd + '/assets/data/datasets/thresholds_reduced_no_rep_all.csv')
#make_df(df_pos_rep, cwd + '/assets/data/datasets/thresholds_virus_pos.csv')
#make_df(df_pos_rep_reduced, cwd + '/assets/data/datasets/thresholds_reduced_virus_pos.csv')