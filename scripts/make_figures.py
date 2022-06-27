from sklearn import metrics
from sklearn.metrics import auc, average_precision_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import scripts.helper_functions as hf

import math
import joblib
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

colors = ['#37AA9C', '#00ccff', '#94F3E4']
colors_dict = {'background' : 'rgb(25, 25, 31)',
        'lightbg' : 'rgb(30, 30, 35)'}

def make_radar_plot(values, labels, markers):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        marker = dict(color = colors[1]),
        name='Custom',
    ))
    
    fig.add_trace(go.Scatterpolar(
        #r = dff.loc[0].values.flatten().tolist()[1:],
        r = markers,
        theta=labels,
        #fill='toself',
        marker = dict(color = colors[0]),
        name='Single Sample'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
            visible=True,
            range=[0, 1]
            )),
        showlegend=False
    )
    return fig

def make_optimized_dataframe(y, y_pred_proba, thresholds, extensive, reverse = False, dp = 3):
    n = 3
    if extensive: n = 7
    accuracy_th = thresholds[0]
    best_accuracy = 0
    f1_th = thresholds[0]
    best_f1 = 0
    cohens_th = thresholds[0]
    best_cohens = 0
    mcc_th = thresholds[0]
    best_mcc = 0
    youden_th = thresholds[0]
    best_youden = 0
    for th in thresholds:
        y_pred_th = hf.get_y_pred_thresholded(y_pred_proba, th, reverse=reverse)
        # print('y')
        # print(y)
        # print('y_pred_proba')
        # print(y_pred_proba)
        # print('y_pred_th')
        # print(y_pred_th)
        tn, fp, fn, tp = metrics.confusion_matrix(y, y_pred_th).ravel()
        #acc
        # print('\nConfusion')
        # print('TN:', tn)
        # print('FP:', fp)
        # print('FN:', fn)
        # print('TP:', tp)
        tmp0 = metrics.accuracy_score(y, y_pred_th)
        if tmp0 > best_accuracy:
            best_accuracy = tmp0
            accuracy_th = th
        #f1
        tmp1 = metrics.f1_score(y, y_pred_th)
        if tmp1 > best_f1:
            best_f1 = tmp1
            f1_th = th
        #cohens kappa
        tmp2 = metrics.cohen_kappa_score(y, y_pred_th)
        if tmp2 > best_cohens:
            best_cohens = tmp2
            cohens_th = th
        #mcc
        tmp3 = metrics.matthews_corrcoef(y, y_pred_th)
        if tmp3 > best_mcc:
            best_mcc = tmp3
            mcc_th = th
        #youden
        #tmp4 =metrics.balanced_accuracy_score(y, y_pred_th, adjusted = True)
        specificity = round(tn / (tn+fp), dp)
        fpr = round(fp / (fp+tn), dp)
        tpr = round(tp / (tp+fn), dp)
        fnr = round(fn/ (fn+tn), dp)

        tmp4 = round(tpr + specificity -1, dp)
        if tmp4 > best_youden:
            best_youden = tmp4
            youden_th = th

    # print('best_acc:', best_accuracy, 'at', accuracy_th)
    # print('best_cf1:', best_f1, 'at', f1_th)
    # print('best_cohens:', best_cohens, 'at', cohens_th)
    # print('best_mcc:', best_mcc, 'at', mcc_th)
    # print('best_youden:', best_youden, 'at', youden_th)

    d = {'Name' : ['Accuracy', 'F1_Score', 'Youden', 'Cohen\'s Kappa', 'MCC'],
        'Threshold' : [round(accuracy_th, dp), round(f1_th, dp), round(youden_th, dp), round(cohens_th, dp), round(mcc_th, dp)],
        'Score' : [round(best_accuracy, dp), round(best_f1, dp), round(best_youden, dp), round(best_cohens, dp), round(best_mcc, dp)]
    }
    return pd.DataFrame(data=d).head(n)

def make_invariable_dataframe(y, y_pred_proba, dp = 3):
    auc = round(metrics.roc_auc_score(y, y_pred_proba), dp)
    avg_precision = round(metrics.average_precision_score(y, y_pred_proba), dp)
    brier_loss = round(metrics.brier_score_loss(y, y_pred_proba), dp)

    d = {'Name' : ['ROC AUC', 'Avg Precision', 'Brier Loss'],
        'Value' : [auc, avg_precision, brier_loss]
    }

    return pd.DataFrame(data=d)

def make_variable_dataframe(y, y_pred, extensive, beta = 1, dp = 3):
    n = 8
    if extensive: n = 14
    tn, fp, fn, tp = metrics.confusion_matrix(y, y_pred).ravel()
    accuracy = round(metrics.accuracy_score(y, y_pred), dp)
    balanced_acc = round(metrics.balanced_accuracy_score(y, y_pred), dp)
    f1_score = round(metrics.f1_score(y, y_pred), dp)
    precision = round(metrics.precision_score(y, y_pred), dp)
    cohens = round(metrics.cohen_kappa_score(y, y_pred), dp)
    mcc = round(metrics.matthews_corrcoef(y, y_pred), dp)
    fowlkes = round(metrics.fowlkes_mallows_score(y, y_pred), dp)
    fbeta = round(metrics.fbeta_score(y, y_pred, beta=beta), dp)

    specificity = round(tn / (tn+fp), dp)
    fpr = round(fp / (fp+tn), dp)
    tpr = round(tp / (tp+fn), dp)
    fnr = round(fn/ (fn+tn), dp)

    youden = round(tpr + specificity -1, dp)

    d = {'Name': ['Accuracy',
            'balanced Accuracy',
            'Youden',
            'F1_Score',
            'Precision',
            'Sensitivity (TPR)',
            'Fall-Out (FPR)',
            'Specificity (TNR)',
            'FOR',
            'Cohen\'s Kappa',
            'MCC',
            'fowlkes',
            'fbeta'], 
        'Value': [accuracy,
            balanced_acc,
            youden,
            f1_score,
            precision,
            tpr,
            fpr,
            specificity,
            fnr,
            cohens,
            mcc,
            fowlkes,
            fbeta]}
    return pd.DataFrame(data=d).head(n)

def make_density_plot(hist_data, group_labels, show_l, show_hist = False):
    fig_tmp = ff.create_distplot(hist_data,
                            group_labels,
                            show_hist=show_hist,
                            #bin_size= 8.9,
                            colors = colors)
    fig = go.Figure(data=[go.Scatter(fig_tmp['data'][0],
                            #marker_color='blue'
                            showlegend=show_l,
                            ),
                        go.Scatter(fig_tmp['data'][1],
                            #marker_color='blue'
                            showlegend=show_l,
                            )])
    return fig

def make_perct_histogram(df, show_l, sep = None):

    n_bins = 4
    if df.iloc[:,-1].var() < 50: n_bins = 20
    elif df.iloc[:,-1].var() < 100: n_bins = 15
    elif df.iloc[:,-1].var() < 250: n_bins = 12
    elif df.iloc[:,-1].var() < 500: n_bins = 10
    elif df.iloc[:,-1].var() < 600: n_bins = 8
    elif df.iloc[:,-1].var() < 700: n_bins = 6
    elif df.iloc[:,-1].var() < 850: n_bins = 5

    if n_bins > 20: n_bins = 20
    if n_bins < 4: n_bins = 4
    # print('var:', df.iloc[:,-1].var())
    # print('n_bins:', n_bins)
    binned_df = hf.bin_df(df, n_bins, sep=None)
    # print(binned_df)
    # print('__')
    hovertext = []
    for index, row in binned_df.iterrows():
        hovertext.append('Total: %d<br>Group0: %d<br>Group1: %d' % (row['total'], row['absGroup0'], row['absGroup1']))
    # for total in binned_df['total'].to_numpy():
    #     hovertext.append('Total: ' + str(total) + 'Group0' + str(0))
    fig = go.Figure(data = [
                            go.Bar(x = ((binned_df['range_min'] + binned_df['range_max'])/2), y = binned_df['%Group1'], showlegend=show_l, marker_color=colors[1], hovertext = hovertext, text = round(binned_df['%Group1'],2 )),
                            go.Bar(x = ((binned_df['range_min'] + binned_df['range_max'])/2), y = binned_df['%Group0'], showlegend=show_l, marker_color=colors[0], hovertext = hovertext, text = round(binned_df['%Group0'],2 )),
                            ])
    fig.update_layout(barmode='stack')
    # fig= go.Figure(data=[go.Box(x=y1, showlegend=False, notched=True, marker_color="#3f3f3f", name='3'),
    #                     go.Box(x=y1, showlegend=False, notched=True, marker_color="#3f3f3f", name='3'),])
    return fig

def make_histogram(hist_data, show_l, nbins=10):
    fig= go.Figure(data=[go.Histogram(x=hist_data[0], nbinsx= nbins, showlegend=show_l, marker_color=colors[0],),
                        go.Histogram(x=hist_data[1], nbinsx= nbins, showlegend=show_l, marker_color=colors[1],)])
    return fig

def make_feature_importances(ft_importances, feature_names, t_test_df, sort_by):
    feature_importances = pd.DataFrame(ft_importances, columns = feature_names).transpose()
    feature_importances['Mean'] = feature_importances[feature_importances.columns[1:]].sum(axis = 1)/len(feature_names)
    #print(feature_importances)
    #print(t_test_df)
    feature_importances['t_score'] = t_test_df.iloc[:, 0]
    feature_importances['p_value'] = t_test_df.iloc[:, 1]
    hover_data = {'Mean':False,
                    't_score':':.4f',
                    'p_value': ':.4f',
                }
    fi_sorted = feature_importances.sort_values(by=[sort_by])
    #fi_sorted.head(10)
    fig = px.bar(fi_sorted,
        #x = 
        y = 'Mean',
        color= fi_sorted['Mean'],
        color_continuous_scale=colors,
        hover_data = hover_data
    )
    return fig

def make_confusion_matrix(X_test, y_test, classifier, th, beta, extensive, reverse = False):
    # print('zeros:', (y_test == 0).sum())
    # print('one:', (y_test == 1).sum())
    # print('y_test:', y_test)
    y_pred_proba = classifier.predict_proba(X_test)
    y_pred_th = hf.get_y_pred_thresholded(y_pred_proba[:,1], th, reverse=reverse)
    #print('pred:', y_pred_th)
    tp_count = 0
    for i in np.arange(len(y_test)):
        if y_test[i] == 1 and y_pred_th[i] == 1:
            tp_count += 1

    #print('tp:', tp_count)
    m = np.transpose(metrics.confusion_matrix(y_test, y_pred_th))
    x0 = ['Negative', 'Positive']
    y0 = ['Predicted<br>Negative', 'Predicted<br>Positive']

    # print('m[0][0]:', m[0][0])
    # print('m[0][1]:', m[0][1])
    # print('m[1][0]:', m[1][0])
    # print('m[1][1]:', m[1][1])
    # fpr = m[1][0] / (m[1][1] + m[0][1])
    # tpr = m[1][1] / (m[1][1] + m[0][1])
    var_df = make_variable_dataframe(y_test, y_pred_th, extensive, beta=beta)

    #fig = ff.create_annotated_heatmap(m, x=x0, y=y0, colorscale='Viridis')
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=[[1, 2],
                    [1, 2]],
                x=x0, y=y0,
                xgap=0,
                ygap=0,
                colorscale=[colors[0], colors[1]],
                showscale=False,
            ),
            go.Heatmap(
                z=m,
                x=x0, y=y0,
                xgap=5,
                ygap=5,
                colorscale=[colors_dict['lightbg'], colors_dict['lightbg']],
                showscale=False,
            )
        ],
        ).update_layout(
            yaxis_scaleanchor="x",
            plot_bgcolor=colors_dict['lightbg'],
            xaxis_showgrid=False, yaxis_showgrid=False
        )
    #fig.update_traces(text=m, texttemplate="%{text}", textfont_color=colors[1], textfont_size=42,)
    anno = str(round(m[0][0]/len(y_test)*100, 2)) + '%'
    fig.add_annotation(text=anno,
                font=dict(color=colors[1], size=12),
                x=0, y=-0.15, showarrow=False)
    anno = str(round(m[1][0]/len(y_test)*100, 2)) + '%'
    fig.add_annotation(text=anno,
                font=dict(color=colors[2], size=12),
                x=0, y=0.85, showarrow=False)
    anno = str(round(m[0][1]/len(y_test) *100, 2)) + '%'
    fig.add_annotation(text=anno,
                font=dict(color=colors[2], size=12),
                x=1, y=-0.15, showarrow=False)
    anno = str(round(m[1][1]/len(y_test)*100, 2)) + '%'
    fig.add_annotation(text=anno,
                font=dict(color=colors[1], size=12),
                x=1, y=0.85, showarrow=False)


    fig.add_annotation(text='TN',
                font=dict(color=colors[1], size=16),
                x=0, y=0.15, showarrow=False)
    fig.add_annotation(text='FP',
                font=dict(color=colors[2], size=16),
                x=0, y=1.15, showarrow=False)
    fig.add_annotation(text='FN',
                font=dict(color=colors[2], size=16),
                x=1, y=0.15, showarrow=False)
    fig.add_annotation(text='TP',
                font=dict(color=colors[1], size=16),
                x=1, y=1.15, showarrow=False)


    fig.add_annotation(text=str(m[0][0]),
                font=dict(color=colors[1], size=42),
                x=0, y=0, showarrow=False)
    fig.add_annotation(text=str(m[1][0]),
                font=dict(color=colors[2], size=42),
                x=0, y=1, showarrow=False)
    fig.add_annotation(text=str(m[0][1]),
                font=dict(color=colors[2], size=42),
                x=1, y=0, showarrow=False)
    fig.add_annotation(text=str(m[1][1]),
                font=dict(color=colors[1], size=42),
                x=1, y=1, showarrow=False)
    return fig, var_df

def make_roc_figure(X_train, X_test, y_train, y_test, cv_split, classifier, cl_name, split_set, slider_val, extensive):
    print('making roc figure ...')
    visible_in_legend = 'legendonly'
    if extensive: visible_in_legend = True
    #actual training
    tprs = []
    aucs = []
    accs = []
    ft_importances = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    fig = go.Figure()
    for train, test in cv_split.split(X_train, y_train):
        classifier.fit(X_train[train], y_train[train])
        probas_ = classifier.predict_proba(X_train[test])
        not_probas = classifier.predict(X_train[test])

        if cl_name == 'knn': ft_importances.append(np.zeros(len(X_train)))
        elif cl_name == 'tree' or cl_name == 'rdf': ft_importances.append(classifier.feature_importances_)
        elif cl_name == 'logreg' or cl_name == 'svm': ft_importances.append(abs(classifier.coef_[0]))
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_train[test], probas_[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = metrics.auc(fpr, tpr)
        cv_youden_th = np.argmax(tpr - fpr)
        cv_youden_score = thresholds[cv_youden_th]
        acc = metrics.accuracy_score(y_train[test], not_probas)
        accs.append(acc)
        aucs.append(roc_auc)
        fig.add_scatter(x=fpr, y=tpr, mode='lines', opacity = 0.5, line=dict(color="#A6A6A6"), visible = visible_in_legend, name = 'Fold ' + str(i))
        i += 1

    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    fig.add_scatter(x=mean_fpr, y=mean_tpr, mode='lines', opacity = 1.0, line=dict(color=colors[0]), name = 'Mean CV Curve',  visible = visible_in_legend)

    #y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba[:,1], drop_intermediate=True)

    th_index = 0
    for th in thresholds:
        # print('slider_val:', slider_val)
        # print('th:', th)
        if slider_val > th: break
        th_index += 1
    #print(thresholds[th_index])
    
    test_youden_index = np.argmax(tpr - fpr)
    df_opt = make_optimized_dataframe(y_test,  y_pred_proba[:,1], thresholds, extensive)
    inv_df = make_invariable_dataframe(y_test, y_pred_proba[:,1])

    fig.add_scatter(x=fpr, y=tpr, mode='lines', opacity = 1.0, line=dict(color=colors[1]), name = 'Test Curve')
    fig.add_traces(
        px.scatter(x=[fpr[th_index]], y =[tpr[th_index]]).update_traces(marker_size=15, marker_color=colors[2],  name = 'Threshold', showlegend =  True).data
    )
    fig.add_traces(
        px.scatter(x=[fpr[test_youden_index]], y =[tpr[test_youden_index]]).update_traces(marker_size=10, marker_color=colors[1], name = 'opt. Youden', showlegend =  True, visible = 'legendonly').data
    )
    #title = dataset_folder + ' - ' + df_name + ' - ' + classifier_name + '<br>AUC: %0.3f, Acc: %0.3f, F1-Score: %0.3f' % (test_auc, test_acc, test_f1)
    return fig, df_opt, inv_df, ft_importances, thresholds[th_index]