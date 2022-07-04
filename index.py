from pickletools import optimize
from unittest import result
from click import option
from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import dash_daq as daq

import pandas as pd
import numpy as np
import os
import joblib
import copy
import argparse

from scripts.get_dataframes import get_pos_no_rep, get_virus_negative, get_no_rep_all, get_pos_rep
from scripts.make_figures import *
#make_optimized_dataframe, make_perct_histogram, make_roc_figure, make_confusion_matrix, make_feature_importances, make_invariable_dataframe, make_radar_plot
import scripts.helper_functions as hf
from scipy.__config__ import show

from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from scipy import stats

import plotly.express as px  # (version 4.7.0 or higher)
import plotly.figure_factory as ff
from plotly.graph_objs import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Connect to main app.py file
from app import app
from app import server

# Connect to your app pages
from apps import data_exploration, model_prediction, default

cwd = os.getcwd()
colors = ['#37AA9C', '#00ccff', '#94F3E4']
colors_dict = {'background' : 'rgb(25, 25, 31)',
        'lightbg' : 'rgb(30, 30, 35)'}

# styling the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": colors_dict['lightbg'],
}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "16rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

graph_layout = {
        'font': {'color': 'rgba(255,255,255, 255)'},
        'plot_bgcolor': colors_dict['lightbg'],
        'paper_bgcolor':  colors_dict['lightbg'],
        'font_color' : colors[0]
    }

default_figure = default.default_figure

data_table_style_header = default.data_table_style_header
data_table_style_data = default.data_table_style_data

path_to_csv = default.path_to_csv
path_to_datasets = default.path_to_datasets

#get data
dataset_dict = default.dataset_dict

#get th data
thresholds_df_dict = default.thresholds_df_dict

# df = None
# features = df.columns

inp_list_inputs = []
out_list_inputs = []
inp_list_switches = []

def update_feat_inputs(dataset):
    print('Dataset:', dataset)
    #global df
    dff = dataset_dict[dataset]['df']
    global features 
    features = dff.columns
    input_labels = []
    inputs = []
    feat_switches = []
    global inp_list_inputs
    global out_list_inputs
    global inp_list_switches
    inp_list_inputs = []
    out_list_inputs = []
    inp_list_switches = []
    outputs = []
    i = 0
    for feat in features[1:]:
        input_labels.append(
            html.Label(feat,
                style={'margin-bottom': 7,
                'font-size':15},)
            )
        #print('Input value:', round(df[feat].mean(), 2))
        inputs.append(dcc.Input(id='range'+str(i), type='number', min=0, max=dff[feat].max()*1.5, value = round(dff[feat].mean(), 2), style={'width': "90%"}))
        outputs.append(round(dff[feat].mean(), 2))
        inp_list_inputs.append(Input(component_id='range'+str(i), component_property = 'value'))
        out_list_inputs.append(Output(component_id='range'+str(i), component_property = 'value'))
        on = False
        dis = False
        if i < 4: on = True
        if i == 0: dis = True
        #feat_switches.append(daq.ToggleSwitch(id='feat_switch' + str(i), value = on, style={'margin-bottom': 4},))
        feat_switches.append(daq.BooleanSwitch(id='feat_switch' + str(i), on=on, color = colors[0], disabled = dis, style={'margin-bottom': 4},))
        inp_list_switches.append(Input(component_id='feat_switch'+str(i), component_property = 'on'))
        i += 1
    return outputs

def make_titles(labels, cols, rugs):
    if rugs:
        ret = []
        index = 0
        for label in labels:
            ret.append(label)
            index += 1
            if index == cols:
                for i in np.arange(0, cols):
                    ret.append('')
                index = 0
        return ret
    else:
        return labels

def switch_middle(indizes):
    if len(indizes) == 2: return indizes
    arrays = np.split(indizes, indizes[-1])
    #print(arrays)
    for i in np.arange(0, len(arrays), step = 2):
        tmp = arrays[i][int(len(arrays[i])/2)]
        arrays[i][int(len(arrays[i])/2)] = arrays[i+1][int(len(arrays[i+1])/2)-1]
        arrays[i+1][int(len(arrays[i+1])/2)-1] = tmp
    #print(arrays)
    return np.concatenate(arrays, axis = 0)

def make_specs(n, type = 'xy', max_cols = 2, quad = False, rugs = False, b = 0.0):
    if quad == True:
        specs = []
        if n%max_cols == 0:
            to_append = []
            for i in np.arange(0, max_cols):
                to_append.append({"type": type, 'b': b})
            rows = int(n/max_cols)
            cols = max_cols
            for i in np.arange(0, rows):
                specs.append(to_append)
        elif n%2 == 0:
            rows = int(n/2)
            cols = 2
            for i in np.arange(0, rows):
                specs.append([{"type": type, 'b': b}, {"type": type, 'b': b}])
        else:
            rows = n
            cols = 1
            for i in np.arange(0, rows):
                specs.append([{"type": type, 'b': b}])
    else:
        specs = []
        to_append = []
        m = max_cols
        if max_cols > n: m = n 
        for i in np.arange(0, m):
            to_append.append({"type": type, 'b': b})
        
        rows = int(n/max_cols)
        print('n/.max_cols:', n%max_cols)
        if n%max_cols != 0: rows = int(n/max_cols+1)
        cols = m
        for i in np.arange(0, rows):
            specs.append(copy.deepcopy(to_append))
        # print('before:')
        # print(specs)
        # if n%max_cols != 0 and n > max_cols:
        #     for j in np.arange(1, n%max_cols):
        #         specs[-1][-j] = None
        # print('\nafter:')
        print(specs)
    if rugs:
        rows = rows*2
        specs.extend(specs)
        for i in np.arange(0, len(specs), step =2):
            for spec in specs[i]:
                spec['b'] = -0.07
        # print(specs)
    return rows, cols, specs

model_dict = hf.get_models(path = cwd + path_to_datasets + 'virus_pos_no_rep/df0/best_models/')
eng_dfs_dict = hf.get_eng_dfs(path = cwd + path_to_datasets + 'virus_pos_no_rep/')


update_feat_inputs(list(dataset_dict.keys())[0])

#patiend_ids = np.arange(0, df.shape[0])
# dd_options = {}
# for feat in features:
#     # tmp = {'label': feat, 'value': feat}
#     # dd_options.append(tmp)
#     dd_options[feat] = feat
loading_style = {'position': 'absolute', 'align-self': 'center'}

logo_card = dbc.Card([
    dbc.CardImg(src = '/assets/hth_logo.png', title = 'how to Health GmbH', top = True),
    dbc.CardBody([
        dbc.CardLink('howto.health', href= 'https://business.howto.health/', target = '_blank'),
        html.Label('E-Mail_data: business@howto.health'),
        html.Label('Phone: +49 (0)30 57713053')
    ])
], color = colors_dict['lightbg'])

# radar_plot_card = data_exploration.radar_plot_card
# feat_figure_card = data_exploration.feat_figure_card
# auc_plot_card = data_exploration.auc_plot_card

#model_auc_card = model_prediction.model_auc_card

sidebar = html.Div(
    [
        #html.H2("Sidebar", className="display-4"),
        logo_card,
        html.Hr(),
        html.P(
            "Display", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Data Exploration", href="/apps/data_exploration", active="exact"),
                dbc.NavLink("Model Prediction", href="/apps/model_prediction", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)

#---------------------------------------------------------------------------------------------------------------------------------
#Layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    sidebar,
    content,
    html.Div(id= 'hidden_div', style={'display':'none'})
])


@app.callback([Output('page-content', 'children'),
                Output('url', 'pathname'),],
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/data_exploration':
        return data_exploration.layout, pathname
    if pathname == '/apps/model_prediction':
        return model_prediction.layout, pathname
    else:
        return default.layout, '/'

@app.callback(
    [out_list_inputs],
    [Input(component_id='slct_data', component_property = 'value'),]
)
def update_inputs(dataset):
    ret = update_feat_inputs(dataset)
    #print(ret)
    return [ret]

@app.callback(
    [Output(component_id='density_plots_div_left', component_property='children'),
     Output(component_id='density_plots_div_right', component_property='children'),
    ],
    [Input(component_id='slct_data', component_property = 'value'),
     Input(component_id='type', component_property = 'value'),
     Input(component_id='distribution', component_property = 'value'),
     Input(component_id='separator', component_property = 'value'),
     Input(component_id='force_n_bins', component_property = 'value'),
     inp_list_inputs, inp_list_switches]
)
def update_density_graphs(dataset, plot_type, option_dist, separator, force_n_bins, f0, fs0):
    dff = dataset_dict[dataset]['df']
    option_slctd = []
    i = 1       #only show feature graphs for true switches
    features = dff.columns
    for sw in fs0:
        if sw: option_slctd.append(features[i])
        i += 1
    
    if force_n_bins == 0: force_n_bins =  None

    show_rugs = True
    if option_dist == 'None': show_rugs = False

    dict_ = {features[0] : [2]}
    i = 0
    for feat in features[1:]:
        dict_[feat] = [f0[i]]
        i += 1
    df2 = pd.DataFrame(dict_)
    df2 = pd.concat([dff, df2])

    feat_index = 0
    ret = [[], []]
    rows = 1
    row_heights = [1.0]
    if show_rugs:
        rows = 2
        if option_dist == 'rug': row_heights = [0.9, 0.1]
        else: row_heights = [0.7, 0.3]
    for feat in option_slctd:
        fig = make_subplots(
            rows=rows, cols=1,
            #subplot_titles=titles,
            row_heights = row_heights,
            vertical_spacing=0.1,
        )
        df_with_sample = df2.copy()
        chosen = ['Group', feat]
        df_with_sample = df_with_sample[chosen]
        x = df_with_sample[df_with_sample['Group'] == 0]
        y = df_with_sample[df_with_sample['Group'] == 1]
        samp = df_with_sample[df_with_sample['Group'] == 2]
        z = df_with_sample[df_with_sample['Group'] == 3]

        group_labels = ['Group 0', 'Group 1', 'Group 3']
        hist_data = [x[feat], y[feat], z[feat]]
        if len(z) == 0:
            group_labels = ['Group 0', 'Group 1']
            hist_data = [x[feat], y[feat]]
        
        t_test, p_value = stats.ttest_ind(x[feat].to_numpy(), y[feat].to_numpy(), equal_var=False)

        show_l = True
        if plot_type == 'Density': fig_tmp = make_density_plot(hist_data, group_labels, show_l)
        elif plot_type == 'Histogram': fig_tmp = make_histogram(hist_data, show_l)
        elif plot_type == 'perct Hist':
            # if separator == 'None': sep = None
            # elif separator == 'Input':
            #     f_index = 0
            #     for ft in features:
            #         if ft == feat: break
            #         f_index += 1
            #     if f_index > len(f0)-1: f_index = len(f0) -1
            #     sep = f0[f_index]
            # elif separator == 'Mean': sep = dff[feat].mean()
            # elif separator == 'Youden': sep = dff[feat].mean()
            # elif separator == 'F1_Score': sep = dff[feat].mean()
            fig_tmp = make_perct_histogram(dff[chosen], show_l, sep = None, force_n_bins = force_n_bins)
        #density plot lines
        fig.add_trace(fig_tmp['data'][0], row=1, col=1
            )
        fig.add_trace(fig_tmp['data'][1], row=1, col=1
        )
        if len(fig_tmp['data']) > 2:
            fig.add_trace(fig_tmp['data'][2], row=1, col=1
            )
        if show_rugs:
            df_with_sample['rug 1'] = 1.1
            df_with_sample['rug 2'] = 1
            df_with_sample['rug 3'] = 0.9
            if option_dist == 'rug':
                fig.add_trace(go.Scatter(name = group_labels[0], x=hist_data[0], y = df_with_sample['rug 1'],
                                    mode = 'markers',
                                    showlegend=False,
                                    marker=dict(color = colors[0], symbol='line-ns-open')
                                        ), row=2, col=1)
                fig.add_trace(go.Scatter(name =group_labels[1], x=hist_data[1], y = df_with_sample['rug 2'],
                                    mode = 'markers',
                                    showlegend=False,
                                    marker=dict(color = colors[1], symbol='line-ns-open')
                                        ), row=2, col=1)
                if len(fig_tmp['data']) > 2:
                    fig.add_trace(go.Scatter(name =group_labels[2], x=hist_data[2], y = df_with_sample['rug 3'],
                                        mode = 'markers',
                                        showlegend=False,
                                        marker=dict(color = colors[2], symbol='line-ns-open')
                                            ), row=2, col=1)
            elif option_dist == 'box':
                fig.add_trace(go.Box(x=hist_data[0], marker_color = colors[0],
                                    showlegend=False), row=2, col=1)
                fig.add_trace(go.Box(x=hist_data[1], marker_color = colors[1],
                                    showlegend=False), row=2, col=1)
                if len(fig_tmp['data']) > 2:
                    fig.add_trace(go.Box(x=hist_data[2], marker_color = colors[2],
                                    showlegend=False), row=2, col=1)
            elif option_dist == 'violin':
                fig.add_trace(go.Violin(x=hist_data[0], showlegend=False, box_visible=True, line_color= colors[0],
                               meanline_visible=True), row=2, col=1)
                fig.add_trace(go.Violin(x=hist_data[1], box_visible=True, showlegend=False, line_color=colors[1],
                               meanline_visible=True), row=2, col=1)
                if len(fig_tmp['data']) > 2:
                    fig.add_trace(go.Violin(x=hist_data[2], box_visible=True, showlegend=False, line_color=colors[2],
                               meanline_visible=True), row=2, col=1)
            fig.update_yaxes(showgrid=False,
                #range=[0.95,1.15],
                tickfont=dict(color='rgba(0,0,0,0)', size=14), row=2,col=1)
            fig.update_xaxes(showgrid=False, visible=False, showticklabels=False, row=2,col=1)
        fig.update_yaxes(showticklabels=False, showgrid=False)
        fig.update_xaxes(showgrid=False)
        if separator == 'Input': line_x = samp[feat].iloc[0]
        elif separator == 'Mean': line_x = dff[feat].mean()
        elif separator == 'Youden': line_x = thresholds_df_dict[dataset]['df'].loc[thresholds_df_dict[dataset]['df']['Feature'] == feat]['youden_th'].item()
        elif separator == 'F1 Score': line_x = thresholds_df_dict[dataset]['df'].loc[thresholds_df_dict[dataset]['df']['Feature'] == feat]['f1_th'].item()
        if separator != 'None':
            print(line_x)
            fig.add_vline(
                x=line_x, line_width=1.5,
                line_color="yellow",
                row = 1, col = 1
            )
        fig.update_layout(
            title_text = feat + '<br>t_score = ' + '{:4f}'.format(t_test) + '<br>p_value = ' + '{:4f}'.format(p_value)[1:],
            title_x=0.5,
            legend=dict(
                yanchor="top", y=0.99,
                xanchor="left", x=0.75,
                bgcolor="rgba(30,30,35,0.5)",
            )
        )
        fig.update_layout(barmode='stack')
        fig.update_layout(graph_layout)
        height = 350 
        width = 350
        if show_rugs:
            height += 46
            if option_dist != 'rug': height += 86
        fig.update_layout(height = height, width=width, margin=dict(l = 5, r = 0, t = 90, b = 5))
        tmp_graph = dcc.Graph(id='feature_plot' + str(feat_index), figure=fig)
        ret[feat_index%2].append(tmp_graph)
        feat_index += 1
    return ret[0], ret[1]

#another callback
@app.callback(
    [Output(component_id='radar_plot', component_property='figure'),
     #Output('loading1', 'parent_style'),
    ],
    [Input(component_id='slct_data', component_property = 'value'),
     Input(component_id='axis_scale_switch', component_property = 'on'),
     Input(component_id='metric_dd', component_property = 'value'),
    inp_list_inputs]
)
def update_radar_graph(dataset, logarithmic, metric, f0):
    #make radar plot
    dff = dataset_dict[dataset]['df']
    if dataset in thresholds_df_dict: thresholds_df = thresholds_df_dict[dataset]['df']
    else: return [default_figure]
    # print(dff.head(5))
    # if logarithmic:
    #     dff = np.log(1 + dff)
    #     #df_scaled = df_scaled.apply(1 + np.log)
    # print(dff.head(5))

    maxs = dff.max(axis =1)
    scaler = MinMaxScaler()
    
    features = dff.columns[:10]
    dict_ = {features[0] : [2]}
    i = 0
    for feat in features[1:]:
        dict_[feat] = [f0[i]]
        i += 1
    df2 = pd.DataFrame(dict_)
    df2 = pd.concat([dff, df2])

    #df2.iloc[:,1] = 100 - df2.iloc[:,1]         #baseline = 100 - baseline, so it does not behave inversely

    df_scaled = pd.DataFrame(scaler.fit_transform(df2), columns=df2.columns)

    # print(df_scaled.columns[1:2].item())
    # means = {df_scaled.columns[1:2].item() : 1 - df_scaled[df_scaled.columns[1:2].item()].mean()}
    means = {}
    for feat in df_scaled.columns[1:]:
        means[feat] = df_scaled[feat].mean()

    # values = [1 - df_scaled.tail(1).values.flatten().tolist()[1]]
    # values.extend(df_scaled.tail(1).values.flatten().tolist()[2:])
    values = df_scaled.tail(1).values.flatten().tolist()[1:]
    # df_tmp = pd.DataFrame(dict(
    #     r = values,
    #     theta = df_scaled.columns[1:]))
    #fig2 = px.line_polar(df_tmp, r='r', theta='theta', line_close=True)
    col_tmp = df_scaled.columns[1:].tolist()
    labels = []
    for i in np.arange(0, len(col_tmp)):
        labels.append(col_tmp[i].split('(')[0])

        #add markers
    if metric == 'mean': markers = list(means.values())
    else : markers = thresholds_df['scaled_' + metric].tolist()

    fig = make_radar_plot(values, labels, markers)

    fig.update_layout(graph_layout)
    #fig2.update_traces(line_close=True)
    fig.update_layout(height = 300, width=475,
        #title_text="Feature Values in Relation to Cutoff Points"
        )
    fig.update_layout(margin=dict(l = 0, r = 0, t = 12, b = 15))
    new_loading_style = loading_style
    return [fig]

#another callback
@app.callback(
    [Output(component_id='auc_graph', component_property='figure'),
     #Output(component_id='slct_auc_feat', component_property = 'value'),
     Output(component_id='slct_auc_feat', component_property = 'disabled'),
     Output(component_id='combine_method', component_property = 'disabled'),
     Output(component_id='auc_graph_opt_div', component_property = 'children'),
     Output(component_id='auc_graph_inv_div', component_property = 'children'),
    ],
    [Input(component_id='slct_data', component_property = 'value'),
     Input(component_id='slct_auc_feat', component_property = 'value'),
     Input(component_id='combine_method', component_property = 'value'),]
)
def update_auc_expl(dataset, slct, method):
    disable_sclt = False
    disable_method = False
    #dff = df.copy()

    print(slct)
    dff = dataset_dict[dataset]['df']
    dff = dff[dff['Group'] != 3]
    title_method = 'Sum'
    if method == 'Ratio':
        # if len(slct) > 2:
        #     slct = slct[:2]
        # if len(slct) < 2:
        #     slct = features[1:3]
        dff = hf.add_ratio_of(dff, slct[0], slct[1], 'tmp')
        disable_sclt = True
        title_method = 'Ratio'
    elif method == 'Sum': dff = hf.add_sum_of(dff, slct, 'tmp')
    if len(slct) != 2: disable_method = True

    tprs = [0]
    fprs = [0]
    ths = []
    thresholds = np.arange(0, np.max(dff['tmp']), step = np.max(dff['tmp'])/100)
    #print('sum of:', slct)
    opt_df = make_optimized_dataframe(dff.iloc[:,0].to_numpy(), dff.iloc[:,-1].to_numpy(), thresholds, False, reverse=True)
    opt_table = dash_table.DataTable(
                    opt_df.to_dict('records'), [{"name": ['optimized data', i], "id": i} for i in opt_df.columns],
                    id='table_optimized_expl',
                    style_header= data_table_style_header,
                    style_data = data_table_style_data,
                    merge_duplicate_headers=True,
                )

    scaler = MinMaxScaler()
    #print(dff.iloc[:,-1].to_numpy())
    df_scaled = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
    #print(1- df_scaled.iloc[:,-1].to_numpy())
    inv_df = make_invariable_dataframe(dff.iloc[:,0].to_numpy(), 1-df_scaled.iloc[:,-1].to_numpy())
    inv_table = dash_table.DataTable(
                inv_df.to_dict('records'), [{"name": ['invariant data', i], "id": i} for i in inv_df.columns],
                id='table_inv_expl',
                style_header= data_table_style_header,
                style_data = data_table_style_data,
                merge_duplicate_headers=True,
            )
    for th in thresholds:
        #print(th)
        df_tmp = dff.copy()
        y = df_tmp.iloc[:,0].to_numpy()
        y_pred_proba = df_tmp.iloc[:,-1].to_numpy()
        fpr, tpr, thresholds_ = metrics.roc_curve(y, y_pred_proba)
        df_tmp.loc[df_tmp['tmp'] < th, 'tmp'] = 1
        df_tmp.loc[df_tmp['tmp'] != 1, 'tmp'] = 0
        y_pred = df_tmp.iloc[:,-1].to_numpy()
        y.astype(int)
        y_pred.astype(int)
        index = 0
        tpr_count = 0
        fpr_count = 0
        for sample in y:
            if sample == 1 and y_pred[index] == 1: tpr_count += 1
            if sample == 0 and y_pred[index] == 1: fpr_count += 1
            index += 1
        tpr = tpr_count/np.count_nonzero(y == 1)
        fpr = fpr_count/np.count_nonzero(y == 0)
        tprs.append(tpr)
        fprs.append(fpr)
        ths.append(th)
    # Plotly Express
    tprs.append(1)
    fprs.append(1)
    ths.append(np.max(dff['tmp']))
    auc = metrics.auc(fprs, tprs)
    hover_data = np.asarray(tprs)-np.asarray(fprs)
    for i in np.arange(0, len(hover_data)):
        hover_data[i] = round(hover_data[i], 3)
    fig = px.area(
        x=fprs, y=tprs,
        color_discrete_sequence= colors,
        hover_data = {
            'youden' : hover_data,
            }
        # labels=dict(
        #     x='False Positive Rate', 
        #     y='True Positive Rate')
        )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1)
    
    fig.update_layout(graph_layout)
    ft_string = ' features'
    if len(slct) == 1: ft_string = ' feature'
    fig.update_layout(
        title_text = ' Roc Curve of the ' + title_method + ' of ' + str(len(slct)) + ft_string,
        width=425, height=425,
        margin=dict(l = 5, r = 5, t = 40, b = 5)
    )
    fig.update_xaxes(title_text='False Positive Rate', showgrid=False,showspikes=True)
    fig.update_yaxes(title_text='True Positive Rate', showgrid=False,showspikes=True)
    return fig, disable_sclt, disable_method, opt_table, inv_table

#another callback
@app.callback(
    [Output(component_id='auc_plot', component_property='figure'),
     #Output('loading1', 'parent_style'),
     Output(component_id='model_output', component_property='children'),
     Output(component_id='confusion_matrix', component_property='figure'),
     Output(component_id='ft_importance_graph', component_property='figure'),
     Output(component_id='table_variant_div', component_property='children'),
     Output(component_id='table_optimized_div', component_property='children'),
     Output(component_id='table_invariant_div', component_property='children'),
     Output(component_id='beta_slider_div', component_property = 'children'),
    ],
    [Input(component_id='slct_data', component_property = 'value'),
     Input(component_id='eng_df', component_property = 'value'),
     Input(component_id='slct_clf', component_property = 'value'),
     Input(component_id='set', component_property = 'value'),
     Input(component_id='thresh_slider', component_property = 'value'),
     Input(component_id='beta_slider_div', component_property = 'children'),
     Input(component_id='ft_sort_by', component_property = 'value'),
     Input(component_id='c_button', component_property = 'on'),
     inp_list_inputs]
)
def update_auc_pred(dataset, df_label, clf, split_set, slider_val, beta_slider_div, sort_by, c_button, f0):
    print('updating auc')
    beta_div_output = []
    dataset_label = dataset_dict[dataset]['label']
    dff = dataset_dict[dataset]['df']
    dff = dff[dff['Group'] != 3]
    df_id = eng_dfs_dict[df_label]
    cv_split = StratifiedKFold(n_splits=5)

    print('df_id:', df_id)
    print('clf:', clf)
    print('dataset_label', dataset_label)

    print(dff.columns)
    eng_feat_list = hf.read_from_json(cwd + path_to_datasets + dataset_label + df_id + '/features.json')
    feat_eng_df = hf.get_feat_eng_df(dff, eng_feat_list['features'])
    feat_eng_df.insert(0, 'Group', dff['Group']) #add group again because it is not a feature and will be deleted by hf.get_feat_eng_df
    print('ft eng df columns')
    print(feat_eng_df.columns)

    t_test_dict = {}
    for feat in feat_eng_df.columns[1:]:
        x = feat_eng_df[feat_eng_df['Group'] == 0]
        y = feat_eng_df[feat_eng_df['Group'] == 1]
        t_test, p_value = stats.ttest_ind(x[feat].to_numpy(), y[feat].to_numpy(), equal_var=False)
        t_test_dict[feat] = [t_test, p_value]
    t_test_df = pd.DataFrame.from_dict(t_test_dict).T
    #t_test_df = t_test_df.rename(columns = {'0':'t_score', '1':'p_value'})

    #if beta_slider == 2: beta_slider = 1024

    X = feat_eng_df.iloc[:,1:].to_numpy()
    y = feat_eng_df.iloc[:,0].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify = y)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    classifier = joblib.load(cwd + path_to_datasets + dataset_label + df_id + '/best_models/' + clf)
    cl_name = clf.split('.')[0]

    fig, df_opt, inv_df, ft_imp, actual_th = make_roc_figure(X_train, X_test, y_train, y_test, cv_split, classifier, cl_name, split_set, slider_val, c_button)
    fig.update_layout(graph_layout)
    fig.update_layout(legend=dict(
        yanchor="bottom",
        y=0.1,
        xanchor="right",
        x=0.99
        ),
    )
    fig.update_layout(height = 600, width=600,
        title_text= 'CV and Testset ROC Curves - ' + clf + ' - ' + dataset,
        margin=dict(l = 10, r = 10, t = 40, b = 5)
    )
    fig.update_xaxes(title_text='False Positive Rate', showgrid=False,showspikes=True)
    fig.update_yaxes(title_text='True Positive Rate', showgrid=False,showspikes=True)

    opt_table = dash_table.DataTable(
                    df_opt.to_dict('records'), [{"name": ['optimized data', i], "id": i} for i in df_opt.columns],
                    id='table_optimized',
                    style_header= data_table_style_header,
                    style_data = data_table_style_data,
                    merge_duplicate_headers=True,
                    )
    beta_slider = 1
    if len(beta_slider_div) != 0: beta_slider = beta_slider_div[1]['props']['value']
    if c_button: beta_div_output = [html.H5('beta value'),
                    dcc.Slider(0, 2, 0.1,
                    value=beta_slider,
                        marks={
                            0: '0',
                            1: 'equal',
                            2: '2',
                        },
                    id='beta_slider',
                    tooltip={"placement": "bottom", "always_visible": True}
                    ),]
    fig2, var_df = make_confusion_matrix(X_test, y_test, classifier, actual_th, beta_slider, c_button)
    fig2.update_layout(graph_layout)
    #fig2.for_each_trace(lambda t: t.update(textfont_color=colors[1], textfont_size=36))
    fig2.update_layout(
        title_text= 'Confusion Matrix',
        margin=dict(l = 5, r = 5, t = 40, b = 5)
    )

    var_table = dash_table.DataTable(
                    var_df.to_dict('records'), [{"name": ['variant data', i], "id": i} for i in var_df.columns],
                    id='table_variant',
                    style_header= data_table_style_header,
                    style_data = data_table_style_data,
                    merge_duplicate_headers=True,
                    )
    inv_table = dash_table.DataTable(
                    inv_df.to_dict('records'), [{"name": ['invariant data', i], "id": i} for i in inv_df.columns],
                    id='table_invariant',
                    style_header= data_table_style_header,
                    style_data = data_table_style_data,
                    merge_duplicate_headers=True,
                    )

    fig3 = make_feature_importances(ft_imp, feat_eng_df.columns[1:], t_test_df, sort_by)
    fig3.update_layout(graph_layout)
    fig3.update_layout(height = 350, width=600,
                    title_text= 'Feature Importance',
                    xaxis_title = '',
                    yaxis_title = 'Feature Imp. (mean)',
                    margin=dict(l = 5, r = 5, t = 40, b = 5))

    #features = feat_eng_df.columns
    dict_ = {features[0] : [2]}
    i = 0
    for feat in features[1:]:
        dict_[feat] = [f0[i]]
        i += 1
    df2 = pd.DataFrame(dict_)
    df2 = pd.concat([dff, df2])

    #model_clone = joblib.load(cwd + path_to_datasets + dataset_label + df_id + '/best_models/' + clf)
    feat_eng = hf.get_eng_values(df2.loc[df2['Group'] == 2], eng_feat_list['features']) #group = 2 is where our input patient is saved
    print(feat_eng)
    X_sample = scaler.transform(np.array(feat_eng).reshape(1, -1))
    class_index, conf = hf.get_proba(X_sample, classifier, slider_val)
    output = 'Model predicts Group ' + str(class_index) + ' with a %0.3f' %conf + ' certainty.'

    return fig, output, fig2, fig3, var_table, opt_table, inv_table, beta_div_output

#another callback
@app.callback(
    [Output(component_id='table_dataset_data_div', component_property='children'),
    ],
    [Input(component_id='slct_data', component_property = 'value'),
    ]
)
def update_dataset_data(dataset):
    dataset_label = dataset_dict[dataset]['label']
    dff = dataset_dict[dataset]['df']
    name = ['n_samples']
    value = [dff.shape[0]]
    for i in np.arange(0, 5):
        if dff['Group'][dff['Group'].isin([i])].empty == False:
            name.append('Group' + str(i))
            value.append(dff['Group'].value_counts()[i])
    descriptions =  ['', 'died/deteriorated', 'improvement', 'stagnant', 'unknown', 'unknown']
    #d = {'Name': ['n_samples', 'of Group0', 'of Group1', 'of Group3'], 'Value': [dff.shape[0], dff['Group'].value_counts()[0], dff['Group'].value_counts()[1], dff['Group'].value_counts()[3]]}
    d = {'Description' : descriptions[:len(name)],'Name' : name, 'Value' : value}
    data_df = pd.DataFrame(data=d)
    dataset_table = dash_table.DataTable(
                data_df.to_dict('records'), [{"name": ['Dataset', i], "id": i} for i in data_df.columns],
                id='table_dataset_data',
                style_header= data_table_style_header,
                style_data = data_table_style_data,
                merge_duplicate_headers=True,
                )
    return [dataset_table]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--port',
        type=str,
        help='port to be run on',
        required = True
    )

    args = parser.parse_args()

    port = 8001
    if args.port != None: port = args.port
    app.run_server(debug=True,
                  port=int(port),
                  host = '127.0.0.20')