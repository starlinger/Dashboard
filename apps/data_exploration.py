import pandas as pd
import numpy as np
import os
import joblib
import copy

from scripts.get_dataframes import get_pos_no_rep, get_no_rep_all, get_pos_rep, get_virus_negative
import scripts.helper_functions as hf
from scipy.__config__ import show
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

import plotly.express as px  # (version 4.7.0 or higher)
import plotly.figure_factory as ff
from plotly.graph_objs import *
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table
import dash_daq as daq
import dash_bootstrap_components as dbc

from apps import model_prediction, default

'''
Main Dashboard script for displaying various plots regarding HeartBiopsi data. https://plotly.com/dash/
dash plotly bootstrap (Cols and Rows) for the Layout. https://dash-bootstrap-components.opensource.faculty.ai/
Data in .csv form on disk. Models exported in .pkl form on disk

Functionality
Display .csv data, input patient data, display input in relation, display prediction.
Multiple dataframes, multiple models
'''
#TODO
#1. daten normalisieren - done
#2. auc curve classifier mit threshold slider
#3. auc von feature (summen, ratios etc) - Done
#4. pipeline voll automatisieren
#5. multi page (with sidebar) - done
#6. save and import feature engineered columns

colors_dict = {'background' : 'rgb(25, 25, 31)',
        'lightbg' : 'rgb(30, 30, 35)'}

graph_layout = {
        'font': {'color': 'rgba(255,255,255, 255)'},
        'plot_bgcolor': colors_dict['lightbg'],
        'paper_bgcolor':  colors_dict['lightbg']
    }
cwd = os.getcwd()
colors = ['#37AA9C', '#00ccff', '#94F3E4']

data_table_style_header={
    'backgroundColor': 'rgb(30, 30, 30)',
    'color': colors[1]
}

data_table_style_data={
    'backgroundColor': 'rgb(50, 50, 50)',
    'color': colors[1]
}

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
    print(arrays)
    for i in np.arange(0, len(arrays), step = 2):
        tmp = arrays[i][int(len(arrays[i])/2)]
        arrays[i][int(len(arrays[i])/2)] = arrays[i+1][int(len(arrays[i+1])/2)-1]
        arrays[i+1][int(len(arrays[i+1])/2)-1] = tmp
    print(arrays)
    return np.concatenate(arrays, axis = 0)

def get_proba(f0, model):
    f0.extend([1.1, 4.3, 66.3, 1.2, 4.05, 67.3, 1.24, 28.2])
    custom = np.array([f0]).reshape(1, -1)
    confidences = model.predict_proba(custom)[0]
    id = np.argmax(confidences)
    return id, confidences[id]

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

#get data
dataset_dict = default.dataset_dict

# = dataset_dict[list(dataset_dict.keys())[0]]['df']
features = dataset_dict[list(dataset_dict.keys())[0]]['df'].columns
ft_rows = default.ft_rows

#patiend_ids = np.arange(0, df.shape[0])
dd_options = {}
for feat in features:
    # tmp = {'label': feat, 'value': feat}
    # dd_options.append(tmp)
    dd_options[feat] = feat
loading_style = {'position': 'absolute', 'align-self': 'center'}

logo_card2 = default.logo_card2

metric_dd_options = ['youden_th', 'f1_th', 'accuracy_th', 'mcc_th', 'cohens_th', 'mean']
radar_plot_card = dbc.Card([
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Label(daq.BooleanSwitch(id='axis_scale_switch', on=False, color = colors[0], label="logarithmic", style={'margin-bottom': 4},)),
            ], width = 5),
            dbc.Col([
                html.H5('Threshold'),
            ], width = 3),
            dbc.Col([
                dcc.Dropdown(id="metric_dd",
                    options = metric_dd_options,
                    value = metric_dd_options[0],
                    clearable = False,
                    multi=False,
                    style={'width': "95%"}
                ),
            ], width = 4),
        ]),
        dbc.Row([
            #dcc.Loading(id='loading1', parent_style=loading_style, children = [dcc.Graph(id='radar_plot', figure={})]),
            dcc.Graph(id='radar_plot', figure={}),
        ])
    ])
], body = True, color = colors_dict['lightbg'])

feat_figure_card = dbc.Card([
    dbc.CardBody([
        dcc.RadioItems(
                id='type',
                options=['perct Hist', 'Histogram', 'Density'],
                value='Density', inline=True
            ),
        dcc.RadioItems(
                id='distribution',
                options=['box', 'violin', 'rug', 'None'],
                value='None', inline=True,
            ),
        dbc.Row([
            dbc.Col([
                html.Div(id = 'density_plots_div_left', children = [
                    dcc.Graph(id='feature_plot0', figure={}),
                    dcc.Graph(id='feature_plot2', figure={}),
                ])
            ], width = 6),
            dbc.Col([
                html.Div(id = 'density_plots_div_right', children = [
                    dcc.Graph(id='feature_plot1', figure={}),
                    dcc.Graph(id='feature_plot3', figure={}),
                ])
            ], width = 6)
        ])
    ])
], color = colors_dict['lightbg'])

d = {'Name': ['Accuracy', 'Cohen\'s', 'MCC', 'Youden', 'F1_Score', 'Mean'], 'Treshold': [1, 2, 3, 4, 5, 6], 'Score' : [0.5, 0.5,  0.5, 0.5, 0.5, 0.5]}
opt_df = pd.DataFrame(data=d)

auc_plot_card = dbc.Card([
    dbc.CardBody([
        dbc.Row([html.H1("AUC for single feature, sum or ratio of features", style={'text-align': 'center'}),
                ]),
        dbc.Row([
            html.Label('Selected Features'),
                dcc.Dropdown(id="slct_auc_feat",
                    options = dd_options,
                    value = features[1:4],
                    #value= ['Group', 'EF  baseline (in %, numeric) '],
                    multi=True,
                style={'width': "95%"}),
            html.Label('Method'),
            dcc.Dropdown(id="combine_method",
                    options = ['Sum', 'Ratio'],
                    value = 'Sum',
                    multi=False,
                    clearable = False,
                style={'width': "45%"}),
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='auc_graph',  figure={})
                ], width = 7),
            dbc.Col([
                html.Div(id = 'auc_graph_opt_div', children = [
                            dash_table.DataTable(
                                opt_df.to_dict('records'), [{"name": i, "id": i} for i in opt_df.columns],
                                id='auc_graph_opt',
                                style_header= data_table_style_header,
                                style_data = data_table_style_data
                        )
                    ]),
                html.Div(id = 'auc_graph_inv_div', children = [
                            dash_table.DataTable(
                                opt_df.to_dict('records'), [{"name": i, "id": i} for i in opt_df.columns],
                                id='auc_graph_inv',
                                style_header= data_table_style_header,
                                style_data = data_table_style_data
                        )
                    ])
            ], width = 5)
        ]),
    ])
], color = colors_dict['lightbg'])

plot_card = dbc.Card([
    dcc.Graph(id='auc_graph2',  figure={})
])

# ------------------------------------------------------------------------------
# App layout
layout = dbc.Container([
    # html.Br(),
    # dbc.Row([
    #     #dbc.Col([html.H1('Title')], width = 7),
    #     #dbc.Col([radar_plot_card],width=5),
    # ]),
    # html.Br(),
    dbc.Row([
    dbc.Col([logo_card2,
        ],width=3),
    dbc.Col(
        # dbc.Row([
        #     html.H2('Value Input', style={'textAlign': 'center'}),]),
        # dbc.Row([
        #     dbc.Col([html.Label(feat_switches)],width=3),
        #     dbc.Col([html.Label(input_labels, style={'textAlign': 'right'})],width=6),
        #     dbc.Col([html.Label(inputs)],width=3),      
        # ])
        ft_rows
    ),
    dbc.Col([radar_plot_card],width=5)
    #dbc.Col([dcc.Loading(id='loading1', parent_style=loading_style, children = [dcc.Graph(id='radar_plot', figure={})])],width=5),
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([feat_figure_card],width=6),
        dbc.Col([auc_plot_card,
                html.Br(),
                #plot_card,
                ],width=6),
    ]),
    # html.Br(),
    # dbc.Row([
    # dbc.Col([],width=7),
    # dbc.Col(
    #     [],
    #     width=5),
    # ])
], fluid = True
)