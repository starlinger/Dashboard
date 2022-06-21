import pandas as pd
import numpy as np
import os
import joblib
import copy

from scripts.make_figures import make_roc_figure
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

from apps import data_exploration, default

'''
Model prediction
'''

colors = ['#37AA9C', '#00ccff', '#94F3E4']
colors_dict = {'background' : 'rgb(25, 25, 31)',
        'lightbg' : 'rgb(30, 30, 35)'}

cwd = os.getcwd()

# styling the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

colors_dict = {'background' : 'rgb(25, 25, 31)',
        'lightbg' : 'rgb(30, 30, 35)'}

graph_layout = {
        'font': {'color': 'rgba(255,255,255, 255)'},
        'plot_bgcolor': colors_dict['lightbg'],
        'paper_bgcolor':  colors_dict['lightbg']
    }

data_table_style_header = default.data_table_style_header
data_table_style_data = default.data_table_style_data


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

app = Dash(__name__, external_stylesheets = [dbc.themes.BOOTSTRAP])

dataset_dict = default.dataset_dict
df_label_list = default.df_label_list

#load models
#model_clone = joblib.load('assets/data/datasets/virus_pos_no_rep/df0/best_models/rdf.pkl')
#model_dict = {'pos_no_rep_rdf' : {'label' : 'No Replikation (positive)', 'model' : model_clone}}

model_dict = hf.get_models(path='assets/data/datasets/' + dataset_dict[list(dataset_dict.keys())[0]]['label'] + 'df0/best_models/')
eng_dfs_dict = hf.get_eng_dfs(path='assets/data/datasets/' + dataset_dict[list(dataset_dict.keys())[0]]['label'])

features = dataset_dict[list(dataset_dict.keys())[0]]['df'].columns
ft_rows = default.ft_rows

patiend_ids = np.arange(0, dataset_dict[list(dataset_dict.keys())[0]]['df'].shape[0])
dd_options = {}
for feat in features:
    # tmp = {'label': feat, 'value': feat}
    # dd_options.append(tmp)
    dd_options[feat] = feat
loading_style = {'position': 'absolute', 'align-self': 'center'}

logo_card2 = default.logo_card2

d = {'Name': ['Accuracy', 'balanced Accuracy', 'Youden', 'F1_Score', 'Precision', 'Recall (TPR)', 'Fall-Out (FPR)', 'Specitivity (TNR)', 'Cohen\'s Kappa', 'MCC'], 'Value': [3, 4, 4, 4, 4, 4, 4 , 4, 4, 4]}
var_df = pd.DataFrame(data=d)

d = {'Name': ['ROC AUC', 'Average Precision', 'Brier Loss'], 'Value': [3, 4, 5]}
invar_df = pd.DataFrame(data=d)

d = {'Name': ['Accuracy', 'Cohen\'s Kappa', 'MCC', 'Youden', 'F1_Score', 'F2_Score'], 'Treshold': [1, 2, 3, 4, 5, 6]}
opt_df = pd.DataFrame(data=d)

set_options = ['Testset', 'Mean CV Folds (Test)']

confusion_matrix_card = dbc.Card([
    dbc.CardBody([
        dbc.Row([
        dbc.Col([
            dbc.Row([
                dcc.Markdown('AI predicts: ', id = 'model_output', style = {'font-size':32}),
            ]),
            dbc.Row([
                dbc.Col([
                    html.H5('Engineered Dataframe'),
                    dcc.Dropdown(id="eng_df",
                        options= list(eng_dfs_dict.keys()),
                        multi=False,
                        clearable = False,
                        value = list(eng_dfs_dict.keys())[0],
                        style={'width': "99%"}
                    ),
                    html.Br(),
                    html.H5('AI Model'),
                    dcc.Dropdown(id="slct_clf",
                        options= list(model_dict.keys()),
                        multi=False,
                        clearable = False,
                        value = list(model_dict.keys())[1],
                        style={'width': "99%"}
                    ),
                    html.Br(),
                    html.H5('Dataset Split'),
                    dcc.Dropdown(id="set",
                        options= set_options,
                        multi=False,
                        clearable = False,
                        value = set_options[0],
                        style={'width': "99%"}
                    ),
                    html.Br(),
                    html.Div(id = 'table_optimized_div', children = [
                            dash_table.DataTable(
                            opt_df.to_dict('records'), [{"name": i, "id": i} for i in opt_df.columns],
                            id='table_optimized',
                            style_header= data_table_style_header,
                            style_data = data_table_style_data
                        )
                    ]),
                    html.Br(),
                    daq.PowerButton(
                        id = 'c_button',
                        on = False,
                        color = colors[0]
                    )
                ], width = 5),
                dbc.Col([
                    html.Div(id = 'table_variant_div', children = [
                        dash_table.DataTable(
                        var_df.to_dict('records'), [{"name": i, "id": i} for i in var_df.columns],
                        id='table_variant',
                        style_header= data_table_style_header,
                        style_data = data_table_style_data
                        )]
                    ),
                    html.Div(id = 'beta_slider_div', 
                        children = [
                        html.H5('beta value'),
                        dcc.Slider(0, 2, 0.1,
                        value=1,
                            marks={
                                0: '0',
                                1: 'equal',
                                2: '2',
                            },
                        id='beta_slider',
                        tooltip={"placement": "bottom", "always_visible": True}
                        ),
                    ])
                ], width = 4),
                dbc.Col([
                    html.Div(id = 'table_invariant_div', children = [
                        dash_table.DataTable(
                            invar_df.to_dict('records'), [{"name": i, "id": i} for i in invar_df.columns],
                            id='table_invariant',
                            style_header= data_table_style_header,
                            style_data = data_table_style_data
                        ),
                    ]),
                ], width = 3),
        ])
        ], width = 6),
        dbc.Col([
            dbc.Row([
                dcc.Graph(id='auc_plot',  figure={}),
                ]),
            dbc.Row([
                dcc.Graph(id='confusion_matrix',  figure={})
                ])
        ], width = 5),
        dbc.Col([
            html.Div([
                dcc.Slider(0, 1, 0.001,
                vertical = True,
                value=0.5,
                    marks={
                        0: '0',
                        0.1: '0.1',
                        0.2: '0.2',
                        0.3: '0.3',
                        0.4: '0.4',
                        0.5: '0.5',
                        0.6: '0.6',
                        0.7: '0.7',
                        0.8: '0.8',
                        0.9: '0.9',
                        1.0: '1',
                    },
                id='thresh_slider',
                verticalHeight = 900,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            ], style={'width': '19%','padding-left':'80%', 'padding-right':'1%'})
        ], width = 1)
        ])
    ]),
], color = colors_dict['lightbg'])

model_auc_card = dbc.Card([
    dbc.CardBody([
        dbc.Row([
        dbc.Col([
            dcc.Slider(0, 1, 0.001,
                vertical = True,
                value=0.652,
                    marks={
                        0: '1',
                        0.1: '0.9',
                        0.2: '0.8',
                        0.3: '0.7',
                        0.4: '0.6',
                        0.5: '0.5',
                        0.6: '0.4',
                        0.7: '0.3',
                        0.8: '0.2',
                        0.9: '0.1',
                        1.0: '0',
                    },
                id='thresh_slider',
                verticalHeight = 600,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], width = 1),
        dbc.Col([
            dcc.Graph(id='auc_plot',  figure={}),
        ], width = 3)
        ])
    ])
], color = colors_dict['lightbg'])

ft_sort_options = ['Mean', 't_score', 'p_value']
feature_importance_card = dbc.Card([
    dcc.Dropdown(id="ft_sort_by",
        options= ft_sort_options,
        multi=False,
        clearable = False,
        value = ft_sort_options[0],
        style={'width': "50%"}
    ),
    dcc.Graph(id='ft_importance_graph',  figure={})
], color = colors_dict['lightbg'])

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
        dbc.Col([
            logo_card2,
                ],width=3),
        dbc.Col(
            ft_rows,
            width = 4
        ),
        dbc.Col([
            feature_importance_card,
        ], width = 4),
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            confusion_matrix_card,
            ],width=12),
    ]),
], fluid = True
)