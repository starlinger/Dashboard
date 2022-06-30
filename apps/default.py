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
from sklearn.model_selection import train_test_split

import plotly.express as px  # (version 4.7.0 or higher)
import plotly.figure_factory as ff
from plotly.graph_objs import *
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table
import dash_daq as daq
import dash_bootstrap_components as dbc

from apps import data_exploration, model_prediction

'''
Main Dashboard script for displaying various plots regarding HeartBiopsi data. https://plotly.com/dash/
dash plotly bootstrap (Cols and Rows) for the Layout. https://dash-bootstrap-components.opensource.faculty.ai/
Data in .csv form on disk. Models exported in .pkl form on disk

Functionality
Display .csv data, input patient data, display input in relation, display prediction.
Multiple dataframes, multiple models
'''
cwd = os.getcwd()

paths = hf.read_from_json(cwd + '/paths.json')
path_to_csv = paths['path_to_csv']
path_to_datasets = paths['path_to_datasets']

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
        'lightbg' : 'rgb(30, 30, 35)',
        'darkgrey' : 'rgb(50, 50, 50)',
        'grey' : 'rgb(70, 70, 70)'}

graph_layout = {
        'font': {'color': 'rgba(255, 255, 255, 255)'},
        'plot_bgcolor': colors_dict['lightbg'],
        'paper_bgcolor':  colors_dict['lightbg']
    }

colors = ['#37AA9C', '#00ccff', '#94F3E4']

data_table_style_header={
    'backgroundColor': colors_dict['background'],
    'color': colors[0],
    'border': '1px solid ' + str(colors_dict['background'])
}
data_table_style_data={
    'backgroundColor': colors_dict['darkgrey'],
    'color': colors[1],
    'border': '1px solid ' + str(colors_dict['darkgrey'])
}

default_figure = go.Figure()
default_figure.update_layout(graph_layout)
default_figure.update_layout(height = 300, width=300,
        title_text='Default Figure (probably values missing)'
        )

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

#get data
dataset_dict = {}
for folder in os.listdir(cwd + path_to_datasets):
    if folder == 'virus_pos_no_rep':
        print('Virus Positive (No Rep)')
        df = get_pos_no_rep(path = cwd + path_to_csv + '2022-03-25_norep_all.csv')
        dataset_dict['Virus Positive (No Rep)'] = {'label' : folder + '/', 'df' : df}

        print('Virus Positive (No Rep) (reduced)')
        df_reduced = get_pos_no_rep(path = cwd + path_to_csv + '2022-03-25_norep_all.csv', reduced =True)
        dataset_dict['Virus Positive (No Rep) (reduced)'] = {'label' : folder + '/', 'df' : df_reduced}
    
    elif folder == 'virus_negative':
        print('Virus Negative')
        df_neg = get_virus_negative(path = cwd + path_to_csv + '2022-03-25_virusneg.csv')
        dataset_dict['Virus Negative'] =  {'label' : folder + '/', 'df' : df_neg}

        print('Virus Negative (reduced)')
        df_neg_reduced = get_virus_negative(path = cwd + path_to_csv + '2022-03-25_virusneg.csv', reduced =True)
        dataset_dict['Virus Negative (reduced)'] = {'label' : folder + '/', 'df' : df_neg_reduced}
    
    elif folder == 'no_rep_all':
        print('No Replikation (all)')
        df_no_rep_all = get_no_rep_all(path = cwd + path_to_csv + '2022-03-25_norep_all.csv')
        dataset_dict['No Replikation (all)'] =   {'label' : folder + '/', 'df' : df_no_rep_all}

        print('No Replikation (all) (reduced)')
        df_no_rep_all_reduced = get_no_rep_all(path = cwd + path_to_csv + '2022-03-25_norep_all.csv', reduced =True)
        dataset_dict['No Replikation (all) (reduced)'] =   {'label' : folder + '/', 'df' : df_no_rep_all_reduced}
    
    elif folder == 'virus_pos_rep':
        print('Virus Positive (Rep)')
        df_pos_rep = get_pos_rep(path = cwd + path_to_csv + '2022-05-30-B19_pos.csv')
        dataset_dict['Virus Positive (Rep)'] =  {'label' : folder + '/', 'df' : df_pos_rep}

        print('Virus Positive (Rep) (reduced)')
        df_pos_rep_reduced = get_pos_rep(path = cwd + path_to_csv + '2022-05-30-B19_pos.csv', reduced =True)
        dataset_dict['Virus Positive (Rep) (reduced)'] =  {'label' : folder + '/', 'df' : df_pos_rep_reduced}
    else: print('unknown folder')

scaler = MinMaxScaler()
X = dataset_dict['Virus Positive (No Rep)']['df'].iloc[:,1:].to_numpy()
y = dataset_dict['Virus Positive (No Rep)']['df'].iloc[:,0].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify = y)

print('X_train')
print(X_test[:,0])

X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print('scaled')
print(X_test_scaled[:,0])

df_label_list = []
for entry in dataset_dict:
    print(entry)
    df_label_list.append(dataset_dict[entry]['label'])

thresholds_df_dict = {}
for entry in dataset_dict:
    if entry[-9:] == '(reduced)':
        if os.path.isfile(cwd + path_to_datasets + dataset_dict[entry]['label'] + 'thresholds_reduced.csv'):
            df_reduced_th = pd.read_csv(cwd + path_to_datasets + dataset_dict[entry]['label'] + 'thresholds_reduced.csv')
            thresholds_df_dict[entry] = {'label': dataset_dict[entry]['label'], 'df' : df_reduced_th.copy()}
    else:
        if os.path.isfile(cwd + path_to_datasets + dataset_dict[entry]['label'] + 'thresholds.csv'):
            df_th = pd.read_csv(cwd + path_to_datasets + dataset_dict[entry]['label'] + 'thresholds.csv')
            thresholds_df_dict[entry] = {'label': dataset_dict[entry]['label'], 'df' : df_th.copy()}

#load models
model_clone = joblib.load(cwd + path_to_datasets + 'virus_pos_no_rep/df0/best_models/logreg.pkl')
model_dict = hf.get_models(path= cwd + path_to_datasets + dataset_dict[list(dataset_dict.keys())[0]]['label'] + 'df0/best_models/')
eng_dfs_dict = hf.get_eng_dfs(path= cwd + path_to_datasets + dataset_dict[list(dataset_dict.keys())[0]]['label'])

model_label_list = []
for entry in os.listdir(cwd + path_to_datasets + 'virus_pos_no_rep/df0/best_models/'):
    if entry[0] != '.':
        model_label_list.append(entry)

features = df.columns
ft_rows = [dbc.Row([html.H2('Value Input', style={'textAlign': 'center'})])]
input_labels = []
inputs = []
feat_switches = []
inp_list_inputs = []
inp_list_switches = []
i = 0
for feat in features[1:]:
    inp_list_inputs.append(Input(component_id='range'+str(i), component_property = 'value'))
    on = False
    dis = False
    if i < 4: on = True
    if i == 0: dis = True
    inp_list_switches.append(Input(component_id='feat_switch'+str(i), component_property = 'on'))
    ft_rows.append(dbc.Row([
            dbc.Col([html.Label(daq.BooleanSwitch(id='feat_switch' + str(i), on=on, color = colors[0], disabled = dis, style={'margin-bottom': 4},))],width=3),
            dbc.Col([html.Label(feat,
                style={'margin-bottom': 7,
                'font-size':15, 'textAlign': 'right'})],width=5),
            dbc.Col([html.Label(dcc.Input(id='range'+str(i), type='number', min=0, max=df[feat].max()*1.5, value = round(df[feat].mean(), 2), style={'width': "90%"}))],width=3),      
        ]))
    i += 1

#patiend_ids = np.arange(0, df.shape[0])
dd_options = {}
for feat in features:
    # tmp = {'label': feat, 'value': feat}
    # dd_options.append(tmp)
    dd_options[feat] = feat
loading_style = {'position': 'absolute', 'align-self': 'center'}

logo_card = dbc.Card([
    dbc.CardImg(src = '/assets/hth_logo.png', title = 'how to Health GmbH', top = True),
    dbc.CardBody([
        dbc.CardLink('howto.health', href= 'https://business.howto.health/', target = '_blank'),
        html.Label('E-Mail_data: business@howto.health'),
        html.Label('Phone: +49 (0)30 43722761')
    ])
], color = colors_dict['lightbg'])

d = {'Name': ['n_samples', 'of Group0', 'of Group1'], 'Value': [0, 0, 0]}
dataset_df = pd.DataFrame(data=d)

logo_card2 = dbc.Card([
    dbc.CardImg(src = '/assets/ikdt_logo2.png', title = 'ikdt_logo', top = True),
    dbc.CardBody([
        #html.Label('Datenanalyse von IKDTs Herzbiopsidaten durch howto.health'),
        #html.Br(),
        html.H5('Patient Dataset'),
        dcc.Dropdown(id="slct_data",
            options = list(dataset_dict.keys()),
            multi=False,
            clearable = False,
            value = list(dataset_dict.keys())[0],
            style={'width': "90%"}),
            html.Br(),
            html.Div(id = 'table_dataset_data_div', children = [
                    dash_table.DataTable(
                    dataset_df.to_dict('records'), [{"name": i, "id": i} for i in dataset_df.columns],
                    id='table_dataset_data',
                    style_header= data_table_style_header,
                    style_data = data_table_style_data
                )
            ]),
    ])
], color = colors_dict['lightbg'])

content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)

# ------------------------------------------------------------------------------
# App layout
layout = dbc.Container([
    dbc.Row([
        dbc.Col([logo_card])
    ])
], fluid = True
)