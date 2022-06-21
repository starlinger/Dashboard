import pandas as pd
import numpy as np
import os
import joblib
import copy

from scripts.get_dataframes import get_no_rep_all
import scripts.helper_functions as hf
from scipy.__config__ import show
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

import plotly.express as px  # (version 4.7.0 or higher)
import plotly.figure_factory as ff
from plotly.graph_objs import *
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback_context # pip install dash (version 2.0.0 or higher)
import dash_daq as daq
import dash_bootstrap_components as dbc

'''
Main Dashboard script for displaying various plots regarding HeartBiopsi data. https://plotly.com/dash/
dash plotly bootstrap (Cols and Rows) for the Layout. https://dash-bootstrap-components.opensource.faculty.ai/
Data in .csv form on disk. Models exported in .pkl form on disk

Functionality
Display .csv data, input patient data, display input in relation, display prediction.
Multiple dataframes, multiple models
'''
#TODO
#1. daten normalisieren
#2. auc curve classifier, auc von feature (summen, ratios etc)
#3. pipeline voll automatisieren
#4. multi page (with sidebar)
#5. save and import feature engineered columns

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
cwd = os.getcwd()
colors = ['#37AA9C', '#00ccff', '#94F3E4']

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
df = get_no_rep_all(path = cwd + '/assets/data/2022-03-25_norep_all.csv')
df_dict = {'no_rep_all' : {'label' : 'No Replikation (all)', 'df' : df},
            }

df_label_list = []
for entry in df_dict:
    df_label_list.append(df_dict[entry]['label'])
#load models
model_clone = joblib.load('assets/data/virus_pos_no_rep_cl.pkl')
model_dict = {'pos_no_rep_cl' : {'label' : 'No Replikation (positive)', 'model' : model_clone}}

model_label_list = []
for entry in model_dict:
    model_label_list.append(model_dict[entry]['label'])

features = df.columns
input_labels = []
inputs = []
feat_switches = []
inp_list_inputs = []
inp_list_switches = []
i = 0
for feat in features[1:]:
    input_labels.append(
        html.Label(feat,
            style={'margin-bottom': 7,
            'font-size':15},)
        )
    inputs.append(dcc.Input(id='range'+str(i), type='number', min=0, max=df[feat].max()*1.5, value = round(df[feat].mean(), 2), style={'width': "90%"}))
    inp_list_inputs.append(Input(component_id='range'+str(i), component_property = 'value'))
    on = False
    dis = False
    if i < 4: on = True
    if i == 0: dis = True
    #feat_switches.append(daq.ToggleSwitch(id='feat_switch' + str(i), value = on, style={'margin-bottom': 4},))
    feat_switches.append(daq.BooleanSwitch(id='feat_switch' + str(i), on=on, color = colors[0], disabled = dis, style={'margin-bottom': 4},))
    inp_list_switches.append(Input(component_id='feat_switch'+str(i), component_property = 'on'))
    i += 1

patiend_ids = np.arange(0, df.shape[0])
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
        html.Label('E-Mail: business@howto.health'),
        html.Label('Phone: +49 (0)30 43722761')
    ])
], color = colors_dict['lightbg'])

radar_plot_card = dbc.Card([
    dbc.CardBody([
        #dcc.Loading(id='loading1', parent_style=loading_style, children = [dcc.Graph(id='radar_plot', figure={})]),
        dcc.Graph(id='radar_plot', figure={})
    ])
], body = True, color = colors_dict['lightbg'])

feat_figure_card = dbc.Card([
    dcc.Graph(id='feature_plots', figure={})
], color = colors_dict['lightbg'])

model_predidction_card = dbc.Card([
    dbc.CardBody([
        dcc.Markdown('AI predicts: ', id = 'model_output', style = {'font-size':36}),
        #dcc.Loading(id='loading2', parent_style=loading_style, children = [dcc.Graph(id='auc_graph',  figure={})])
        html.H5('Patient Data'),
        dcc.Dropdown(id="slct_data",
            options = df_label_list,
            multi=False,
            clearable = False,
            value = df_label_list[0],
            style={'width': "80%"}
        ),
        html.Br(),
        html.H5('AI Model'),
        dcc.Dropdown(id="slct_clf",
            options= model_label_list,
            multi=False,
            clearable = False,
            value = model_label_list[0],
            style={'width': "80%"}
        ),
    ])
], color = colors_dict['lightbg'])

plot_card = dbc.Card([
    dcc.Graph(id='auc_graph',  figure={})
])

sidebar = html.Div(
    [
        html.H2("Sidebar", className="display-4"),
        html.Hr(),
        html.P(
            "Number of students per education level", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Page 1", href="/page-1", active="exact"),
                dbc.NavLink("Page 2", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)

# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content
])

@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/":
        return [
                html.H1('Kindergarten in Iran',
                        style={'textAlign':'center'}),
                dcc.Graph(id='bargraph',
                         figure=px.bar(df, barmode='group', x='Years',
                         y=['Girls Kindergarten', 'Boys Kindergarten']))
                ]
    elif pathname == "/page-1":
        return [
                html.H1('Grad School in Iran',
                        style={'textAlign':'center'}),
                dcc.Graph(id='bargraph',
                         figure=px.bar(df, barmode='group', x='Years',
                         y=['Girls Grade School', 'Boys Grade School']))
                ]
    elif pathname == "/page-2":
        return [
                html.H1('High School in Iran',
                        style={'textAlign':'center'}),
                dcc.Graph(id='bargraph',
                         figure=px.bar(df, barmode='group', x='Years',
                         y=['Girls High School', 'Boys High School']))
                ]
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )
# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    [Output(component_id='feature_plots', component_property='figure')],
    [Input(component_id='type', component_property = 'value'),
    Input(component_id='distribution', component_property = 'value'),
    inp_list_inputs, inp_list_switches]
)
def update_graph(plot_type, option_dist, f0, fs0):
    option_slctd = []
    i = 1       #only show feature graphs for true switches
    for sw in fs0:
        if sw: option_slctd.append(features[i])
        i += 1
    
    show_rugs = True
    if option_dist == 'None': show_rugs = False
    rows, cols, specs = make_specs(len(option_slctd), max_cols= 2,  rugs = show_rugs,)
    titles = make_titles(option_slctd, cols, show_rugs)
    if show_rugs:
        row_hgts = [0.9, 0.1]
        row_hgts = np.tile(row_hgts, int(rows/2)).tolist()
    else: row_hgts = np.tile(0.5, rows).tolist()
    #print('n rows:', rows)
    #print('n cols:', cols)
    print(specs)
    print(titles)
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=specs,
        subplot_titles=titles,
        row_heights=row_hgts,
        vertical_spacing=0.1,
    )

    #colors = ['#37AA9C', '#00ccff', '#94F3E4']
    row_indizes = np.repeat(np.arange(1, rows+1), cols)         # = [1,1,1,2,2,2,3,3,3]
    col_indizes = np.tile(np.arange(1, cols+1), rows)           # = [1,2,3,1,2,3,1,2,3]

    if show_rugs:                                               # = [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6]
        row_indizes = switch_middle(row_indizes)                # = [1,2,1,2,1,2,3,4,3,4,3,4,5,6,5,6,5,6]
        col_indizes = np.repeat(col_indizes, 2)                 # = [1,1,2,2,3,3,1,1,2,2,3,3,1,1,2,2,3,3]
    print('rows:', row_indizes)
    print('cols:', col_indizes)
    #make df from inputs
    dict_ = {features[0] : [2]}
    i = 0
    for feat in features[1:]:
        dict_[feat] = [f0[i]]
        i += 1
    df2 = pd.DataFrame(dict_)
    df2 = pd.concat([df, df2])
    #print(df2.tail())
    feat_index = 0
    for feat in option_slctd:
        dff = df2.copy()
        chosen = ['Group', feat]
        dff = dff[chosen]
        x = dff[dff['Group'] == 0]
        y = dff[dff['Group'] == 1]
        z = dff[dff['Group'] == 2]
        group_labels = ['Group 0', 'Group 1']

        hist_data = [x[feat], y[feat]]
        #print('z[', feat ,']:', z[feat].iloc[0])
        if plot_type == 'Density' or plot_type != 'Density':
            fig_tmp = ff.create_distplot(hist_data,
                                        group_labels,
                                        show_hist=False,
                                        colors = colors)
        # else:
        #     fig_tmp = px.histogram(
        #         dff, x=option_slctd, color = 'Group',
        #         color_discrete_sequence= colors,
        #         title= 'Histogram (' + option_slctd + ')',
        #         marginal=option_dist,
        #         hover_data=dff.columns
        #         )
        # Add subplots
        show_l = False
        if feat_index == 99: show_l = True 
        fig.add_trace(
            go.Scatter(fig_tmp['data'][0],
                           #marker_color='blue'
                           showlegend=show_l,
                          ),
            row=row_indizes[feat_index], col=col_indizes[feat_index]
        )
        fig.add_trace(
            go.Scatter(fig_tmp['data'][1],
                           #marker_color='blue'
                           showlegend=show_l,
                          ),
            row=row_indizes[feat_index], col=col_indizes[feat_index]
        )
        fig.update_yaxes(showgrid=False, row=row_indizes[feat_index],col=col_indizes[feat_index])
        fig.update_xaxes(showgrid=False, row=row_indizes[feat_index],col=col_indizes[feat_index])
        fig.add_vline(
            x=z[feat].iloc[0], line_width=1.5,
            #line_dash="dash",
            line_color="yellow"
            , row=row_indizes[feat_index], col=col_indizes[feat_index])
        # fig.add_shape(type="line",
        #             x0=z[feat].iloc[0], y0=0, x1=z[feat].iloc[0], y1=2,
        #             line=dict(color="RoyalBlue",width=3),
        #             row=row_indizes[feat_index], col=col_indizes[feat_index]
        #     )
        #make rugs
        if show_rugs == True:
            feat_index += 1
            # rug / margin plot to immitate ff.create_distplot
            dff['rug 1'] = 1.1
            dff['rug 2'] = 1
            fig.add_trace(go.Scatter(x=hist_data[0], y = dff['rug 1'],
                                mode = 'markers',
                                showlegend=False,
                                marker=dict(color = colors[0], symbol='line-ns-open')
                                    ), row=row_indizes[feat_index], col=col_indizes[feat_index])

            fig.add_trace(go.Scatter(x=hist_data[1], y = dff['rug 2'],
                                mode = 'markers',
                                showlegend=False,
                                marker=dict(color = colors[1], symbol='line-ns-open')
                                    ), row=row_indizes[feat_index], col=col_indizes[feat_index])
            fig.update_yaxes(showgrid=False, range=[0.95,1.15], tickfont=dict(color='rgba(0,0,0,0)', size=14), row=row_indizes[feat_index],col=col_indizes[feat_index])
            fig.update_xaxes(showgrid=False, visible=False, showticklabels=False, row=row_indizes[feat_index],col=col_indizes[feat_index])
        feat_index += 1

    fig.update_layout(graph_layout)
    height = 600
    if show_rugs:   height = (120 + (112* rows)) + (30 + (12 * rows))
    else:           height = (120 + (225 * rows))
    if cols == 1: height = 600
    print('height:', height)
    width = 725
    fig.update_layout(height = height, width=width, margin=dict(l = 75, r = 0, t = 40, b = 0))
    print('type:', type(fig))
    return [fig]

#another callback
@app.callback(
    [Output(component_id='radar_plot', component_property='figure'),
     #Output('loading1', 'parent_style'),
     Output(component_id='model_output', component_property='children')
    ],
    [Input(component_id='type', component_property = 'value'),
    inp_list_inputs]
)
def update_graph2(d, f0):
    #make radar plot
    dff = df.copy()
    maxs = dff.max(axis =1)
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
    values = df_scaled.loc[0].values.flatten().tolist()[1:]
    print('values:', values)
    df_tmp = pd.DataFrame(dict(
        #r = maxs,
        #range_r = [[0, 40], [0, 20], [0, 22], [0, 22], [0, 22], [0, 22], [0, 22], [0, 22], [0, 22]],
        r = values,
        theta = df_scaled.columns[1:]))
    fig2 = px.line_polar(df_tmp, r='r', theta='theta', line_close=True)

    print('f0:', np.array(f0))
    f0_scaled = scaler.fit_transform(np.array(f0).reshape(1,-1))
    print('f0_scaled:', f0_scaled)
    
    col_tmp = df_scaled.columns[1:].tolist()
    labels = []
    for i in np.arange(0, len(col_tmp)):
        labels.append(col_tmp[i].split('(')[0])
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r= dff.loc[0].values.flatten().tolist()[1:],
        theta=labels,
        #fill='toself',
        name='Single Sample'
    ))
    fig.add_trace(go.Scatterpolar(
        r=f0,
        theta=labels,
        fill='toself',
        name='Custom',
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        #range=[0, 1]
        )),
    showlegend=False
    )

    class_index, conf = get_proba(f0, model_clone)
    output = 'Model predicts Group ' + str(class_index) + ' with a %0.3f' %conf + ' certainty.'

    fig.update_layout(graph_layout)
    #fig2.update_traces(line_close=True)
    fig.update_layout(height = 300, width=475,
        #title_text="Feature Values in Relation to Cutoff Points"
        )
    fig.update_layout(margin=dict(l = 0, r = 0, t = 12, b = 15))
    new_loading_style = loading_style
    return fig, output
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True,
                  port=8001,
                  host = '127.0.0.20')