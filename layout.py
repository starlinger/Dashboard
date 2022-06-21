app.layout = dbc.Container([
    # html.Br(),
    # dbc.Row([
    #     #dbc.Col([html.H1('Title')], width = 7),
    #     #dbc.Col([radar_plot_card],width=5),
    # ]),
    # html.Br(),
    dbc.Row([
    dbc.Col([logo_card,
        html.Br(),
        dcc.RadioItems(
                id='type',
                options=['Histogram', 'Density'],
                value='Density', inline=True
            ),
        dcc.RadioItems(
                id='distribution',
                options=['box', 'violin', 'rug', 'None'],
                value='rug', inline=True
            ),],width=3),
    dbc.Col([
        dbc.Row([
            html.H2('Value Input', style={'textAlign': 'center'}),]),
        dbc.Row([
            dbc.Col([html.Label(feat_switches)],width=3),
            dbc.Col([html.Label(input_labels, style={'textAlign': 'right'})],width=6),
            dbc.Col([html.Label(inputs)],width=3),      
        ])
    ]),
    dbc.Col([radar_plot_card],width=5)
    #dbc.Col([dcc.Loading(id='loading1', parent_style=loading_style, children = [dcc.Graph(id='radar_plot', figure={})])],width=5),
    ]),
    html.Br(),
    dbc.Row([
    dbc.Col([feat_figure_card],width=7),
    dbc.Col([model_predidction_card,
            html.Br(),
            plot_card],width=5),
    ]),
    html.Br(),
    dbc.Row([
    dbc.Col([],width=7),
    dbc.Col(
        [],
        width=5),
    ])
], fluid = True
)