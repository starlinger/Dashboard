import dash
import dash_bootstrap_components as dbc

"""
This defines the dashboard server an app which is used by index.py
"""

# meta_tags are required for the app layout to be mobile responsive
app = dash.Dash(__name__, suppress_callback_exceptions=True,
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}],
                            external_stylesheets = [dbc.themes.BOOTSTRAP])
server = app.server