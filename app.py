import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import dash
from dash import dcc, html
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
from datetime import datetime as dt
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px

# model
from model import prediction
from sklearn.svm import SVR

app = dash.Dash(__name__, external_stylesheets=['assets/styles.css'])
server = app.server

# Define the layout components

# Navigation component
item1 = html.Div(
    [
        html.P("Welcome to the Stock Dash App!", className="start"),

        html.Div([
            # stock code input
            dcc.Input(id='stock-code', type='text', placeholder='Enter stock code'),
            html.Button('Submit', id='submit-button')
        ], className="stock-input"),

        html.Div([
            # Date range picker input
            dcc.DatePickerRange(
                id='date-range', start_date=dt(2020, 1, 1).date(), end_date=dt.now().date(), className='date-input')
        ]),
        html.Div([
            # Stock price button
            html.Button('Get Stock Price', id='stock-price-button'),

            # Indicators button
            html.Button('Get Indicators', id='indicators-button'),

            # Number of days of forecast input
            dcc.Input(id='forecast-days', type='number', placeholder='Enter number of days'),

            # Forecast button
            html.Button('Get Forecast', id='forecast-button')
        ], className="selectors")
    ],
    className="nav"
)

# Content component
item2 = html.Div(
    [
        html.Div(
            [
                html.Img(id='logo', className='logo'),
                html.H1(id='company-name', className='company-name')
            ],
            className="header"),
        html.Div(id="description"),
        html.Div([], id="graphs-content"),
        html.Div([], id="main-content"),
        html.Div([], id="forecast-content")
    ],
    className="content"
)

# Set the layout
app.layout = html.Div(className='container', children=[item1, item2])

# Callbacks

# Callback to update the data based on the submitted stock code
@app.callback(
    [
        Output("description", "children"),
        Output("logo", "src"),
        Output("company-name", "children"),
        Output("stock-price-button", "n_clicks"),
        Output("indicators-button", "n_clicks"),
        Output("forecast-button", "n_clicks")
    ],
    [Input("submit-button", "n_clicks")],
    [State("stock-code", "value")]
)
def update_data(n, val):
    if n is None:
        return None, None, None, None, None, None
    else:
        if val is None:
            raise PreventUpdate
        else:
            ticker = yf.Ticker(val)
            inf = ticker.info
            if 'logo_url' not in inf:
                return None, None, None, None, None, None
            else:
                name = inf['longName']
                logo_url = inf['logo_url']
                description = inf['longBusinessSummary']
                return description, logo_url, name, None, None, None


# Callback for displaying stock price graphs
@app.callback(
    Output("graphs-content", "children"),
    [
        Input("stock-price-button", "n_clicks"),
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date')
    ],
    [State("stock-code", "value")]
)
def stock_price(n, start_date, end_date, val):
    if n is None or n == 0:
        return ""
    if val is None:
        raise PreventUpdate
    else:
        if start_date is not None:
            df = yf.download(val, str(start_date), str(end_date))
        else:
            df = yf.download(val)

    if df.empty:
        return html.Div("No data available for the selected stock and date range.")

    df.reset_index(inplace=True)
    fig = px.line(df, x="Date", y=["Close", "Open"], title="Closing and Opening Price vs Date")
    return dcc.Graph(figure=fig)


# Callback for displaying indicators
@app.callback(
    Output("main-content", "children"),
    [
        Input("indicators-button", "n_clicks"),
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date')
    ],
    [State("stock-code", "value")]
)
def indicators(n, start_date, end_date, val):
    if n is None or n == 0:
        return ""
    if val is None:
        return ""

    if start_date is None:
        df_more = yf.download(val)
    else:
        df_more = yf.download(val, str(start_date), str(end_date))

    if df_more.empty:
        return html.Div("No data available for the selected stock and date range.")

    df_more.reset_index(inplace=True)
    fig = get_more(df_more)
    return dcc.Graph(figure=fig)


def get_more(df):
    # Calculate Exponential Weighted Moving Average (EWA)
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # Calculate Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Calculate Bollinger Bands
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['20dSTD'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['MA20'] + (df['20dSTD'] * 2)
    df['Lower'] = df['MA20'] - (df['20dSTD'] * 2)

    # Create subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        subplot_titles=('EWA 20', 'RSI', 'Bollinger Bands'), row_width=[0.2, 0.2, 0.6])

    # Add EWA
    fig.add_trace(go.Scatter(x=df['Date'], y=df['EWA_20'], mode='lines', name='EWA 20'), row=1, col=1)

    # Add RSI
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI'), row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)

    # Add Bollinger Bands
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Upper'], mode='lines', name='Upper Band',
                             line=dict(color='rgba(255, 0, 0, 0.5)')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Lower'], mode='lines', name='Lower Band',
                             line=dict(color='rgba(0, 0, 255, 0.5)')), row=3, col=1)

    return fig


# Callback for displaying forecast
@app.callback(
    Output("forecast-content", "children"),
    [Input("forecast-button", "n_clicks")],
    [State("forecast-days", "value"),
     State("stock-code", "value")]
)
def forecast(n, n_days, val):
    if n is None or n == 0:
        return ""
    if val is None:
        raise PreventUpdate

    if n_days is None or n_days <= 0:
        return html.Div("Please enter a valid number of days for forecast.")

    fig, mae, mse, rmse, r2 = prediction(val, int(n_days) + 1)

    # Create the metrics as HTML components
    metrics = [
        html.P(f"Mean Absolute Error (MAE): {mae:.2f}"),
        html.P(f"Mean Squared Error (MSE): {mse:.2f}"),
        html.P(f"Root Mean Squared Error (RMSE): {rmse:.2f}"),
        html.P(f"R-squared (R2): {r2:.2f}")
    ]

    # Return a single element containing both the graph and metrics
    return html.Div([dcc.Graph(figure=fig,style={'width': '600px'})] + metrics)


if __name__ == '__main__':
    app.run_server(debug=True)
