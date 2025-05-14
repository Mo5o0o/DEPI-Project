import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
import numpy as np
import os
from datetime import datetime

# Data Configuration
DATA_CONFIG = {
    'file_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '1. data', 'processed', 'cleaned superstore dataset.csv'),
    'columns': {
        'order_date': 'Order Date',
        'sales': 'Sales',
        'region': 'Region',
        'category': 'Category',
        'subcategory': 'Sub-Category',
        'product': 'Product Name',
        'segment': 'Segment',
        'shipmode': 'Ship Mode',
        'customer': 'Customer Name',
        'country': 'Country',
        'profit': 'Profit',
        'quantity': 'Quantity',
        'discount': 'Discount',
        'state': 'State',
        'city': 'City'
    }
}

# Initialize the Dash app with custom theme
app = dash.Dash(__name__, 
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ]
)
server = app.server

# Custom color scheme
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#2ca02c',
    'accent': '#ff7f0e',
    'background': '#f8f9fa',
    'text': '#2c3e50'
}

def load_data():
    try:
        data_path = DATA_CONFIG['file_path']
        if not os.path.exists(data_path):
            print(f"Data file not found at: {data_path}")
            return None
            
        df = pd.read_csv(data_path)
        df['Order Date'] = pd.to_datetime(df[DATA_CONFIG['columns']['order_date']])
        df['Order Year'] = df['Order Date'].dt.year
        df['Order Month'] = df['Order Date'].dt.month
        df['Sales_log'] = np.log1p(df[DATA_CONFIG['columns']['sales']])
        df['Month Year'] = df['Order Date'].dt.strftime('%Y-%m')
        
        # Calculate additional metrics
        df['Revenue'] = df['Sales'] * (1 - df['Discount'])
        df['Profit Margin'] = (df['Profit'] / df['Sales']) * 100
        
        print("Data loaded successfully!")
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

# Load data
df = load_data()
if df is None:
    raise Exception("Failed to load data")

# Create the layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Superstore Sales Analytics Dashboard", 
                style={'color': COLORS['text'], 'textAlign': 'center'}),
        html.P("Interactive analytics and insights from superstore sales data",
               style={'textAlign': 'center', 'color': COLORS['text']})
    ], style={'padding': '20px', 'backgroundColor': 'white', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
    
    # Main Content
    html.Div([
        # Filters Section
        html.Div([
            html.H3("Filters", style={'color': COLORS['text']}),
            dcc.DatePickerRange(
                id='date-range',
                start_date=df['Order Date'].min(),
                end_date=df['Order Date'].max(),
                style={'marginBottom': '10px'}
            ),
            dcc.Dropdown(
                id='region-filter',
                options=[{'label': x, 'value': x} for x in df['Region'].unique()],
                multi=True,
                placeholder="Select Region(s)",
                style={'marginBottom': '10px'}
            ),
            dcc.Dropdown(
                id='category-filter',
                options=[{'label': x, 'value': x} for x in df['Category'].unique()],
                multi=True,
                placeholder="Select Category(s)",
                style={'marginBottom': '10px'}
            )
        ], style={'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '5px',
                  'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'marginBottom': '20px'}),
        
        # KPI Cards
        html.Div([
            html.Div([
                html.H4("Total Sales"),
                html.H2(id='total-sales', children=f"${df['Sales'].sum():,.2f}")
            ], className='kpi-card'),
            html.Div([
                html.H4("Total Profit"),
                html.H2(id='total-profit', children=f"${df['Profit'].sum():,.2f}")
            ], className='kpi-card'),
            html.Div([
                html.H4("Total Orders"),
                html.H2(id='total-orders', children=f"{len(df):,}")
            ], className='kpi-card'),
            html.Div([
                html.H4("Avg. Profit Margin"),
                html.H2(id='avg-margin', children=f"{df['Profit Margin'].mean():.1f}%")
            ], className='kpi-card')
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}),
        
        # Tabs for different analyses
        dcc.Tabs([
            # Sales Analysis Tab
            dcc.Tab(label='Sales Analysis', children=[
                html.Div([
                    html.H3("Sales Trends"),
                    dcc.Graph(id='sales-trend'),
                    html.H3("Regional Performance"),
                    dcc.Graph(id='regional-sales'),
                    html.H3("Top 10 Products"),
                    dcc.Graph(id='top-products')
                ])
            ]),
            
            # Product Analysis Tab
            dcc.Tab(label='Product Analysis', children=[
                html.Div([
                    html.H3("Category Performance"),
                    dcc.Graph(id='category-performance'),
                    html.H3("Sub-Category Analysis"),
                    dcc.Graph(id='subcategory-analysis'),
                    html.H3("Product Profitability"),
                    dcc.Graph(id='product-profitability')
                ])
            ]),
            
            # Customer Analysis Tab
            dcc.Tab(label='Customer Analysis', children=[
                html.Div([
                    html.H3("Customer Segments"),
                    dcc.Graph(id='customer-segments'),
                    html.H3("Top Customers"),
                    dcc.Graph(id='top-customers'),
                    html.H3("Customer Geography"),
                    dcc.Graph(id='customer-geography')
                ])
            ]),
            
            # Shipping Analysis Tab
            dcc.Tab(label='Shipping Analysis', children=[
                html.Div([
                    html.H3("Shipping Modes"),
                    dcc.Graph(id='shipping-modes'),
                    html.H3("Delivery Performance"),
                    dcc.Graph(id='delivery-performance')
                ])
            ]),
            
            # Profitability Analysis Tab
            dcc.Tab(label='Profitability', children=[
                html.Div([
                    html.H3("Profit Trends"),
                    dcc.Graph(id='profit-trends'),
                    html.H3("Margin Analysis"),
                    dcc.Graph(id='margin-analysis')
                ])
            ])
        ])
    ], style={'padding': '20px'})
])

# Add error handling decorator
def handle_callback_error(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {str(e)}")
            # Return an empty figure with error message
            return go.Figure().add_annotation(
                text=f"Error loading chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
    return wrapper

# Update callbacks with error handling
@app.callback(
    [Output('total-sales', 'children'),
     Output('total-profit', 'children'),
     Output('total-orders', 'children'),
     Output('avg-margin', 'children')],
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('region-filter', 'value'),
     Input('category-filter', 'value')]
)
@handle_callback_error
def update_kpi_cards(start_date, end_date, regions, categories):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['Order Date'] >= start_date) &
            (filtered_df['Order Date'] <= end_date)
        ]
    
    if regions:
        filtered_df = filtered_df[filtered_df['Region'].isin(regions)]
    
    if categories:
        filtered_df = filtered_df[filtered_df['Category'].isin(categories)]
    
    total_sales = f"${filtered_df['Sales'].sum():,.2f}"
    total_profit = f"${filtered_df['Profit'].sum():,.2f}"
    total_orders = f"{len(filtered_df):,}"
    avg_margin = f"{(filtered_df['Profit'].sum() / filtered_df['Sales'].sum() * 100):.1f}%"
    
    return total_sales, total_profit, total_orders, avg_margin

@app.callback(
    Output('sales-trend', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('region-filter', 'value'),
     Input('category-filter', 'value')]
)
@handle_callback_error
def update_sales_trend(start_date, end_date, regions, categories):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['Order Date'] >= start_date) &
            (filtered_df['Order Date'] <= end_date)
        ]
    
    if regions:
        filtered_df = filtered_df[filtered_df['Region'].isin(regions)]
    
    if categories:
        filtered_df = filtered_df[filtered_df['Category'].isin(categories)]
    
    monthly_sales = filtered_df.groupby('Month Year').agg({
        'Sales': 'sum',
        'Order Date': 'first'  # Keep the date for proper sorting
    }).reset_index()
    
    monthly_sales = monthly_sales.sort_values('Order Date')
    
    fig = px.line(monthly_sales, 
                  x='Month Year', 
                  y='Sales',
                  title='Monthly Sales Trend',
                  template='plotly_white')
    
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Sales ($)",
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

@app.callback(
    Output('subcategory-analysis', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('region-filter', 'value'),
     Input('category-filter', 'value')]
)
@handle_callback_error
def update_subcategory_analysis(start_date, end_date, regions, categories):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['Order Date'] >= start_date) &
            (filtered_df['Order Date'] <= end_date)
        ]
    
    if regions:
        filtered_df = filtered_df[filtered_df['Region'].isin(regions)]
    
    if categories:
        filtered_df = filtered_df[filtered_df['Category'].isin(categories)]
    
    subcategory_analysis = filtered_df.groupby(['Category', 'Sub-Category']).agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Quantity': 'sum'
    }).reset_index()
    
    fig = px.treemap(subcategory_analysis,
                     path=[px.Constant("All Categories"), 'Category', 'Sub-Category'],
                     values='Sales',
                     color='Profit',
                     title='Category and Sub-Category Analysis',
                     template='plotly_white',
                     color_continuous_scale='RdYlBu')
    
    fig.update_traces(textinfo="label+value")
    
    return fig

@app.callback(
    Output('customer-geography', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('category-filter', 'value')]
)
@handle_callback_error
def update_customer_geography(start_date, end_date, categories):
    filtered_df = df.copy()
    
    # State name to abbreviation mapping
    state_abbrev = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
        'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC',
        'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL',
        'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
        'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
        'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
        'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
        'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR',
        'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
        'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA',
        'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
    }
    
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['Order Date'] >= start_date) &
            (filtered_df['Order Date'] <= end_date)
        ]
    
    if categories:
        filtered_df = filtered_df[filtered_df['Category'].isin(categories)]
    
    # Add state abbreviations
    filtered_df['State_Code'] = filtered_df['State'].map(state_abbrev)
    
    # Aggregate data by state
    geo_data = filtered_df.groupby(['State', 'State_Code']).agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'count',
        'Customer Name': 'nunique'
    }).reset_index()
    
    # Calculate additional metrics
    geo_data['Avg Order Value'] = geo_data['Sales'] / geo_data['Order ID']
    geo_data['Profit Margin'] = (geo_data['Profit'] / geo_data['Sales'] * 100)
    
    # Create the choropleth map
    fig = go.Figure(data=go.Choropleth(
        locations=geo_data['State_Code'],
        z=geo_data['Sales'],
        locationmode='USA-states',
        colorscale='Viridis',
        colorbar_title="Sales ($)",
        text=geo_data['State'],  # State names for hover text
        customdata=np.stack((
            geo_data['Sales'],
            geo_data['Profit'],
            geo_data['Profit Margin'],
            geo_data['Customer Name'],
            geo_data['Order ID'],
            geo_data['Avg Order Value']
        ), axis=-1)
    ))

    # Update the layout
    fig.update_layout(
        title={
            'text': 'Sales Distribution by State',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        geo=dict(
            scope='usa',
            showlakes=True,
            lakecolor='rgb(255, 255, 255)',
            showland=True,
            landcolor='rgb(242, 242, 242)',
            showcoastlines=True,
            coastlinecolor='rgb(180, 180, 180)'
        ),
        template='plotly_white',
        height=600,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    # Add comprehensive hover template
    fig.update_traces(
        hovertemplate="<b>%{text}</b><br>" +
                      "Sales: $%{customdata[0]:,.0f}<br>" +
                      "Profit: $%{customdata[1]:,.0f}<br>" +
                      "Profit Margin: %{customdata[2]:.1f}%<br>" +
                      "Unique Customers: %{customdata[3]:.0f}<br>" +
                      "Total Orders: %{customdata[4]:.0f}<br>" +
                      "Avg Order Value: $%{customdata[5]:,.2f}<br>" +
                      "<extra></extra>"
    )
    
    return fig

@app.callback(
    Output('delivery-performance', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('region-filter', 'value')]
)
@handle_callback_error
def update_delivery_performance(start_date, end_date, regions):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['Order Date'] >= start_date) &
            (filtered_df['Order Date'] <= end_date)
        ]
    
    if regions:
        filtered_df = filtered_df[filtered_df['Region'].isin(regions)]
    
    # Calculate shipping days
    filtered_df['Ship Date'] = pd.to_datetime(filtered_df['Ship Date'])
    filtered_df['Shipping Days'] = (filtered_df['Ship Date'] - filtered_df['Order Date']).dt.days
    
    shipping_perf = filtered_df.groupby('Ship Mode').agg({
        'Shipping Days': ['mean', 'min', 'max'],
        'Order ID': 'count'
    }).reset_index()
    
    fig = go.Figure()
    
    for mode in shipping_perf['Ship Mode']:
        mode_data = shipping_perf[shipping_perf['Ship Mode'] == mode]
        fig.add_trace(go.Box(
            y=filtered_df[filtered_df['Ship Mode'] == mode]['Shipping Days'],
            name=mode,
            boxpoints='outliers'
        ))
    
    fig.update_layout(
        title='Shipping Days Distribution by Ship Mode',
        yaxis_title='Days to Ship',
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

# Callback for Regional Sales
@app.callback(
    Output('regional-sales', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('category-filter', 'value')]
)
def update_regional_sales(start_date, end_date, categories):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['Order Date'] >= start_date) &
            (filtered_df['Order Date'] <= end_date)
        ]
    
    if categories:
        filtered_df = filtered_df[filtered_df['Category'].isin(categories)]
    
    regional_sales = filtered_df.groupby('Region').agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()
    
    fig = px.bar(regional_sales, x='Region', y=['Sales', 'Profit'],
                 title='Sales and Profit by Region',
                 barmode='group',
                 template='plotly_white')
    
    return fig

# Callback for Top Products
@app.callback(
    Output('top-products', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('region-filter', 'value')]
)
def update_top_products(start_date, end_date, regions):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['Order Date'] >= start_date) &
            (filtered_df['Order Date'] <= end_date)
        ]
    
    if regions:
        filtered_df = filtered_df[filtered_df['Region'].isin(regions)]
    
    top_products = filtered_df.groupby('Product Name').agg({
        'Sales': 'sum',
        'Quantity': 'sum'
    }).sort_values('Sales', ascending=False).head(10).reset_index()
    
    fig = px.bar(top_products, x='Sales', y='Product Name',
                 title='Top 10 Products by Sales',
                 orientation='h',
                 template='plotly_white')
    
    return fig

# Callback for Category Performance
@app.callback(
    Output('category-performance', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('region-filter', 'value')]
)
def update_category_performance(start_date, end_date, regions):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['Order Date'] >= start_date) &
            (filtered_df['Order Date'] <= end_date)
        ]
    
    if regions:
        filtered_df = filtered_df[filtered_df['Region'].isin(regions)]
    
    category_perf = filtered_df.groupby('Category').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Quantity': 'sum'
    }).reset_index()
    
    fig = px.sunburst(filtered_df, 
                      path=['Category', 'Sub-Category'],
                      values='Sales',
                      title='Category and Sub-Category Performance',
                      template='plotly_white')
    
    return fig

# Callback for Customer Segments
@app.callback(
    Output('customer-segments', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('region-filter', 'value'),
     Input('category-filter', 'value')]
)
def update_customer_segments(start_date, end_date, regions, categories):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['Order Date'] >= start_date) &
            (filtered_df['Order Date'] <= end_date)
        ]
    
    if regions:
        filtered_df = filtered_df[filtered_df['Region'].isin(regions)]
    
    if categories:
        filtered_df = filtered_df[filtered_df['Category'].isin(categories)]
    
    segment_analysis = filtered_df.groupby('Segment').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Customer Name': 'nunique'
    }).reset_index()
    
    fig = px.pie(segment_analysis, 
                 values='Sales', 
                 names='Segment',
                 title='Sales Distribution by Customer Segment',
                 template='plotly_white',
                 hole=0.4)
    
    return fig

# Callback for Shipping Analysis
@app.callback(
    Output('shipping-modes', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('region-filter', 'value')]
)
def update_shipping_analysis(start_date, end_date, regions):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['Order Date'] >= start_date) &
            (filtered_df['Order Date'] <= end_date)
        ]
    
    if regions:
        filtered_df = filtered_df[filtered_df['Region'].isin(regions)]
    
    shipping_analysis = filtered_df.groupby(['Ship Mode', 'Category']).agg({
        'Sales': 'sum',
        'Order ID': 'count'
    }).reset_index()
    
    fig = px.bar(shipping_analysis, 
                 x='Category', 
                 y='Sales',
                 color='Ship Mode',
                 title='Sales by Shipping Mode and Category',
                 template='plotly_white',
                 barmode='group')
    
    return fig

# Callback for Profit Trends
@app.callback(
    Output('profit-trends', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('region-filter', 'value'),
     Input('category-filter', 'value')]
)
def update_profit_trends(start_date, end_date, regions, categories):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['Order Date'] >= start_date) &
            (filtered_df['Order Date'] <= end_date)
        ]
    
    if regions:
        filtered_df = filtered_df[filtered_df['Region'].isin(regions)]
    
    if categories:
        filtered_df = filtered_df[filtered_df['Category'].isin(categories)]
    
    profit_trend = filtered_df.groupby('Month Year').agg({
        'Profit': 'sum',
        'Sales': 'sum'
    }).reset_index()
    
    profit_trend['Profit Margin'] = (profit_trend['Profit'] / profit_trend['Sales']) * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=profit_trend['Month Year'],
        y=profit_trend['Profit'],
        name='Profit',
        line=dict(color=COLORS['primary'])
    ))
    
    fig.add_trace(go.Scatter(
        x=profit_trend['Month Year'],
        y=profit_trend['Profit Margin'],
        name='Profit Margin %',
        yaxis='y2',
        line=dict(color=COLORS['accent'], dash='dash')
    ))
    
    fig.update_layout(
        title='Profit and Margin Trends',
        template='plotly_white',
        yaxis2=dict(
            title='Profit Margin %',
            overlaying='y',
            side='right'
        ),
        hovermode='x unified'
    )
    
    return fig

@app.callback(
    Output('product-profitability', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('region-filter', 'value'),
     Input('category-filter', 'value')]
)
@handle_callback_error
def update_product_profitability(start_date, end_date, regions, categories):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['Order Date'] >= start_date) &
            (filtered_df['Order Date'] <= end_date)
        ]
    
    if regions:
        filtered_df = filtered_df[filtered_df['Region'].isin(regions)]
    
    if categories:
        filtered_df = filtered_df[filtered_df['Category'].isin(categories)]
    
    # Calculate product profitability metrics
    product_profit = filtered_df.groupby('Product Name').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Quantity': 'sum'
    }).reset_index()
    
    product_profit['Profit Margin'] = (product_profit['Profit'] / product_profit['Sales'] * 100)
    product_profit['Profit per Unit'] = product_profit['Profit'] / product_profit['Quantity']
    
    # Sort by profit and get top 20 products
    top_products = product_profit.nlargest(20, 'Profit')
    
    fig = px.scatter(top_products,
                     x='Sales',
                     y='Profit',
                     size='Quantity',
                     color='Profit Margin',
                     hover_name='Product Name',
                     hover_data=['Profit Margin', 'Profit per Unit'],
                     title='Top 20 Products - Profitability Analysis',
                     template='plotly_white',
                     color_continuous_scale='RdYlBu')
    
    fig.update_layout(
        xaxis_title="Total Sales ($)",
        yaxis_title="Total Profit ($)",
        coloraxis_colorbar_title="Profit Margin %"
    )
    
    return fig

@app.callback(
    Output('top-customers', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('region-filter', 'value'),
     Input('category-filter', 'value')]
)
@handle_callback_error
def update_top_customers(start_date, end_date, regions, categories):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['Order Date'] >= start_date) &
            (filtered_df['Order Date'] <= end_date)
        ]
    
    if regions:
        filtered_df = filtered_df[filtered_df['Region'].isin(regions)]
    
    if categories:
        filtered_df = filtered_df[filtered_df['Category'].isin(categories)]
    
    # Calculate customer metrics
    customer_analysis = filtered_df.groupby('Customer Name').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'nunique',
        'Quantity': 'sum'
    }).reset_index()
    
    customer_analysis['Avg Order Value'] = customer_analysis['Sales'] / customer_analysis['Order ID']
    customer_analysis['Profit Margin'] = customer_analysis['Profit'] / customer_analysis['Sales'] * 100
    
    # Get top 15 customers
    top_customers = customer_analysis.nlargest(15, 'Sales')
    
    fig = go.Figure()
    
    # Add bars for sales
    fig.add_trace(go.Bar(
        x=top_customers['Customer Name'],
        y=top_customers['Sales'],
        name='Sales',
        marker_color='#1f77b4'
    ))
    
    # Add bars for profit
    fig.add_trace(go.Bar(
        x=top_customers['Customer Name'],
        y=top_customers['Profit'],
        name='Profit',
        marker_color='#2ca02c'
    ))
    
    # Add line for profit margin
    fig.add_trace(go.Scatter(
        x=top_customers['Customer Name'],
        y=top_customers['Profit Margin'],
        name='Profit Margin %',
        yaxis='y2',
        line=dict(color='#ff7f0e', width=2)
    ))
    
    fig.update_layout(
        title='Top 15 Customers by Sales',
        xaxis_title="Customer",
        yaxis_title="Amount ($)",
        yaxis2=dict(
            title='Profit Margin %',
            overlaying='y',
            side='right'
        ),
        barmode='group',
        template='plotly_white',
        showlegend=True,
        xaxis_tickangle=-45,
        height=600
    )
    
    return fig

@app.callback(
    Output('margin-analysis', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('region-filter', 'value'),
     Input('category-filter', 'value')]
)
@handle_callback_error
def update_margin_analysis(start_date, end_date, regions, categories):
    filtered_df = df.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['Order Date'] >= start_date) &
            (filtered_df['Order Date'] <= end_date)
        ]
    
    if regions:
        filtered_df = filtered_df[filtered_df['Region'].isin(regions)]
    
    if categories:
        filtered_df = filtered_df[filtered_df['Category'].isin(categories)]
    
    # Calculate margins by category and sub-category
    margin_analysis = filtered_df.groupby(['Category', 'Sub-Category']).agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Discount': 'mean',
        'Quantity': 'sum'
    }).reset_index()
    
    margin_analysis['Profit Margin'] = margin_analysis['Profit'] / margin_analysis['Sales'] * 100
    margin_analysis['Revenue per Unit'] = margin_analysis['Sales'] / margin_analysis['Quantity']
    margin_analysis['Profit per Unit'] = margin_analysis['Profit'] / margin_analysis['Quantity']
    
    fig = px.sunburst(
        margin_analysis,
        path=['Category', 'Sub-Category'],
        values='Sales',
        color='Profit Margin',
        hover_data=['Profit Margin', 'Discount', 'Revenue per Unit', 'Profit per Unit'],
        title='Profit Margin Analysis by Category',
        template='plotly_white',
        color_continuous_scale='RdYlBu'
    )
    
    fig.update_layout(
        height=600,
        coloraxis_colorbar_title="Profit Margin %"
    )
    
    return fig

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Superstore Analytics Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: ''' + COLORS['background'] + ''';
                margin: 0;
                padding: 0;
            }
            .kpi-card {
                background-color: white;
                border-radius: 5px;
                padding: 20px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                flex: 1;
                margin: 0 10px;
            }
            .kpi-card h4 {
                color: ''' + COLORS['text'] + ''';
                margin: 0;
                font-size: 1.1em;
            }
            .kpi-card h2 {
                color: ''' + COLORS['primary'] + ''';
                margin: 10px 0 0 0;
                font-size: 1.8em;
            }
            .dash-tab {
                padding: 15px !important;
                font-size: 16px !important;
            }
            .dash-graph {
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin: 10px 0;
                padding: 15px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    # Use environment variable for port if available (for deployment)
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=False, host='0.0.0.0', port=port)