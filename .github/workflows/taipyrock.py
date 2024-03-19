# %%
from taipy import Gui
import requests
import pandas as pd
import datetime
import numpy as np
import random
from datetime import datetime
from datetime import timedelta
import squarify
import re

# Functions
def connect_airtable():
    api_key = api_key
    url = url
    headers = headers

    #request to airtable
    response = requests.get(url, headers=headers)
    data = response.json()
    records = []
    
    for record in data['records']:
        fields = record['fields']
        records.append(fields)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 9999)
    df = pd.DataFrame(records)
    return df


# %%
# Parameters for retrieving the airtable data
def get_data():
    df = pd.read_csv('orders.csv')
    #df = connect_airtable()
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    #data preproc
    #df['Project name (from Orders)'].fillna(df['Project name'], inplace=True)
    df['Project name (from Orders)'] = df['Project name']
    df['Name (from Customer)'] = df['Name (from Customer)'].astype(str).str.replace('[^a-zA-Z0-9]', ' ', regex=True).str.strip()
    df['Num (from Customer)'] = df['Num (from Customer)'].apply(lambda x: int(''.join(filter(str.isdigit, str(x).replace('[', '').replace(']', '')))) if pd.notna(x) else np.nan)
    # df.dropna(subset=['Num (from Customer)'], inplace=True)
    df['Bottle Type'] = df['Item description (from Order items)'].apply(lambda x: re.findall(r'[a-zA-Z]+', x.split('-')[0])[0] if isinstance(x, str) and '-' in x else None)
    df['Bottle Type'].fillna('N/A', inplace=True)
    df['Amount Unpaid'] = df['Total paid'] - df['Order total']
    # #df['Balance due quartile desc'] = pd.qcut(df['Balance due'], q=4, labels=['3', '2', '1'], duplicates='drop')
    # #df['Balance due quartile desc'] = pd.qcut(df['Balance due'], q=4, labels=['4', '3', '2', '1'])
    df['Number of Purchases'] = df.groupby('Num (from Customer)')['Num (from Customer)'].transform('count')
    df['Client Type'] = np.where(df['Number of Purchases'] == 1, 'New', 'Returning')
    df['Average Purchase Amount'] = df.groupby('Num (from Customer)')['Order total'].transform('mean')
    df['Created: project name'] = df['Created: project name'].astype(str)
    df['Clean Date'] = df['Created: project name'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ') if x != 'nan' else np.nan)
    # df = df.drop_duplicates(subset=['Num (from Customer)', 'Order num','Project name (from Orders)'], keep='last', inplace=True)
    # df = df.drop_duplicates(subset=['Num (from Customer)','Email (from Manager)','Order total','Created: project name','Date quote sent'], keep='last', inplace=True)
    df = df.replace(" ", np.NaN)
    # numeric_cols = df.select_dtypes(include=[np.number]).columns
    # df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    # #df = df.sort_values(by=['Num (from Customer)','Order num', 'Project name', 'Project name (from Orders)','Created: project name'], ascending=[True,True, True,True, True])
    df = df.sort_values(by=['Num (from Customer)','Order num', 'Project name','Clean Date'], ascending=[True,True, True, True])
    return df




# %%
#sales per month, per client and per project (total/balance due)
def sales_per_month():
    df = get_data()
    df['Clean Date'] = pd.to_datetime(df['Clean Date'])
    df['Month'] = df['Clean Date'].dt.strftime('%Y-%m')

    sales_by_month = df.groupby(['Month', 'Name (from Customer)', 'Num (from Customer)', 'Project name']).agg(
        Total_Sales=('Order total', 'sum'),
        Balance_Due=('Balance due', 'sum')
    ).reset_index()   
    return sales_by_month

def sales_sum_per_month():
    df = sales_per_month()
    sales_sum = df.groupby(['Month'])[['Total_Sales', 'Balance_Due']].sum().reset_index()
    return sales_sum


def project_summary():
    df = get_data()
    project_summary = df.groupby(['Name (from Customer)', 'Num (from Customer)']).agg(
        Project_Count=('Project name', 'nunique'),
        Total_Quantity=('Quantity (from Order items)', 'sum'),
        Total_Order=('Order total', 'sum'),
        Total_Cost=('Balance due to suppliers', 'sum')
    ).reset_index()
    project_summary = project_summary.sort_values(
        by=['Project_Count', 'Total_Order', 'Total_Cost', 'Total_Quantity'],
        ascending=[False, False, False, False]
    )
    
    return project_summary



def month_df():
    df = get_data()
    df['Month'] = df['Clean Date'].dt.strftime('%m-%Y')
    unique_months = df['Month'].unique()  # Get unique month-year combinations
    return unique_months

# %%
def customer_df():
    df = get_data()
    cust = sorted(df['Name (from Customer)'].unique())
    return cust


def fetch_project_details(month, cust):
    df = get_data()
    df = df[df['Name (from Customer)'] == cust]
    df['Month'] = df['Clean Date'].dt.strftime('%m-%Y')
    #filter for month
    if isinstance(month, str):
        month = [month]
    if month:
        df = df[df['Month'].isin(month)]

    project_details = df.groupby(['Project name']).agg(
        Total_Sales=('Order total', 'sum'),
        Balance_Due=('Balance due', 'sum')
    ).reset_index()
    
    return project_details

# Initialize variables
fetch_airtable = get_data()
sales_df = sales_per_month()
contributors = project_summary()

sales_sum = sales_sum_per_month()


layout={ "barmode": "stack" }


month_lov = month_df()
cust_lov = customer_df() 
selected_month = month_lov[0]
selected_cust = cust_lov[0] 

table = fetch_project_details(selected_month, selected_cust)

# Markdown

# Root page - page shared by everyone
root_md = """
<center><h2> Rock **KYC** </h2></center>

--------------------------------------------------------------------
<center><|navbar|></center>
"""

# Sales dashboard page
sales_dashboard_md = """
<h4> Sales Dashboard (CNY) </h4>

<|Refresh w Airtable|button|on_action=refresh_table|>

<|layout|columns=1 1 1|
<|{sales_sum}|chart|mode=lines|x=Month|y[1]=Total_Sales|y[2]=Balance_Due|line[2]=dash|color[1]=blue|>

<|Contributors|expandable|
    ...
<|{contributors}|table|>
    ...
|>

<|{contributors}|chart|type=bar|x=Name (from Customer)|y[1]=Total_Order|y[2]=Total_Cost|layout={layout}|>
|>

<|layout|columns=1 1|
Test select month and name
<|{selected_cust}|selector|lov={cust_lov}|dropdown|on_change=update_table|>
<|{selected_month}|selector|lov={month_lov}|dropdown|on_change=update_table|>

Render projects for <|{selected_cust}|> on <|{selected_month}|>   
<|{table}|table|show_all|>

<|{table}|chart|>
|>
"""

# Client segmentation page
client_segmentation_md = """
<h4> Client segmentation by behaviour & transaction </h4>

<|layout|columns = 1 5|
<h6>Search for a client (enter name):  </h6>
<|{selected_cust}|input|> 

|>
"""

# Cohort analysis page
cohort_analysis_md = """
<h4> Client cohort (behaviour evolution) analysis  </h4>
"""

# Callbacks
def refresh_table(state):
    state.fetch_airtable = get_data()
    state.sales_df = sales_per_month()
    state.contributors = project_summary()
    state.month_lov = month_df()
    state.cust_lov = customer_df()
    state.selected_month = state.month_lov[0]
    state.selected_cust = state.cust_lov[0]
    state.table = fetch_project_details(state.selected_month, state.selected_cust)


def update_table(state):
    state.table = fetch_project_details(state.selected_month, state.selected_cust)


# Putting pages in a dict
pages = {
    '/': root_md,
    'Sales-Dashboard': sales_dashboard_md,
    'Client-Segmentation': client_segmentation_md,
    'Cohort-Analysis': cohort_analysis_md
}

# Run
gui = Gui(pages=pages)
gui.run(dark_mode=False)