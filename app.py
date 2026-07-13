"""
Sales Forecasting & Demand Intelligence Dashboard
Internship Project — Week 3 & 4
Run: streamlit run app.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# ── Page config ──
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──
st.markdown("""
<style>
    .metric-box {
        background: #f0f4ff;
        border-left: 4px solid #2E5C8A;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 6px 0;
    }
    .big-number { font-size: 2rem; font-weight: bold; color: #2E5C8A; }
    .section-header { color: #2E5C8A; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Load & cache data ──
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv', encoding='latin1')
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['Ship Date']  = pd.to_datetime(df['Ship Date'],  dayfirst=True)
    df['Year']       = df['Order Date'].dt.year
    df['Month']      = df['Order Date'].dt.month
    df['Quarter']    = df['Order Date'].dt.quarter
    df['ShipDays']   = (df['Ship Date'] - df['Order Date']).dt.days

    monthly = (df.groupby(df['Order Date'].dt.to_period('M'))['Sales']
                 .sum().reset_index())
    monthly['Order Date'] = monthly['Order Date'].dt.to_timestamp()
    monthly.columns = ['ds', 'y']

    weekly = (df.groupby(df['Order Date'].dt.to_period('W'))['Sales']
                .sum().reset_index())
    weekly['Order Date'] = weekly['Order Date'].dt.to_timestamp()
    weekly.columns = ['ds', 'y']

    return df, monthly, weekly

df, monthly_sales, weekly_sales = load_data()

# ── Sidebar navigation ──
st.sidebar.image("https://img.icons8.com/color/96/combo-chart--v1.png", width=60)
st.sidebar.title("Sales Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    ["Sales Overview",
     "Forecast Explorer",
     "Anomaly Report",
     "Product Demand Segments"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** Superstore Sales")
st.sidebar.markdown(f"**Records:** {len(df):,}")
st.sidebar.markdown(f"**Period:** 2015 – 2018")

# PAGE 1 — SALES OVERVIEW

if page == "Sales Overview":
    st.title("Sales Overview Dashboard")
    st.markdown("High-level view of company sales performance across years, categories, and regions.")

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue",    f"${df['Sales'].sum()/1e6:.2f}M")
    col2.metric("Total Orders",     f"{len(df):,}")
    col3.metric("Avg Order Value",  f"${df['Sales'].mean():.0f}")
    col4.metric("Avg Ship Days",    f"{df['ShipDays'].mean():.1f} days")

    st.markdown("---")

    # Filters
    col_f1, col_f2 = st.columns(2)
    years      = st.sidebar.multiselect("Filter by Year",
                    sorted(df['Year'].unique()),
                    default=sorted(df['Year'].unique()))
    categories = st.sidebar.multiselect("Filter by Category",
                    df['Category'].unique(),
                    default=df['Category'].unique().tolist())

    filtered = df[df['Year'].isin(years) & df['Category'].isin(categories)]

    # Chart 1 — Total sales by year
    st.subheader("Total Sales by Year")
    yearly = filtered.groupby('Year')['Sales'].sum().reset_index()
    fig_yr = px.bar(yearly, x='Year', y='Sales',
                    color='Year', color_continuous_scale='Blues',
                    labels={'Sales': 'Total Sales ($)'},
                    text_auto='.2s')
    fig_yr.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_yr, use_container_width=True)

    # Chart 2 — Monthly trend
    st.subheader("Monthly Sales Trend")
    monthly_filtered = (filtered.groupby(filtered['Order Date'].dt.to_period('M'))['Sales']
                                .sum().reset_index())
    monthly_filtered['Order Date'] = monthly_filtered['Order Date'].dt.to_timestamp()

    fig_mo = go.Figure()
    fig_mo.add_trace(go.Scatter(
        x=monthly_filtered['Order Date'],
        y=monthly_filtered['Sales'],
        mode='lines+markers',
        line=dict(color='steelblue', width=2),
        fill='tozeroy', fillcolor='rgba(70,130,180,0.1)',
        name='Monthly Sales'
    ))
    fig_mo.update_layout(height=350,
                          yaxis_title='Sales ($)',
                          xaxis_title='Date')
    st.plotly_chart(fig_mo, use_container_width=True)

    # Chart 3 & 4 side by side — Region and Category
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Sales by Region")
        reg = filtered.groupby('Region')['Sales'].sum().reset_index()
        fig_reg = px.pie(reg, values='Sales', names='Region',
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         hole=0.4)
        fig_reg.update_layout(height=350)
        st.plotly_chart(fig_reg, use_container_width=True)

    with col_r:
        st.subheader("Sales by Category")
        cat = filtered.groupby('Category')['Sales'].sum().reset_index()
        fig_cat = px.bar(cat, x='Category', y='Sales',
                         color='Category',
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig_cat.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_cat, use_container_width=True)

    # Sub-category heatmap
    st.subheader("Sales Heatmap — Sub-Category by Year")
    heat = filtered.groupby(['Sub-Category', 'Year'])['Sales'].sum().reset_index()
    heat_pivot = heat.pivot(index='Sub-Category', columns='Year', values='Sales').fillna(0)
    fig_heat = px.imshow(heat_pivot,
                          color_continuous_scale='Blues',
                          labels=dict(color="Sales ($)"),
                          aspect='auto')
    fig_heat.update_layout(height=450)
    st.plotly_chart(fig_heat, use_container_width=True)

# PAGE 2 — FORECAST EXPLORER
elif page == "Forecast Explorer":
    st.title("Forecast Explorer")
    st.markdown("Select a segment and forecast horizon to see predicted future sales.")

    col1, col2 = st.columns(2)
    with col1:
        segment_type = st.selectbox(
            "Select Segment Type",
            ["Overall", "Category", "Region"]
        )
    with col2:
        horizon = st.slider("Forecast Horizon (months)", 1, 6, 3)

    if segment_type == "Category":
        segment_val = st.selectbox("Select Category",
                                    df['Category'].unique().tolist())
        seg_df = df[df['Category'] == segment_val]
    elif segment_type == "Region":
        segment_val = st.selectbox("Select Region",
                                    df['Region'].unique().tolist())
        seg_df = df[df['Region'] == segment_val]
    else:
        segment_val = "Overall"
        seg_df = df.copy()

    # Aggregate
    seg_monthly = (seg_df.groupby(seg_df['Order Date'].dt.to_period('M'))['Sales']
                         .sum().reset_index())
    seg_monthly['Order Date'] = seg_monthly['Order Date'].dt.to_timestamp()
    seg_monthly.columns = ['ds', 'y']

    if st.button("Generate Forecast", type="primary"):
        with st.spinner(f"Fitting Prophet model for {segment_val}..."):
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                interval_width=0.95
            )
            model.fit(seg_monthly)
            future   = model.make_future_dataframe(periods=horizon, freq='MS')
            forecast = model.predict(future)

        # Evaluate on last 3 months of history
        actual   = seg_monthly['y'].values[-3:]
        pred_hist = forecast[forecast['ds'].isin(seg_monthly['ds'].values[-3:])]['yhat'].values
        if len(pred_hist) == 3:
            mae  = mean_absolute_error(actual, pred_hist)
            rmse = np.sqrt(mean_squared_error(actual, pred_hist))
        else:
            mae, rmse = 0, 0

        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Segment",   segment_val)
        m2.metric("MAE",       f"${mae:,.0f}")
        m3.metric("RMSE",      f"${rmse:,.0f}")

        # Plot
        hist_part = forecast[forecast['ds'] <= seg_monthly['ds'].max()]
        fore_part = forecast[forecast['ds'] > seg_monthly['ds'].max()]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=seg_monthly['ds'], y=seg_monthly['y'],
            mode='lines', name='Historical Sales',
            line=dict(color='steelblue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=fore_part['ds'], y=fore_part['yhat'],
            mode='lines+markers', name='Forecast',
            line=dict(color='tomato', width=2.5),
            marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([fore_part['ds'], fore_part['ds'][::-1]]),
            y=pd.concat([fore_part['yhat_upper'], fore_part['yhat_lower'][::-1]]),
            fill='toself', fillcolor='rgba(255,99,71,0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval'
        ))

        fig.update_layout(
            title=f"Sales Forecast — {segment_val} ({horizon} months ahead)",
            yaxis_title='Sales ($)',
            xaxis_title='Date',
            height=450,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Forecast table
        st.subheader("Forecast Values")
        fc_table = fore_part[['ds','yhat','yhat_lower','yhat_upper']].copy()
        fc_table.columns = ['Month','Forecast ($)','Lower Bound ($)','Upper Bound ($)']
        fc_table['Month'] = fc_table['Month'].dt.strftime('%B %Y')
        fc_table = fc_table.set_index('Month').round(0)
        st.dataframe(fc_table.style.format("${:,.0f}"), use_container_width=True)

# PAGE 3 — ANOMALY REPORT

elif page == "Anomaly Report":
    st.title("Anomaly Report")
    st.markdown("Unusual sales weeks detected using Isolation Forest and Z-Score methods.")

    ws = weekly_sales.copy().set_index('ds').sort_index()

    # Isolation Forest
    iso = IsolationForest(contamination=0.07, random_state=42, n_estimators=200)
    ws['iso_anomaly'] = iso.fit_predict(ws[['y']]) == -1

    # Z-Score
    ws['rolling_mean'] = ws['y'].rolling(8, center=True, min_periods=1).mean()
    ws['rolling_std']  = ws['y'].rolling(8, center=True, min_periods=1).std()
    ws['z_score']      = ((ws['y'] - ws['rolling_mean']) /
                          ws['rolling_std'].replace(0, 1))
    ws['z_anomaly']    = ws['z_score'].abs() > 2.0

    method = st.radio("Detection Method",
                       ["Isolation Forest", "Z-Score", "Both Methods"],
                       horizontal=True)

    if method == "Isolation Forest":
        anomaly_col = 'iso_anomaly'
    elif method == "Z-Score":
        anomaly_col = 'z_anomaly'
    else:
        ws['both_anomaly'] = ws['iso_anomaly'] & ws['z_anomaly']
        anomaly_col = 'both_anomaly'

    # KPIs
    n_anomalies = ws[anomaly_col].sum()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Weeks Analyzed", len(ws))
    col2.metric("Anomalies Detected",   int(n_anomalies))
    col3.metric("Anomaly Rate",         f"{n_anomalies/len(ws)*100:.1f}%")

    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ws.index, y=ws['y'],
        mode='lines', name='Weekly Sales',
        line=dict(color='steelblue', width=1.5)
    ))
    if method == "Z-Score" or method == "Both Methods":
        fig.add_trace(go.Scatter(
            x=ws.index, y=ws['rolling_mean'] + 2*ws['rolling_std'],
            mode='lines', name='+2 Std Dev',
            line=dict(color='orange', dash='dash', width=1)
        ))
        fig.add_trace(go.Scatter(
            x=ws.index, y=ws['rolling_mean'] - 2*ws['rolling_std'],
            mode='lines', name='-2 Std Dev',
            line=dict(color='orange', dash='dash', width=1),
            fill='tonexty', fillcolor='rgba(255,165,0,0.08)'
        ))

    anomaly_ws = ws[ws[anomaly_col]]
    fig.add_trace(go.Scatter(
        x=anomaly_ws.index, y=anomaly_ws['y'],
        mode='markers', name='Anomaly',
        marker=dict(color='red', size=10, symbol='circle-open',
                    line=dict(width=2, color='red'))
    ))
    fig.update_layout(
        title=f'Weekly Sales Anomaly Detection — {method}',
        yaxis_title='Sales ($)',
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

    # Anomaly table
    st.subheader("Detected Anomaly Dates")
    anomaly_table = anomaly_ws[['y']].copy().reset_index()
    anomaly_table.columns = ['Week', 'Sales ($)']
    anomaly_table['Direction'] = anomaly_table['Sales ($)'].apply(
        lambda x: 'SPIKE' if x > ws['y'].mean() else 'DIP')
    anomaly_table['Possible Cause'] = anomaly_table.apply(
        lambda r: 'Holiday/festive season surge' if r['Week'].month in [11, 12]
        else 'Post-holiday slowdown' if r['Week'].month in [1, 2]
        else 'Promotional event or supply disruption', axis=1)
    anomaly_table['Week'] = anomaly_table['Week'].dt.strftime('%Y-%m-%d')
    anomaly_table['Sales ($)'] = anomaly_table['Sales ($)'].apply(lambda x: f'${x:,.0f}')
    st.dataframe(anomaly_table, use_container_width=True)

# PAGE 4 — PRODUCT DEMAND SEGMENTS
elif page == "Product Demand Segments":
    st.title("Product Demand Segments")
    st.markdown("Products grouped by demand behavior to guide inventory and stocking strategy.")

    @st.cache_data
    def build_clusters():
        sub_monthly = (df.groupby(['Sub-Category',
                                    df['Order Date'].dt.to_period('M')])['Sales']
                         .sum().reset_index())
        sub_monthly['Order Date'] = sub_monthly['Order Date'].dt.to_timestamp()

        rows = []
        for sub in df['Sub-Category'].unique():
            sdf = sub_monthly[sub_monthly['Sub-Category'] == sub].sort_values('Order Date')
            total   = sdf['Sales'].sum()
            vol     = sdf['Sales'].std()
            avg_ord = df[df['Sub-Category'] == sub]['Sales'].mean()
            fy = sdf[sdf['Order Date'].dt.year == sdf['Order Date'].dt.year.min()]['Sales'].mean()
            ly = sdf[sdf['Order Date'].dt.year == sdf['Order Date'].dt.year.max()]['Sales'].mean()
            growth = ((ly - fy) / (fy + 1)) * 100
            rows.append({'Sub-Category': sub, 'Total Sales': total,
                         'YoY Growth %': growth, 'Volatility': vol,
                         'Avg Order Value': avg_ord})

        feat = pd.DataFrame(rows)
        scaler = StandardScaler()
        X = scaler.fit_transform(feat[['Total Sales','YoY Growth %',
                                       'Volatility','Avg Order Value']])
        km = KMeans(n_clusters=4, random_state=42, n_init=10)
        feat['Cluster'] = km.fit_predict(X)

        stats = feat.groupby('Cluster')[['Total Sales','YoY Growth %','Volatility']].mean()
        labels = {}
        for c in range(4):
            s = stats.loc[c]
            if s['Total Sales'] > feat['Total Sales'].median() and s['Volatility'] < feat['Volatility'].median():
                labels[c] = 'High Volume, Stable Demand'
            elif s['YoY Growth %'] > 50:
                labels[c] = 'Growing Demand'
            elif s['Total Sales'] < feat['Total Sales'].median() and s['Volatility'] > feat['Volatility'].median():
                labels[c] = 'Low Volume, High Volatility'
            else:
                labels[c] = 'Moderate / Declining Demand'

        feat['Cluster Label'] = feat['Cluster'].map(labels)

        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        feat['PCA1'] = X_pca[:, 0]
        feat['PCA2'] = X_pca[:, 1]
        return feat

    feat_matrix = build_clusters()

    # Cluster scatter plot
    color_map = {
        'High Volume, Stable Demand':    '#2196F3',
        'Growing Demand':                 '#4CAF50',
        'Low Volume, High Volatility':    '#F44336',
        'Moderate / Declining Demand':    '#FF9800',
    }
    fig = px.scatter(feat_matrix,
                      x='PCA1', y='PCA2',
                      color='Cluster Label',
                      text='Sub-Category',
                      color_discrete_map=color_map,
                      size='Total Sales',
                      size_max=40,
                      hover_data=['Total Sales','YoY Growth %','Volatility'],
                      title='Product Demand Segmentation (K-Means + PCA)')
    fig.update_traces(textposition='top center', textfont_size=10)
    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True)

    # Stocking strategy cards
    st.subheader("Stocking Strategy per Cluster")
    strategies = {
        'High Volume, Stable Demand':   ('🔵', 'Maintain steady inventory. Use just-in-time replenishment. Low overstock risk.'),
        'Growing Demand':               ('🟢', 'Increase stock 15–20% above last year. Prioritize warehouse space.'),
        'Low Volume, High Volatility':  ('🔴', 'Keep minimal safety stock. Order in small batches. Avoid dead stock.'),
        'Moderate / Declining Demand':  ('🟠', 'Gradually reduce inventory. Clear stock with promotions. Consider discontinuing.'),
    }

    for label, (icon, strategy) in strategies.items():
        subs = feat_matrix[feat_matrix['Cluster Label'] == label]['Sub-Category'].tolist()
        with st.expander(f"{icon} {label} — {len(subs)} products"):
            st.markdown(f"**Strategy:** {strategy}")
            st.markdown(f"**Products:** {', '.join(subs)}")
            sub_stats = feat_matrix[feat_matrix['Cluster Label'] == label][
                ['Sub-Category','Total Sales','YoY Growth %','Volatility']
            ].round(1)
            sub_stats['Total Sales'] = sub_stats['Total Sales'].apply(lambda x: f'${x:,.0f}')
            st.dataframe(sub_stats.set_index('Sub-Category'), use_container_width=True)
