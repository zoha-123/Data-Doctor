import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.combine import SMOTEENN, SMOTETomek
import io
import base64
from PIL import Image
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# Install packages if needed
import sys
import subprocess
for package in ['matplotlib', 'seaborn', 'scikit-learn', 'imbalanced-learn', 'plotly']:
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Page configuration with light theme
st.set_page_config(
    page_title="DataPrep Pro - Smart Data Cleaning",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern light theme
st.markdown("""
<style>
    /* Main theme colors - Light and clean */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        color: #495057;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .insight-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border-radius: 25px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Hide sidebar */
    section[data-testid="stSidebar"] {
        display: none;
    }
    
    /* Progress bar */
    .stProgress .progress-bar {
        background-color: #667eea;
    }
    
    /* Dataframe styling */
    .dataframe {
        background: white;
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'target_variable' not in st.session_state:
    st.session_state.target_variable = None
if 'insights' not in st.session_state:
    st.session_state.insights = {}

# Header
st.markdown("""
<div class="header-container">
    <h1>✨ DataPrep Pro</h1>
    <p>Transform Your Data into ML-Ready Insights with AI-Powered Cleaning</p>
</div>
""", unsafe_allow_html=True)

# Main content with tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Upload & Insights", 
    "🔍 Data Quality", 
    "🧹 Clean Data", 
    "⚖️ Balance Data", 
    "📏 Scale Features", 
    "🔗 Correlations", 
    "🛠️ Feature Engineering", 
    "💾 Export"
])

# Function to generate instant insights
def generate_insights(df):
    insights = {
        'overview': {},
        'quality': {},
        'recommendations': []
    }
    
    # Overview insights
    insights['overview'] = {
        'total_rows': df.shape[0],
        'total_columns': df.shape[1],
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    # Quality insights
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    high_missing = missing_pct[missing_pct > 20].to_dict()
    
    insights['quality'] = {
        'high_missing_columns': high_missing,
        'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100).round(2),
        'duplicate_percentage': (df.duplicated().sum() / len(df) * 100).round(2)
    }
    
    # Recommendations
    if insights['quality']['missing_percentage'] > 10:
        insights['recommendations'].append("🔍 High missing values detected - consider imputation strategies")
    
    if insights['quality']['duplicate_percentage'] > 5:
        insights['recommendations'].append("🔄 Significant duplicates found - review for removal")
    
    if insights['overview']['categorical_columns'] > 0:
        insights['recommendations'].append("🏷️ Categorical variables detected - encoding needed for ML")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        for col in numeric_cols[:3]:  # Check first 3 numeric columns
            if df[col].skew() > 2:
                insights['recommendations'].append(f"📈 {col} is highly skewed - consider transformation")
                break
    
    return insights

# Function to create visual metrics
def create_metric_card(title, value, subtitle="", color="#667eea"):
    return f"""
    <div class="metric-card">
        <h3 style="color: {color}; margin: 0;">{title}</h3>
        <h2 style="margin: 0.5rem 0;">{value}</h2>
        <p style="margin: 0; color: #6c757d;">{subtitle}</p>
    </div>
    """

# Tab 1: Upload & Instant Insights
with tab1:
    st.markdown("### 📤 Upload Your Dataset")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        try:
            # Load data
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            st.session_state.original_data = data.copy()
            st.session_state.processed_data = data.copy()
            
            # Generate instant insights
            insights = generate_insights(data)
            st.session_state.insights = insights
            
            # Success message
            st.markdown("""
            <div class="success-card">
                <h3>🎉 Data Loaded Successfully!</h3>
                <p>Your dataset has been analyzed and insights are ready below.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Key metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(create_metric_card(
                    "Total Rows", 
                    f"{insights['overview']['total_rows']:,}",
                    "Records"
                ), unsafe_allow_html=True)
            
            with col2:
                st.markdown(create_metric_card(
                    "Features", 
                    insights['overview']['total_columns'],
                    f"{insights['overview']['numeric_columns']} numeric, {insights['overview']['categorical_columns']} categorical"
                ), unsafe_allow_html=True)
            
            with col3:
                st.markdown(create_metric_card(
                    "Missing Data", 
                    f"{insights['quality']['missing_percentage']}%",
                    f"{insights['overview']['missing_values']} cells"
                ), unsafe_allow_html=True)
            
            with col4:
                st.markdown(create_metric_card(
                    "Duplicates", 
                    f"{insights['quality']['duplicate_percentage']}%",
                    f"{insights['overview']['duplicate_rows']} rows"
                ), unsafe_allow_html=True)
            
            # Data preview with visualizations
            st.markdown("### 📊 Data Preview & Distribution")
            
            # Sample data display (more visual)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Sample Data**")
                # Create a styled table
                fig = go.Figure(data=[go.Table(
                    header=dict(values=list(data.columns),
                               fill_color='lightgray',
                               align='left'),
                    cells=dict(values=[data[col].head().tolist() for col in data.columns],
                              fill_color='white',
                              align='left'))
                ])
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Data Types Distribution**")
                dtype_counts = data.dtypes.value_counts()
                fig = px.pie(
                    values=dtype_counts.values, 
                    names=dtype_counts.index,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
            
            # Instant insights
            st.markdown("### 💡 AI-Generated Insights")
            
            if insights['recommendations']:
                for rec in insights['recommendations']:
                    st.markdown(f"""
                    <div class="insight-card">
                        <p>{rec}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-card">
                    <p>✅ Your data looks clean and ready for processing!</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Target variable selection
            st.markdown("### 🎯 Define Your Objective")
            
            col1, col2 = st.columns(2)
            
            with col1:
                objective = st.selectbox(
                    "What's your goal?",
                    ["Classification", "Regression", "Clustering", "Other"],
                    key="objective"
                )
            
            with col2:
                target_options = ["None"] + list(data.columns)
                target = st.selectbox("Target variable (if any)", target_options, key="target")
                if target != "None":
                    st.session_state.target_variable = target
            
            # Target variable visualization
            if target != "None" and target in data.columns:
                st.markdown(f"### 📈 Target Variable: {target}")
                
                if pd.api.types.is_numeric_dtype(data[target]):
                    fig = px.histogram(data, x=target, marginal="box", color_discrete_sequence=['#667eea'])
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.bar(data[target].value_counts().reset_index(), 
                                x='index', y=target, 
                                color_discrete_sequence=['#667eea'])
                    fig.update_layout(height=300, xaxis_title=target, yaxis_title="Count")
                    st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading file: {e}")

# Tab 2: Data Quality Assessment
with tab2:
    if st.session_state.data is None:
        st.warning("Please upload data first!")
    else:
        data = st.session_state.data
        
        st.markdown("### 🔍 Data Quality Dashboard")
        
        # Missing values visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Missing Values Pattern**")
            missing_data = data.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) > 0:
                fig = px.bar(x=missing_data.index, y=missing_data.values, 
                             color=missing_data.values,
                             color_continuous_scale='Reds',
                             labels={'x': 'Columns', 'y': 'Missing Count'})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No missing values found!")
        
        with col2:
            st.markdown("**Missing Values Heatmap**")
            fig = px.imshow(data.isnull().T, color_continuous_scale='RdYlBu_r', 
                           aspect="auto", height=400)
            fig.update_layout(yaxis_title="Columns")
            st.plotly_chart(fig, use_container_width=True)
        
        # Data quality metrics
        st.markdown("### 📊 Quality Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            completeness = (1 - data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
            st.metric("Data Completeness", f"{completeness:.1f}%")
        
        with col2:
            uniqueness = (data.nunique().sum() / (data.shape[0] * data.shape[1])) * 100
            st.metric("Uniqueness Score", f"{uniqueness:.1f}%")
        
        with col3:
            st.metric("Duplicate Rows", data.duplicated().sum())
        
        with col4:
            st.metric("Zero Values", (data == 0).sum().sum())
        
        # Column-wise analysis
        st.markdown("### 📋 Column Analysis")
        
        # Create column analysis data
        col_analysis = []
        for col in data.columns:
            col_info = {
                'Column': col,
                'Type': str(data[col].dtype),
                'Missing': data[col].isnull().sum(),
                'Missing %': f"{(data[col].isnull().sum() / len(data) * 100):.1f}%",
                'Unique': data[col].nunique(),
                'Zeros': (data[col] == 0).sum()
            }
            col_analysis.append(col_info)
        
        col_df = pd.DataFrame(col_analysis)
        
        # Visual column analysis
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(col_df.columns),
                       fill_color='lightblue',
                       align='left'),
            cells=dict(values=[col_df[col] for col in col_df.columns],
                      fill_color='white',
                      align='left'))
        ])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# Tab 3: Clean Data
with tab3:
    if st.session_state.data is None:
        st.warning("Please upload data first!")
    else:
        data = st.session_state.processed_data.copy()
        
        st.markdown("### 🧹 Data Cleaning Options")
        
        # Missing values handling
        with st.expander("🔍 Handle Missing Values", expanded=True):
            missing_cols = data.isnull().sum()
            missing_cols = missing_cols[missing_cols > 0]
            
            if len(missing_cols) > 0:
                st.write(f"Found {len(missing_cols)} columns with missing values")
                
                # Method selection
                method = st.selectbox("Choose method:", [
                    "Drop rows with missing values",
                    "Fill with mean (numeric)",
                    "Fill with median (numeric)",
                    "Fill with mode",
                    "Forward fill",
                    "Backward fill",
                    "KNN Imputer"
                ])
                
                if st.button("Apply Cleaning"):
                    if method == "Drop rows with missing values":
                        data = data.dropna()
                    elif method == "Fill with mean (numeric)":
                        for col in missing_cols.index:
                            if pd.api.types.is_numeric_dtype(data[col]):
                                data[col].fillna(data[col].mean(), inplace=True)
                    elif method == "Fill with median (numeric)":
                        for col in missing_cols.index:
                            if pd.api.types.is_numeric_dtype(data[col]):
                                data[col].fillna(data[col].median(), inplace=True)
                    elif method == "Fill with mode":
                        for col in missing_cols.index:
                            data[col].fillna(data[col].mode()[0], inplace=True)
                    elif method == "Forward fill":
                        data = data.fillna(method='ffill')
                    elif method == "Backward fill":
                        data = data.fillna(method='bfill')
                    elif method == "KNN Imputer":
                        from sklearn.impute import KNNImputer
                        imputer = KNNImputer(n_neighbors=5)
                        numeric_cols = data.select_dtypes(include=[np.number]).columns
                        data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
                    
                    st.session_state.processed_data = data
                    st.success("✅ Missing values handled successfully!")
                    st.rerun()
            else:
                st.success("✅ No missing values found!")
        
        # Duplicate handling
        with st.expander("🔄 Handle Duplicates"):
            duplicates = data.duplicated().sum()
            if duplicates > 0:
                st.write(f"Found {duplicates} duplicate rows")
                
                if st.button("Remove Duplicates"):
                    data = data.drop_duplicates()
                    st.session_state.processed_data = data
                    st.success(f"✅ Removed {duplicates} duplicate rows!")
                    st.rerun()
            else:
                st.success("✅ No duplicates found!")
        
        # Outlier detection
        with st.expander("📊 Handle Outliers"):
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                col_to_check = st.selectbox("Select column to check:", numeric_cols)
                
                if col_to_check:
                    # Show distribution
                    fig = px.box(data, y=col_to_check, color_discrete_sequence=['#667eea'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    method = st.selectbox("Outlier removal method:", [
                        "IQR Method",
                        "Z-Score Method",
                        "Isolation Forest"
                    ])
                    
                    if st.button("Remove Outliers"):
                        if method == "IQR Method":
                            Q1 = data[col_to_check].quantile(0.25)
                            Q3 = data[col_to_check].quantile(0.75)
                            IQR = Q3 - Q1
                            data = data[~((data[col_to_check] < (Q1 - 1.5 * IQR)) | 
                                       (data[col_to_check] > (Q3 + 1.5 * IQR)))]
                        elif method == "Z-Score Method":
                            from scipy import stats
                            z_scores = np.abs(stats.zscore(data[col_to_check]))
                            data = data[z_scores < 3]
                        elif method == "Isolation Forest":
                            from sklearn.ensemble import IsolationForest
                            iso = IsolationForest(contamination=0.1)
                            outliers = iso.fit_predict(data[[col_to_check]])
                            data = data[outliers == 1]
                        
                        st.session_state.processed_data = data
                        st.success("✅ Outliers removed successfully!")
                        st.rerun()
            else:
                st.info("No numeric columns found for outlier detection")

# Tab 4: Balance Data
with tab4:
    if st.session_state.processed_data is None:
        st.warning("Please upload and clean data first!")
    else:
        data = st.session_state.processed_data.copy()
        
        st.markdown("### ⚖️ Data Balancing")
        
        if st.session_state.target_variable and st.session_state.target_variable in data.columns:
            target = st.session_state.target_variable
            
            # Show current distribution
            st.markdown("**Current Class Distribution**")
            
            if pd.api.types.is_numeric_dtype(data[target]) and data[target].nunique() < 20:
                data[target] = data[target].astype('category')
            
            class_counts = data[target].value_counts()
            
            fig = px.bar(x=class_counts.index, y=class_counts.values, 
                        color=class_counts.values,
                        color_continuous_scale='Blues',
                        labels={'x': 'Class', 'y': 'Count'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Check imbalance
            imbalance_ratio = class_counts.max() / class_counts.min()
            
            if imbalance_ratio > 2:
                st.warning(f"⚠️ Data is imbalanced (ratio: {imbalance_ratio:.1f}:1)")
                
                # Balancing options
                method = st.selectbox("Choose balancing method:", [
                    "SMOTE (Oversampling)",
                    "Random Undersampling",
                    "SMOTE + ENN (Combined)"
                ])
                
                if st.button("Apply Balancing"):
                    X = data.drop(columns=[target])
                    y = data[target]
                    
                    if method == "SMOTE (Oversampling)":
                        from imblearn.over_sampling import SMOTE
                        smote = SMOTE(random_state=42)
                        X_res, y_res = smote.fit_resample(X, y)
                    elif method == "Random Undersampling":
                        from imblearn.under_sampling import RandomUnderSampler
                        rus = RandomUnderSampler(random_state=42)
                        X_res, y_res = rus.fit_resample(X, y)
                    else:  # SMOTE + ENN
                        from imblearn.combine import SMOTEENN
                        smote_enn = SMOTEENN(random_state=42)
                        X_res, y_res = smote_enn.fit_resample(X, y)
                    
                    balanced_data = pd.concat([X_res, y_res], axis=1)
                    st.session_state.processed_data = balanced_data
                    
                    # Show new distribution
                    new_counts = y_res.value_counts()
                    fig = px.bar(x=new_counts.index, y=new_counts.values,
                                color=new_counts.values,
                                color_continuous_scale='Greens',
                                labels={'x': 'Class', 'y': 'Count'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success("✅ Data balanced successfully!")
            else:
                st.success("✅ Data appears to be balanced!")
        else:
            st.warning("Please select a target variable in the Upload & Insights tab first!")

# Tab 5: Scale Features
with tab5:
    if st.session_state.processed_data is None:
        st.warning("Please complete previous steps first!")
    else:
        data = st.session_state.processed_data.copy()
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if st.session_state.target_variable and st.session_state.target_variable in numeric_cols:
            numeric_cols = numeric_cols.drop(st.session_state.target_variable)
        
        if len(numeric_cols) > 0:
            st.markdown("### 📏 Feature Scaling")
            
            # Show current distributions
            st.markdown("**Current Feature Distributions**")
            
            selected_cols = st.multiselect("Select features to scale:", numeric_cols, default=list(numeric_cols)[:4])
            
            if selected_cols:
                fig = make_subplots(rows=2, cols=2, subplot_titles=selected_cols[:4])
                
                for i, col in enumerate(selected_cols[:4]):
                    row = i // 2 + 1
                    col_idx = i % 2 + 1
                    fig.add_trace(
                        go.Histogram(x=data[col], name=col, opacity=0.7),
                        row=row, col=col_idx
                    )
                
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Scaling method
                method = st.selectbox("Choose scaling method:", [
                    "StandardScaler (Z-score)",
                    "MinMaxScaler (0-1)",
                    "RobustScaler"
                ])
                
                if st.button("Apply Scaling"):
                    if method == "StandardScaler (Z-score)":
                        scaler = StandardScaler()
                    elif method == "MinMaxScaler (0-1)":
                        scaler = MinMaxScaler()
                    else:
                        scaler = RobustScaler()
                    
                    data[selected_cols] = scaler.fit_transform(data[selected_cols])
                    st.session_state.processed_data = data
                    
                    # Show scaled distributions
                    fig = make_subplots(rows=2, cols=2, subplot_titles=selected_cols[:4])
                    
                    for i, col in enumerate(selected_cols[:4]):
                        row = i // 2 + 1
                        col_idx = i % 2 + 1
                        fig.add_trace(
                            go.Histogram(x=data[col], name=col, opacity=0.7),
                            row=row, col=col_idx
                        )
                    
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success("✅ Features scaled successfully!")
        else:
            st.info("No numeric features to scale!")

# Tab 6: Correlations
with tab6:
    if st.session_state.processed_data is None:
        st.warning("Please complete previous steps first!")
    else:
        data = st.session_state.processed_data.copy()
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) > 1:
            st.markdown("### 🔗 Correlation Analysis")
            
            # Correlation heatmap
            corr_matrix = numeric_data.corr()
            
            fig = px.imshow(corr_matrix, 
                           color_continuous_scale='RdBu_r',
                           aspect="auto",
                           title="Feature Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
            
            # Target correlations
            if st.session_state.target_variable and st.session_state.target_variable in numeric_data.columns:
                target_corr = corr_matrix[st.session_state.target_variable].abs().sort_values(ascending=False)
                
                st.markdown(f"### 🎯 Features Correlated with {st.session_state.target_variable}")
                
                fig = px.bar(x=target_corr.drop(st.session_state.target_variable).index,
                            y=target_corr.drop(st.session_state.target_variable).values,
                            color=target_corr.drop(st.session_state.target_variable).values,
                            color_continuous_scale='Viridis',
                            labels={'x': 'Features', 'y': 'Absolute Correlation'})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature selection
                st.markdown("### 🎛️ Feature Selection")
                
                threshold = st.slider("Correlation threshold for removal:", 0.0, 0.95, 0.8, 0.05)
                
                # Find highly correlated pairs
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > threshold:
                            high_corr_pairs.append((
                                corr_matrix.columns[i],
                                corr_matrix.columns[j],
                                corr_matrix.iloc[i, j]
                            ))
                
                if high_corr_pairs:
                    st.write(f"Found {len(high_corr_pairs)} highly correlated pairs:")
                    
                    # Create suggestions for removal
                    to_remove = set()
                    for feat1, feat2, corr in high_corr_pairs:
                        if st.session_state.target_variable in corr_matrix:
                            # Keep feature more correlated with target
                            if abs(corr_matrix.loc[feat1, st.session_state.target_variable]) < \
                               abs(corr_matrix.loc[feat2, st.session_state.target_variable]):
                                to_remove.add(feat1)
                            else:
                                to_remove.add(feat2)
                        else:
                            to_remove.add(feat2)  # Remove second feature by default
                    
                    selected_to_remove = st.multiselect("Select features to remove:", list(to_remove), default=list(to_remove))
                    
                    if st.button("Remove Selected Features"):
                        data = data.drop(columns=selected_to_remove)
                        st.session_state.processed_data = data
                        st.success(f"✅ Removed {len(selected_to_remove)} highly correlated features!")
                        st.rerun()
                else:
                    st.success(f"✅ No features with correlation > {threshold} found!")
        else:
            st.info("Need at least 2 numeric features for correlation analysis!")

# Tab 7: Feature Engineering
with tab7:
    if st.session_state.processed_data is None:
        st.warning("Please complete previous steps first!")
    else:
        data = st.session_state.processed_data.copy()
        
        st.markdown("### 🛠️ Feature Engineering")
        
        engineering_type = st.selectbox("Choose engineering technique:", [
            "Create Interaction Features",
            "Create Polynomial Features",
            "Binning Numeric Features",
            "Log Transformation"
        ])
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if st.session_state.target_variable and st.session_state.target_variable in numeric_cols:
            numeric_cols = numeric_cols.drop(st.session_state.target_variable)
        
        if engineering_type == "Create Interaction Features":
            if len(numeric_cols) >= 2:
                selected = st.multiselect("Select features for interaction:", numeric_cols, default=list(numeric_cols)[:3])
                
                if len(selected) >= 2 and st.button("Create Interactions"):
                    for i in range(len(selected)):
                        for j in range(i+1, len(selected)):
                            feat_name = f"{selected[i]}_x_{selected[j]}"
                            data[feat_name] = data[selected[i]] * data[selected[j]]
                    
                    st.session_state.processed_data = data
                    st.success(f"✅ Created {len(selected) * (len(selected) - 1) // 2} interaction features!")
            else:
                st.info("Need at least 2 numeric features!")
        
        elif engineering_type == "Create Polynomial Features":
            if len(numeric_cols) >= 1:
                selected = st.multiselect("Select features:", numeric_cols, default=list(numeric_cols)[:2])
                degree = st.slider("Polynomial degree:", 2, 4, 2)
                
                if selected and st.button("Create Polynomial Features"):
                    from sklearn.preprocessing import PolynomialFeatures
                    poly = PolynomialFeatures(degree=degree, include_bias=False)
                    poly_features = poly.fit_transform(data[selected])
                    
                    feature_names = poly.get_feature_names_out(selected)
                    poly_df = pd.DataFrame(poly_features, columns=feature_names, index=data.index)
                    
                    # Add new features (excluding original ones)
                    new_features = [name for name in feature_names if name not in selected]
                    data = pd.concat([data, poly_df[new_features]], axis=1)
                    
                    st.session_state.processed_data = data
                    st.success(f"✅ Created {len(new_features)} polynomial features!")
            else:
                st.info("No numeric features available!")
        
        elif engineering_type == "Binning Numeric Features":
            if len(numeric_cols) >= 1:
                selected = st.selectbox("Select feature to bin:", numeric_cols)
                n_bins = st.slider("Number of bins:", 2, 10, 5)
                
                if selected and st.button("Apply Binning"):
                    bin_name = f"{selected}_binned"
                    data[bin_name] = pd.cut(data[selected], bins=n_bins, labels=False)
                    
                    st.session_state.processed_data = data
                    st.success(f"✅ Created {bin_name} with {n_bins} bins!")
            else:
                st.info("No numeric features available!")
        
        else:  # Log Transformation
            if len(numeric_cols) >= 1:
                selected = st.multiselect("Select features for log transform:", numeric_cols, 
                                         default=[col for col in numeric_cols if (data[col] > 0).all()][:2])
                
                if selected and st.button("Apply Log Transform"):
                    for col in selected:
                        if (data[col] <= 0).any():
                            # Add constant to make positive
                            data[f"{col}_log"] = np.log(data[col] - data[col].min() + 1)
                        else:
                            data[f"{col}_log"] = np.log(data[col])
                    
                    st.session_state.processed_data = data
                    st.success(f"✅ Applied log transformation to {len(selected)} features!")
            else:
                st.info("No numeric features available!")

# Tab 8: Export
with tab8:
    if st.session_state.processed_data is None:
        st.warning("No data to export!")
    else:
        data = st.session_state.processed_data
        
        st.markdown("### 💾 Export Your Processed Data")
        
        # Summary of processing
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Final Rows", data.shape[0])
        with col2:
            st.metric("Final Columns", data.shape[1])
        with col3:
            st.metric("Missing Values", data.isnull().sum().sum())
        
        # Processing summary
        st.markdown("### 📋 Processing Summary")
        
        original_shape = st.session_state.original_data.shape
        processed_shape = data.shape
        
        summary = f"""
        - **Original Dataset**: {original_shape[0]:,} rows × {original_shape[1]} columns
        - **Final Dataset**: {processed_shape[0]:,} rows × {processed_shape[1]} columns
        - **Rows Changed**: {processed_shape[0] - original_shape[0]:+,}
        - **Columns Added**: {processed_shape[1] - original_shape[1]:+,}
        - **Data Quality**: {(1 - data.isnull().sum().sum() / (data.shape[0] * data.shape[1]) * 100):.1f}% complete
        """
        
        st.markdown(summary)
        
        # Export options
        st.markdown("### 📤 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            format_choice = st.radio("Export format:", ["CSV", "Excel", "JSON"])
            
            if format_choice == "CSV":
                filename = st.text_input("Filename:", "processed_data.csv")
                if st.button("Download CSV"):
                    csv = data.to_csv(index=False)
                    st.download_button(
                        label="Download CSV file",
                        data=csv,
                        file_name=filename,
                        mime='text/csv'
                    )
            
            elif format_choice == "Excel":
                filename = st.text_input("Filename:", "processed_data.xlsx")
                if st.button("Download Excel"):
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        data.to_excel(writer, index=False, sheet_name='Processed Data')
                    st.download_button(
                        label="Download Excel file",
                        data=output.getvalue(),
                        file_name=filename,
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
            
            else:  # JSON
                filename = st.text_input("Filename:", "processed_data.json")
                if st.button("Download JSON"):
                    json_data = data.to_json(orient='records')
                    st.download_button(
                        label="Download JSON file",
                        data=json_data,
                        file_name=filename,
                        mime='application/json'
                    )
        
        with col2:
            st.markdown("### 🚀 Ready for ML!")
            
            st.code("""
# Example code to use your data
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your data
data = pd.read_csv('processed_data.csv')

# Prepare features and target
X = data.drop('target_column', axis=1)
y = data['target_column']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Now train your model!
            """, language='python')
            
            if st.button("🔄 Start New Project"):
                # Clear session state
                for key in st.session_state.keys():
                    del st.session_state[key]
                st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6c757d;'>"
    "Made with ❤️ by DataPrep Pro | Transform Your Data into Insights"
    "</div>", 
    unsafe_allow_html=True
)
