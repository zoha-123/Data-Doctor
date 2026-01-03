import sys
import subprocess
import os

# Direct pip install
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install all required packages
required_packages = [
    'matplotlib',
    'seaborn', 
    'scikit-learn',
    'imbalanced-learn',
    'pandas',
    'numpy'
]

print("Installing required packages...")
for package in required_packages:
    try:
        __import__(package)
        print(f"✓ {package} already installed")
    except ImportError:
        print(f"Installing {package}...")
        install(package)
        print(f"✓ {package} installed successfully")

print("All packages installed!")

# Now import everything
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
warnings.filterwarnings('ignore')

# Continue with the rest of your code...

# Set page configuration
st.set_page_config(
    page_title="DataPrep Pro - ML Data Cleaning & Preprocessing",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FFA500;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #FFA500;
        margin-bottom: 1rem;
    }
    .step-card {
        background-color: #F0F2F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #FFA500;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
    }
    .stButton>button:hover {
        background-color: #FF8C00;
    }
    .stSelectbox>div>div>select {
        color: #FFA500;
    }
    .stRadio>div>div>label {
        color: #FFA500;
    }
    .stProgress .progress-bar {
        background-color: #FFA500;
    }
    .metric-card {
        background-color: #F0F2F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    .dataframe {
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
def init_session_state():
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'original_data' not in st.session_state:
        st.session_state.original_data = None
    if 'user_objective' not in st.session_state:
        st.session_state.user_objective = None
    if 'target_variable' not in st.session_state:
        st.session_state.target_variable = None
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'steps_completed' not in st.session_state:
        st.session_state.steps_completed = []
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None

# Define the steps in the data preparation pipeline
steps = [
    "Upload Data",
    "Explore Data",
    "Handle Missing Values",
    "Encode Categorical Variables",
    "Handle Outliers",
    "Address Data Imbalance",
    "Feature Scaling",
    "Correlation Analysis",
    "Feature Engineering",
    "Visualize Results",
    "Export Data"
]

# Sidebar for navigation
def render_sidebar():
    st.sidebar.title("DataPrep Pro")
    st.sidebar.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/public/logo.svg", width=100)

    # Progress indicator
    st.sidebar.markdown("### Progress")
    progress = len(st.session_state.steps_completed) / len(steps)
    st.sidebar.progress(progress)
    st.sidebar.markdown(f"**{len(st.session_state.steps_completed)}/{len(steps)} steps completed**")

    # Navigation
    st.sidebar.markdown("### Navigation")
    for i, step in enumerate(steps):
        if st.sidebar.button(step, key=f"nav_{i}"):
            st.session_state.current_step = i
            st.rerun()

    # Reset button
    if st.sidebar.button("Reset All", type="secondary"):
        for key in st.session_state.keys():
            del st.session_state[key]
        init_session_state()
        st.rerun()

# Step 1: Upload Data
def upload_data_step():
    st.markdown('<h1 class="main-header">Upload Your Data</h1>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                # Read the data
                data = pd.read_csv(uploaded_file)

                # Store in session state
                st.session_state.data = data
                st.session_state.original_data = data.copy()
                st.session_state.processed_data = data.copy()

                # Display success message
                st.success(f"File uploaded successfully! Dataset shape: {data.shape}")

                # Display data preview
                st.dataframe(data.head())

                # Add to steps completed
                if "Upload Data" not in st.session_state.steps_completed:
                    st.session_state.steps_completed.append("Upload Data")

                # Add next step button
                if st.button("Next Step → Explore Data", type="primary"):
                    st.session_state.current_step = 1
                    st.rerun()

            except Exception as e:
                st.error(f"Error reading file: {e}")

    with col2:
        st.markdown("### Tips")
        st.info("""
        - Upload a CSV file with your raw data
        - The first row should contain column names
        - Supported formats: CSV
        - Maximum file size: 200MB
        """)

# Step 2: Explore Data
def explore_data_step():
    st.markdown('<h1 class="main-header">Explore Your Data</h1>', unsafe_allow_html=True)

    if st.session_state.data is None:
        st.warning("Please upload data first!")
        return

    data = st.session_state.data

    # Display basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", data.shape[0])
    with col2:
        st.metric("Columns", data.shape[1])
    with col3:
        st.metric("Missing Values", data.isnull().sum().sum())

    # Data info
    st.markdown('<h2 class="sub-header">Data Information</h2>', unsafe_allow_html=True)

    buffer = io.StringIO()
    data.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

    # Data description
    st.markdown('<h2 class="sub-header">Statistical Summary</h2>', unsafe_allow_html=True)
    st.dataframe(data.describe())

    # Data types
    st.markdown('<h2 class="sub-header">Data Types</h2>', unsafe_allow_html=True)
    dtype_counts = data.dtypes.value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    dtype_counts.plot(kind='bar', color='#FFA500', ax=ax)
    plt.title('Data Types Distribution')
    plt.tight_layout()
    st.pyplot(fig)

    # Set user objective and target variable
    st.markdown('<h2 class="sub-header">Set Your Objective</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        objective = st.selectbox(
            "What is your primary objective with this data?",
            [
                'Classification (predicting a category)',
                'Regression (predicting a continuous value)',
                'Clustering (grouping similar data points)',
                'Dimensionality Reduction (reducing number of features)',
                'Anomaly Detection (finding unusual data points)',
                'Other'
            ]
        )
        st.session_state.user_objective = objective

    with col2:
        target_options = ['None'] + list(data.columns)
        target = st.selectbox("Select your target variable (if applicable)", target_options)
        if target != 'None':
            st.session_state.target_variable = target

    # Show target variable distribution if selected
    if st.session_state.target_variable and st.session_state.target_variable in data.columns:
        st.markdown('<h2 class="sub-header">Target Variable Distribution</h2>', unsafe_allow_html=True)

        target_col = st.session_state.target_variable
        if pd.api.types.is_numeric_dtype(data[target_col]):
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data[target_col], kde=True, color='#FFA500', ax=ax)
            plt.title(f'Distribution of {target_col}')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(y=data[target_col], color='#FFA500', ax=ax)
            plt.title(f'Distribution of {target_col}')
            plt.tight_layout()
            st.pyplot(fig)

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Previous Step", type="secondary"):
            st.session_state.current_step = 0
            st.rerun()
    with col2:
        if st.button("Next Step → Handle Missing Values", type="primary"):
            if "Explore Data" not in st.session_state.steps_completed:
                st.session_state.steps_completed.append("Explore Data")
            st.session_state.current_step = 2
            st.rerun()

# Step 3: Handle Missing Values
def handle_missing_values_step():
    st.markdown('<h1 class="main-header">Handle Missing Values</h1>', unsafe_allow_html=True)

    if st.session_state.data is None:
        st.warning("Please upload data first!")
        return

    data = st.session_state.data.copy()

    # Check for missing values
    missing_values = data.isnull().sum()
    missing_cols = missing_values[missing_values > 0]

    if len(missing_cols) == 0:
        st.success("No missing values found in the dataset!")
        if "Handle Missing Values" not in st.session_state.steps_completed:
            st.session_state.steps_completed.append("Handle Missing Values")
        st.session_state.processed_data = data
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Previous Step", type="secondary"):
                st.session_state.current_step = 1
                st.rerun()
        with col2:
            if st.button("Next Step → Encode Categorical Variables", type="primary"):
                st.session_state.current_step = 3
                st.rerun()
        return

    # Display missing values overview
    st.markdown('<h2 class="sub-header">Missing Values Overview</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(missing_cols)
    with col2:
        # Visualize missing values
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data.isnull(), cbar=False, cmap='viridis', ax=ax)
        plt.title('Missing Values Heatmap')
        plt.tight_layout()
        st.pyplot(fig)

    # Missing values handling options
    st.markdown('<h2 class="sub-header">Missing Values Handling</h2>', unsafe_allow_html=True)

    method = st.radio(
        "Choose a method to handle missing values:",
        ["Individual Column Handling", "Bulk Handling"]
    )

    if method == "Individual Column Handling":
        st.markdown("### Handle Each Column Individually")

        for col in missing_cols.index:
            with st.expander(f"Column: {col} ({missing_cols[col]} missing values)"):
                is_numeric = pd.api.types.is_numeric_dtype(data[col])

                if is_numeric:
                    fill_method = st.radio(
                        f"Fill method for {col}:",
                        ["Mean", "Median", "Mode", "KNN Imputer", "Drop Rows"]
                    )
                else:
                    fill_method = st.radio(
                        f"Fill method for {col}:",
                        ["Forward Fill", "Backward Fill", "Mode", "Drop Rows"]
                    )

                if st.button(f"Apply to {col}", key=f"apply_{col}"):
                    if is_numeric:
                        if fill_method == "Mean":
                            fill_value = data[col].mean()
                            data[col].fillna(fill_value, inplace=True)
                            st.success(f"Filled missing values in {col} with mean: {fill_value}")
                        elif fill_method == "Median":
                            fill_value = data[col].median()
                            data[col].fillna(fill_value, inplace=True)
                            st.success(f"Filled missing values in {col} with median: {fill_value}")
                        elif fill_method == "Mode":
                            fill_value = data[col].mode()[0]
                            data[col].fillna(fill_value, inplace=True)
                            st.success(f"Filled missing values in {col} with mode: {fill_value}")
                        elif fill_method == "KNN Imputer":
                            imputer = KNNImputer(n_neighbors=5)
                            data[col] = imputer.fit_transform(data[[col]]).ravel()
                            st.success(f"Filled missing values in {col} using KNN Imputer")
                        else:  # Drop Rows
                            data.dropna(subset=[col], inplace=True)
                            st.success(f"Dropped rows with missing values in {col}")
                    else:
                        if fill_method == "Forward Fill":
                            data[col].fillna(method='ffill', inplace=True)
                            st.success(f"Filled missing values in {col} using forward fill")
                        elif fill_method == "Backward Fill":
                            data[col].fillna(method='bfill', inplace=True)
                            st.success(f"Filled missing values in {col} using backward fill")
                        elif fill_method == "Mode":
                            fill_value = data[col].mode()[0]
                            data[col].fillna(fill_value, inplace=True)
                            st.success(f"Filled missing values in {col} with mode: {fill_value}")
                        else:  # Drop Rows
                            data.dropna(subset=[col], inplace=True)
                            st.success(f"Dropped rows with missing values in {col}")

                    # Update session state
                    st.session_state.processed_data = data
                    st.rerun()
    else:  # Bulk Handling
        st.markdown("### Apply the Same Method to All Columns")

        if st.checkbox("Show advanced options"):
            bulk_method = st.radio(
                "Choose a method:",
                [
                    "Fill all numeric with mean",
                    "Fill all numeric with median",
                    "Fill all with mode",
                    "Drop all rows with missing values",
                    "Drop all columns with missing values"
                ]
            )
        else:
            bulk_method = st.radio(
                "Choose a method:",
                [
                    "Fill all numeric with mean",
                    "Fill all numeric with median",
                    "Fill all with mode",
                    "Drop all rows with missing values"
                ]
            )

        if st.button("Apply to All Columns"):
            if bulk_method == "Fill all numeric with mean":
                for col in missing_cols.index:
                    if pd.api.types.is_numeric_dtype(data[col]):
                        fill_value = data[col].mean()
                        data[col].fillna(fill_value, inplace=True)
                st.success("Filled all numeric missing values with mean")
            elif bulk_method == "Fill all numeric with median":
                for col in missing_cols.index:
                    if pd.api.types.is_numeric_dtype(data[col]):
                        fill_value = data[col].median()
                        data[col].fillna(fill_value, inplace=True)
                st.success("Filled all numeric missing values with median")
            elif bulk_method == "Fill all with mode":
                for col in missing_cols.index:
                    fill_value = data[col].mode()[0]
                    data[col].fillna(fill_value, inplace=True)
                st.success("Filled all missing values with mode")
            elif bulk_method == "Drop all rows with missing values":
                data.dropna(inplace=True)
                st.success("Dropped all rows with missing values")
            else:  # Drop all columns with missing values
                cols_to_drop = missing_cols.index
                data.drop(columns=cols_to_drop, inplace=True)
                st.success(f"Dropped columns: {list(cols_to_drop)}")

            # Update session state
            st.session_state.processed_data = data
            st.rerun()

    # Show updated missing values
    updated_missing = data.isnull().sum()
    st.markdown('<h2 class="sub-header">Updated Missing Values</h2>', unsafe_allow_html=True)
    st.dataframe(updated_missing[updated_missing > 0])

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Previous Step", type="secondary"):
            st.session_state.current_step = 1
            st.rerun()
    with col2:
        if st.button("Next Step → Encode Categorical Variables", type="primary"):
            if "Handle Missing Values" not in st.session_state.steps_completed:
                st.session_state.steps_completed.append("Handle Missing Values")
            st.session_state.current_step = 3
            st.rerun()

# Step 4: Encode Categorical Variables
def encode_categorical_step():
    st.markdown('<h1 class="main-header">Encode Categorical Variables</h1>', unsafe_allow_html=True)

    if st.session_state.processed_data is None:
        st.warning("Please complete the previous steps first!")
        return

    data = st.session_state.processed_data.copy()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns

    if len(categorical_cols) == 0:
        st.success("No categorical columns found!")
        if "Encode Categorical Variables" not in st.session_state.steps_completed:
            st.session_state.steps_completed.append("Encode Categorical Variables")
        st.session_state.processed_data = data
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Previous Step", type="secondary"):
                st.session_state.current_step = 2
                st.rerun()
        with col2:
            if st.button("Next Step → Handle Outliers", type="primary"):
                st.session_state.current_step = 4
                st.rerun()
        return

    # Display categorical columns
    st.markdown('<h2 class="sub-header">Categorical Columns</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(pd.DataFrame({
            'Column': categorical_cols,
            'Unique Values': [data[col].nunique() for col in categorical_cols],
            'Data Type': [data[col].dtype for col in categorical_cols]
        }))

    with col2:
        # Show example values for each categorical column
        for col in categorical_cols[:5]:  # Limit to first 5 columns
            st.write(f"**{col}**: {list(data[col].unique()[:10])}")

    # Encoding options
    st.markdown('<h2 class="sub-header">Encoding Options</h2>', unsafe_allow_html=True)

    method = st.radio(
        "Choose a method to encode categorical variables:",
        ["Individual Column Encoding", "Bulk Encoding"]
    )

    if method == "Individual Column Encoding":
        st.markdown("### Encode Each Column Individually")

        for col in categorical_cols:
            with st.expander(f"Column: {col}"):
                unique_values = data[col].nunique()
                st.write(f"Number of unique values: {unique_values}")

                encoding_method = st.radio(
                    f"Encoding method for {col}:",
                    ["Label Encoding", "One-Hot Encoding", "Target Encoding"]
                )

                if st.button(f"Apply to {col}", key=f"encode_{col}"):
                    if encoding_method == "Label Encoding":
                        le = LabelEncoder()
                        data[col] = le.fit_transform(data[col].astype(str))
                        st.success(f"Applied Label Encoding to {col}")
                    elif encoding_method == "One-Hot Encoding":
                        dummies = pd.get_dummies(data[col], prefix=col)
                        data = pd.concat([data.drop(columns=[col]), dummies], axis=1)
                        st.success(f"Applied One-Hot Encoding to {col}")
                    else:  # Target Encoding
                        if st.session_state.target_variable and st.session_state.target_variable in data.columns:
                            target_mean = data.groupby(col)[st.session_state.target_variable].mean()
                            data[col] = data[col].map(target_mean)
                            st.success(f"Applied Target Encoding to {col}")
                        else:
                            st.error("Target Encoding requires a target variable. Please set one in the Explore Data step.")

                    # Update session state
                    st.session_state.processed_data = data
                    st.rerun()
    else:  # Bulk Encoding
        st.markdown("### Apply the Same Encoding Method to All Categorical Variables")

        encoding_method = st.radio(
            "Choose a method:",
            ["Label Encoding", "One-Hot Encoding"]
        )

        if st.button("Apply to All Columns"):
            if encoding_method == "Label Encoding":
                for col in categorical_cols:
                    le = LabelEncoder()
                    data[col] = le.fit_transform(data[col].astype(str))
                st.success("Applied Label Encoding to all categorical columns")
            else:  # One-Hot Encoding
                data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
                st.success("Applied One-Hot Encoding to all categorical columns")

            # Update session state
            st.session_state.processed_data = data
            st.rerun()

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Previous Step", type="secondary"):
            st.session_state.current_step = 2
            st.rerun()
    with col2:
        if st.button("Next Step → Handle Outliers", type="primary"):
            if "Encode Categorical Variables" not in st.session_state.steps_completed:
                st.session_state.steps_completed.append("Encode Categorical Variables")
            st.session_state.current_step = 4
            st.rerun()

# Step 5: Handle Outliers
def handle_outliers_step():
    st.markdown('<h1 class="main-header">Handle Outliers</h1>', unsafe_allow_html=True)

    if st.session_state.processed_data is None:
        st.warning("Please complete the previous steps first!")
        return

    data = st.session_state.processed_data.copy()
    numeric_cols = data.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        st.warning("No numeric columns found for outlier detection!")
        if "Handle Outliers" not in st.session_state.steps_completed:
            st.session_state.steps_completed.append("Handle Outliers")
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Previous Step", type="secondary"):
                st.session_state.current_step = 3
                st.rerun()
        with col2:
            if st.button("Next Step → Address Data Imbalance", type="primary"):
                st.session_state.current_step = 5
                st.rerun()
        return

    # Visualize outliers
    st.markdown('<h2 class="sub-header">Outlier Detection</h2>', unsafe_allow_html=True)

    # Select columns to visualize
    selected_cols = st.multiselect(
        "Select columns to visualize for outliers:",
        numeric_cols,
        default=numeric_cols[:min(4, len(numeric_cols))]
    )

    if selected_cols:
        # Box plots for selected columns
        fig, axes = plt.subplots(len(selected_cols), 1, figsize=(10, 4*len(selected_cols)))
        if len(selected_cols) == 1:
            axes = [axes]

        for i, col in enumerate(selected_cols):
            sns.boxplot(x=data[col], color='#FFA500', ax=axes[i])
            axes[i].set_title(f'Boxplot of {col}')

        plt.tight_layout()
        st.pyplot(fig)

    # Outlier handling options
    st.markdown('<h2 class="sub-header">Outlier Handling</h2>', unsafe_allow_html=True)

    detection_method = st.radio(
        "Choose an outlier detection method:",
        ["Z-Score", "IQR", "Isolation Forest", "Local Outlier Factor"]
    )

    if detection_method == "Z-Score":
        st.markdown("### Z-Score Method")
        threshold = st.slider("Z-Score threshold:", 1.0, 5.0, 3.0, 0.1)

        if st.button("Detect and Remove Outliers"):
            outlier_count = 0
            for col in selected_cols:
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outliers = z_scores > threshold
                outlier_count += outliers.sum()
                data = data[~outliers]

            st.success(f"Removed {outlier_count} outliers using Z-Score method with threshold {threshold}")
            st.session_state.processed_data = data
            st.rerun()

    elif detection_method == "IQR":
        st.markdown("### IQR Method")
        iqr_factor = st.slider("IQR factor:", 1.0, 3.0, 1.5, 0.1)

        if st.button("Detect and Remove Outliers"):
            outlier_count = 0
            for col in selected_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - iqr_factor * IQR
                upper_bound = Q3 + iqr_factor * IQR

                outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
                outlier_count += outliers.sum()
                data = data[~outliers]

            st.success(f"Removed {outlier_count} outliers using IQR method with factor {iqr_factor}")
            st.session_state.processed_data = data
            st.rerun()

    elif detection_method == "Isolation Forest":
        st.markdown("### Isolation Forest Method")
        contamination = st.slider("Contamination (expected proportion of outliers):", 0.01, 0.5, 0.1, 0.01)

        if st.button("Detect and Remove Outliers"):
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outliers = iso_forest.fit_predict(data[selected_cols])
            outlier_mask = outliers == -1
            outlier_count = outlier_mask.sum()
            data = data[~outlier_mask]

            st.success(f"Removed {outlier_count} outliers using Isolation Forest with contamination {contamination}")
            st.session_state.processed_data = data
            st.rerun()

    else:  # Local Outlier Factor
        st.markdown("### Local Outlier Factor Method")
        n_neighbors = st.slider("Number of neighbors:", 5, 50, 20, 5)
        contamination = st.slider("Contamination (expected proportion of outliers):", 0.01, 0.5, 0.1, 0.01)

        if st.button("Detect and Remove Outliers"):
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            outliers = lof.fit_predict(data[selected_cols])
            outlier_mask = outliers == -1
            outlier_count = outlier_mask.sum()
            data = data[~outlier_mask]

            st.success(f"Removed {outlier_count} outliers using Local Outlier Factor with {n_neighbors} neighbors")
            st.session_state.processed_data = data
            st.rerun()

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Previous Step", type="secondary"):
            st.session_state.current_step = 3
            st.rerun()
    with col2:
        if st.button("Next Step → Address Data Imbalance", type="primary"):
            if "Handle Outliers" not in st.session_state.steps_completed:
                st.session_state.steps_completed.append("Handle Outliers")
            st.session_state.current_step = 5
            st.rerun()

# Step 6: Address Data Imbalance
def address_imbalance_step():
    st.markdown('<h1 class="main-header">Address Data Imbalance</h1>', unsafe_allow_html=True)

    if st.session_state.processed_data is None:
        st.warning("Please complete the previous steps first!")
        return

    data = st.session_state.processed_data.copy()

    # Check if target variable is set
    if not st.session_state.target_variable or st.session_state.target_variable not in data.columns:
        st.warning("No target variable set or target variable not in data. Please set a target variable in the Explore Data step.")

        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Previous Step", type="secondary"):
                st.session_state.current_step = 4
                st.rerun()
        with col2:
            if st.button("Next Step → Feature Scaling", type="primary"):
                if "Address Data Imbalance" not in st.session_state.steps_completed:
                    st.session_state.steps_completed.append("Address Data Imbalance")
                st.session_state.current_step = 6
                st.rerun()
        return

    target = st.session_state.target_variable

    # Check if target is categorical or continuous
    is_categorical = not pd.api.types.is_numeric_dtype(data[target]) or data[target].nunique() < 20

    if not is_categorical:
        st.warning("Target variable appears to be continuous. Data imbalance techniques are typically used for classification problems.")

        # Option to convert to categorical
        if st.checkbox("Convert continuous target to categorical for imbalance handling"):
            method = st.radio("Choose a method to convert to categorical:", ["Equal-width binning", "Equal-frequency binning", "Custom thresholds"])

            if method == "Equal-width binning":
                n_bins = st.slider("Number of bins:", 2, 10, 5)
                if st.button("Convert and Continue"):
                    data[target] = pd.cut(data[target], bins=n_bins, labels=False)
                    st.success(f"Converted {target} to categorical with {n_bins} bins")
                    st.session_state.processed_data = data
                    st.rerun()

            elif method == "Equal-frequency binning":
                n_bins = st.slider("Number of bins:", 2, 10, 5)
                if st.button("Convert and Continue"):
                    data[target] = pd.qcut(data[target], q=n_bins, labels=False, duplicates='drop')
                    st.success(f"Converted {target} to categorical with {n_bins} bins")
                    st.session_state.processed_data = data
                    st.rerun()

            else:  # Custom thresholds
                thresholds = st.text_input("Enter comma-separated threshold values:", "0,25,50,75,100")
                if st.button("Convert and Continue"):
                    try:
                        thresh_values = [float(x.strip()) for x in thresholds.split(',')]
                        data[target] = pd.cut(data[target], bins=thresh_values, labels=False, include_lowest=True)
                        st.success(f"Converted {target} to categorical with custom thresholds")
                        st.session_state.processed_data = data
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error converting target: {e}")

        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Previous Step", type="secondary"):
                st.session_state.current_step = 4
                st.rerun()
        with col2:
            if st.button("Next Step → Feature Scaling", type="primary"):
                if "Address Data Imbalance" not in st.session_state.steps_completed:
                    st.session_state.steps_completed.append("Address Data Imbalance")
                st.session_state.current_step = 6
                st.rerun()
        return

    # Show class distribution
    st.markdown('<h2 class="sub-header">Class Distribution</h2>', unsafe_allow_html=True)

    class_counts = data[target].value_counts()
    class_percentages = data[target].value_counts(normalize=True) * 100

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(pd.DataFrame({
            'Class': class_counts.index,
            'Count': class_counts.values,
            'Percentage': class_percentages.values
        }))

    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        class_counts.plot(kind='bar', color='#FFA500', ax=ax)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    # Check if data is imbalanced
    imbalance_ratio = class_counts.max() / class_counts.min()
    if imbalance_ratio > 2:
        st.warning(f"Data appears to be imbalanced with a ratio of {imbalance_ratio:.2f}:1 between majority and minority classes.")
    else:
        st.success("Data appears to be balanced.")

    # Imbalance handling options
    st.markdown('<h2 class="sub-header">Imbalance Handling</h2>', unsafe_allow_html=True)

    method = st.radio(
        "Choose a method to handle imbalance:",
        ["None", "Oversampling", "Undersampling", "Combination"]
    )

    if method == "Oversampling":
        st.markdown("### Oversampling Techniques")
        technique = st.radio(
            "Choose an oversampling technique:",
            ["Random Oversampling", "SMOTE", "ADASYN"]
        )

        if technique == "Random Oversampling":
            if st.button("Apply Random Oversampling"):
                # Get the majority and minority classes
                majority_class = class_counts.idxmax()
                minority_classes = class_counts.index.drop(majority_class)

                # Resample minority classes to match majority class
                balanced_data = data.copy()
                for minority_class in minority_classes:
                    minority_data = data[data[target] == minority_class]
                    majority_count = class_counts[majority_class]
                    resampled_minority = resample(
                        minority_data,
                        replace=True,
                        n_samples=majority_count,
                        random_state=42
                    )
                    balanced_data = pd.concat([balanced_data, resampled_minority])

                # Remove original minority data
                balanced_data = balanced_data[~balanced_data.index.isin(
                    data[data[target].isin(minority_classes)].index
                )]

                st.session_state.processed_data = balanced_data
                st.success("Applied Random Oversampling")
                st.rerun()

        elif technique == "SMOTE":
            if st.button("Apply SMOTE"):
                X = data.drop(columns=[target])
                y = data[target]

                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y)

                balanced_data = pd.concat([X_resampled, y_resampled], axis=1)
                st.session_state.processed_data = balanced_data
                st.success("Applied SMOTE")
                st.rerun()

        else:  # ADASYN
            if st.button("Apply ADASYN"):
                X = data.drop(columns=[target])
                y = data[target]

                adasyn = ADASYN(random_state=42)
                X_resampled, y_resampled = adasyn.fit_resample(X, y)

                balanced_data = pd.concat([X_resampled, y_resampled], axis=1)
                st.session_state.processed_data = balanced_data
                st.success("Applied ADASYN")
                st.rerun()

    elif method == "Undersampling":
        st.markdown("### Undersampling Techniques")
        technique = st.radio(
            "Choose an undersampling technique:",
            ["Random Undersampling", "NearMiss"]
        )

        if technique == "Random Undersampling":
            if st.button("Apply Random Undersampling"):
                # Get the minority and majority classes
                minority_class = class_counts.idxmin()
                majority_classes = class_counts.index.drop(minority_class)

                # Resample majority classes to match minority class
                balanced_data = data[data[target] == minority_class].copy()
                for majority_class in majority_classes:
                    majority_data = data[data[target] == majority_class]
                    minority_count = class_counts[minority_class]
                    resampled_majority = resample(
                        majority_data,
                        replace=False,
                        n_samples=minority_count,
                        random_state=42
                    )
                    balanced_data = pd.concat([balanced_data, resampled_majority])

                st.session_state.processed_data = balanced_data
                st.success("Applied Random Undersampling")
                st.rerun()

        else:  # NearMiss
            version = st.radio("Choose NearMiss version:", [1, 2, 3])
            if st.button(f"Apply NearMiss-{version}"):
                X = data.drop(columns=[target])
                y = data[target]

                nearmiss = NearMiss(version=version)
                X_resampled, y_resampled = nearmiss.fit_resample(X, y)

                balanced_data = pd.concat([X_resampled, y_resampled], axis=1)
                st.session_state.processed_data = balanced_data
                st.success(f"Applied NearMiss-{version}")
                st.rerun()

    elif method == "Combination":
        st.markdown("### Combination Techniques")
        technique = st.radio(
            "Choose a combination technique:",
            ["SMOTEENN", "SMOTETomek"]
        )

        if technique == "SMOTEENN":
            if st.button("Apply SMOTEENN"):
                X = data.drop(columns=[target])
                y = data[target]

                smoteenn = SMOTEENN(random_state=42)
                X_resampled, y_resampled = smoteenn.fit_resample(X, y)

                balanced_data = pd.concat([X_resampled, y_resampled], axis=1)
                st.session_state.processed_data = balanced_data
                st.success("Applied SMOTEENN")
                st.rerun()

        else:  # SMOTETomek
            if st.button("Apply SMOTETomek"):
                X = data.drop(columns=[target])
                y = data[target]

                smotetomek = SMOTETomek(random_state=42)
                X_resampled, y_resampled = smotetomek.fit_resample(X, y)

                balanced_data = pd.concat([X_resampled, y_resampled], axis=1)
                st.session_state.processed_data = balanced_data
                st.success("Applied SMOTETomek")
                st.rerun()

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Previous Step", type="secondary"):
            st.session_state.current_step = 4
            st.rerun()
    with col2:
        if st.button("Next Step → Feature Scaling", type="primary"):
            if "Address Data Imbalance" not in st.session_state.steps_completed:
                st.session_state.steps_completed.append("Address Data Imbalance")
            st.session_state.current_step = 6
            st.rerun()

# Step 7: Feature Scaling
def feature_scaling_step():
    st.markdown('<h1 class="main-header">Feature Scaling</h1>', unsafe_allow_html=True)

    if st.session_state.processed_data is None:
        st.warning("Please complete the previous steps first!")
        return

    data = st.session_state.processed_data.copy()
    numeric_cols = data.select_dtypes(include=[np.number]).columns

    # Exclude target variable if it's set
    if st.session_state.target_variable and st.session_state.target_variable in numeric_cols:
        numeric_cols = numeric_cols.drop(st.session_state.target_variable)

    if len(numeric_cols) == 0:
        st.warning("No numeric columns found for scaling!")
        if "Feature Scaling" not in st.session_state.steps_completed:
            st.session_state.steps_completed.append("Feature Scaling")
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Previous Step", type="secondary"):
                st.session_state.current_step = 5
                st.rerun()
        with col2:
            if st.button("Next Step → Correlation Analysis", type="primary"):
                st.session_state.current_step = 7
                st.rerun()
        return

    # Visualize distributions before scaling
    st.markdown('<h2 class="sub-header">Feature Distributions Before Scaling</h2>', unsafe_allow_html=True)

    # Select columns to visualize
    selected_cols = st.multiselect(
        "Select columns to visualize:",
        numeric_cols,
        default=numeric_cols[:min(4, len(numeric_cols))]
    )

    if selected_cols:
        # Histograms for selected columns
        fig, axes = plt.subplots(len(selected_cols), 1, figsize=(10, 4*len(selected_cols)))
        if len(selected_cols) == 1:
            axes = [axes]

        for i, col in enumerate(selected_cols):
            sns.histplot(data[col], kde=True, color='#FFA500', ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')

        plt.tight_layout()
        st.pyplot(fig)

    # Scaling options
    st.markdown('<h2 class="sub-header">Scaling Options</h2>', unsafe_allow_html=True)

    scaling_method = st.radio(
        "Choose a scaling method:",
        ["StandardScaler (Z-score normalization)", "MinMaxScaler (0-1 scaling)", "RobustScaler", "None"]
    )

    if scaling_method == "StandardScaler (Z-score normalization)":
        st.markdown("### StandardScaler (Z-score normalization)")
        st.info("StandardScaler removes the mean and scales to unit variance. It's a good choice when the data follows a Gaussian distribution.")

        if st.button("Apply StandardScaler"):
            scaler = StandardScaler()
            data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
            st.session_state.processed_data = data
            st.success("Applied StandardScaler")
            st.rerun()

    elif scaling_method == "MinMaxScaler (0-1 scaling)":
        st.markdown("### MinMaxScaler (0-1 scaling)")
        st.info("MinMaxScaler scales features to a given range, typically [0, 1]. It's useful when you need bounded values.")

        if st.button("Apply MinMaxScaler"):
            scaler = MinMaxScaler()
            data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
            st.session_state.processed_data = data
            st.success("Applied MinMaxScaler")
            st.rerun()

    elif scaling_method == "RobustScaler":
        st.markdown("### RobustScaler")
        st.info("RobustScaler scales features using statistics that are robust to outliers. It uses the interquartile range.")

        if st.button("Apply RobustScaler"):
            scaler = RobustScaler()
            data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
            st.session_state.processed_data = data
            st.success("Applied RobustScaler")
            st.rerun()

    # Show distributions after scaling if scaling has been applied
    if scaling_method != "None" and st.button("Show Distributions After Scaling"):
        # Histograms for selected columns
        fig, axes = plt.subplots(len(selected_cols), 1, figsize=(10, 4*len(selected_cols)))
        if len(selected_cols) == 1:
            axes = [axes]

        for i, col in enumerate(selected_cols):
            sns.histplot(data[col], kde=True, color='#FFA500', ax=axes[i])
            axes[i].set_title(f'Distribution of {col} After Scaling')

        plt.tight_layout()
        st.pyplot(fig)

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Previous Step", type="secondary"):
            st.session_state.current_step = 5
            st.rerun()
    with col2:
        if st.button("Next Step → Correlation Analysis", type="primary"):
            if "Feature Scaling" not in st.session_state.steps_completed:
                st.session_state.steps_completed.append("Feature Scaling")
            st.session_state.current_step = 7
            st.rerun()

# Step 8: Correlation Analysis
def correlation_analysis_step():
    st.markdown('<h1 class="main-header">Correlation Analysis</h1>', unsafe_allow_html=True)

    if st.session_state.processed_data is None:
        st.warning("Please complete the previous steps first!")
        return

    data = st.session_state.processed_data.copy()
    numeric_cols = data.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) < 2:
        st.warning("Not enough numeric columns for correlation analysis!")
        if "Correlation Analysis" not in st.session_state.steps_completed:
            st.session_state.steps_completed.append("Correlation Analysis")
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Previous Step", type="secondary"):
                st.session_state.current_step = 6
                st.rerun()
        with col2:
            if st.button("Next Step → Feature Engineering", type="primary"):
                st.session_state.current_step = 8
                st.rerun()
        return

    # Calculate correlation matrix
    corr_matrix = data[numeric_cols].corr()

    # Display correlation heatmap
    st.markdown('<h2 class="sub-header">Correlation Matrix</h2>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=0.5,
        ax=ax
    )
    plt.title('Correlation Matrix')
    plt.tight_layout()
    st.pyplot(fig)

    # Target correlation if target is set
    if st.session_state.target_variable and st.session_state.target_variable in numeric_cols:
        st.markdown('<h2 class="sub-header">Correlation with Target Variable</h2>', unsafe_allow_html=True)

        target = st.session_state.target_variable
        target_corr = corr_matrix[target].abs().sort_values(ascending=False)

        # Plot correlations with target
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#FFA500' if x > 0 else '#1f77b4' for x in corr_matrix[target].drop(target)]
        corr_matrix[target].drop(target).sort_values().plot(kind='bar', color=colors, ax=ax)
        plt.title(f'Feature Correlation with {target}')
        plt.ylabel('Correlation Coefficient')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig)

        # Show top correlated features
        st.markdown("### Top Features Correlated with Target")
        st.dataframe(target_corr.drop(target).head(10))

    # Feature selection based on correlation
    st.markdown('<h2 class="sub-header">Feature Selection Based on Correlation</h2>', unsafe_allow_html=True)

    # Find highly correlated pairs
    threshold = st.slider("Correlation threshold for feature selection:", 0.5, 0.95, 0.8, 0.05)

    # Find pairs of features with correlation above threshold
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
        st.write(f"Found {len(high_corr_pairs)} feature pairs with correlation above {threshold}:")
        st.dataframe(pd.DataFrame(high_corr_pairs, columns=['Feature 1', 'Feature 2', 'Correlation']))

        # Option to drop features
        if st.checkbox("Show options to drop highly correlated features"):
            # Create a list of features to potentially drop
            features_to_drop = set()
            for feat1, feat2, _ in high_corr_pairs:
                # If target is set, prefer to keep the feature more correlated with target
                if st.session_state.target_variable and st.session_state.target_variable in numeric_cols:
                    target = st.session_state.target_variable
                    corr1 = abs(corr_matrix.loc[feat1, target])
                    corr2 = abs(corr_matrix.loc[feat2, target])
                    if corr1 < corr2:
                        features_to_drop.add(feat1)
                    else:
                        features_to_drop.add(feat2)
                else:
                    # Otherwise, just drop the second feature in each pair
                    features_to_drop.add(feat2)

            # Show features to drop
            st.write("Suggested features to drop:")
            selected_features = st.multiselect(
                "Select features to drop:",
                list(features_to_drop),
                default=list(features_to_drop)
            )

            if st.button("Drop Selected Features"):
                data = data.drop(columns=selected_features)
                st.session_state.processed_data = data
                st.success(f"Dropped {len(selected_features)} features")
                st.rerun()
    else:
        st.success(f"No feature pairs found with correlation above {threshold}")

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Previous Step", type="secondary"):
            st.session_state.current_step = 6
            st.rerun()
    with col2:
        if st.button("Next Step → Feature Engineering", type="primary"):
            if "Correlation Analysis" not in st.session_state.steps_completed:
                st.session_state.steps_completed.append("Correlation Analysis")
            st.session_state.current_step = 8
            st.rerun()

# Step 9: Feature Engineering
def feature_engineering_step():
    st.markdown('<h1 class="main-header">Feature Engineering</h1>', unsafe_allow_html=True)

    if st.session_state.processed_data is None:
        st.warning("Please complete the previous steps first!")
        return

    data = st.session_state.processed_data.copy()

    # Feature engineering options
    st.markdown('<h2 class="sub-header">Feature Engineering Options</h2>', unsafe_allow_html=True)

    engineering_type = st.radio(
        "Choose a feature engineering technique:",
        ["Polynomial Features", "Interaction Features", "Binning", "Log Transformation", "None"]
    )

    if engineering_type == "Polynomial Features":
        st.markdown("### Polynomial Features")
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        # Exclude target variable if it's set
        if st.session_state.target_variable and st.session_state.target_variable in numeric_cols:
            numeric_cols = numeric_cols.drop(st.session_state.target_variable)

        if len(numeric_cols) == 0:
            st.warning("No numeric columns available for polynomial features!")
        else:
            selected_cols = st.multiselect(
                "Select columns for polynomial features:",
                numeric_cols,
                default=numeric_cols[:min(2, len(numeric_cols))]
            )

            degree = st.slider("Polynomial degree:", 2, 4, 2)
            include_bias = st.checkbox("Include bias term", value=False)

            if st.button("Create Polynomial Features"):
                from sklearn.preprocessing import PolynomialFeatures

                poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
                poly_features = poly.fit_transform(data[selected_cols])

                # Create feature names
                feature_names = poly.get_feature_names_out(selected_cols)

                # Create a DataFrame with the new features
                poly_df = pd.DataFrame(poly_features, columns=feature_names, index=data.index)

                # Add the new features to the data
                data = pd.concat([data, poly_df], axis=1)

                st.session_state.processed_data = data
                st.success(f"Created {len(feature_names)} polynomial features")
                st.rerun()

    elif engineering_type == "Interaction Features":
        st.markdown("### Interaction Features")
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        # Exclude target variable if it's set
        if st.session_state.target_variable and st.session_state.target_variable in numeric_cols:
            numeric_cols = numeric_cols.drop(st.session_state.target_variable)

        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for interaction features!")
        else:
            selected_cols = st.multiselect(
                "Select columns for interaction features:",
                numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))]
            )

            if len(selected_cols) >= 2:
                if st.button("Create Interaction Features"):
                    # Create interaction features
                    for i in range(len(selected_cols)):
                        for j in range(i+1, len(selected_cols)):
                            feat1 = selected_cols[i]
                            feat2 = selected_cols[j]
                            interaction_name = f"{feat1}_x_{feat2}"
                            data[interaction_name] = data[feat1] * data[feat2]

                    st.session_state.processed_data = data
                    st.success(f"Created {len(selected_cols) * (len(selected_cols) - 1) // 2} interaction features")
                    st.rerun()

    elif engineering_type == "Binning":
        st.markdown("### Binning")
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        # Exclude target variable if it's set
        if st.session_state.target_variable and st.session_state.target_variable in numeric_cols:
            numeric_cols = numeric_cols.drop(st.session_state.target_variable)

        if len(numeric_cols) == 0:
            st.warning("No numeric columns available for binning!")
        else:
            selected_col = st.selectbox("Select column for binning:", numeric_cols)

            binning_method = st.radio("Choose binning method:", ["Equal-width", "Equal-frequency", "Custom"])

            if binning_method == "Equal-width":
                n_bins = st.slider("Number of bins:", 2, 20, 5)

                if st.button("Apply Equal-width Binning"):
                    bin_name = f"{selected_col}_binned"
                    data[bin_name] = pd.cut(data[selected_col], bins=n_bins, labels=False)

                    st.session_state.processed_data = data
                    st.success(f"Created {bin_name} with {n_bins} equal-width bins")
                    st.rerun()

            elif binning_method == "Equal-frequency":
                n_bins = st.slider("Number of bins:", 2, 20, 5)

                if st.button("Apply Equal-frequency Binning"):
                    bin_name = f"{selected_col}_binned"
                    data[bin_name] = pd.qcut(data[selected_col], q=n_bins, labels=False, duplicates='drop')

                    st.session_state.processed_data = data
                    st.success(f"Created {bin_name} with {n_bins} equal-frequency bins")
                    st.rerun()

            else:  # Custom
                thresholds = st.text_input("Enter comma-separated threshold values:", "0,25,50,75,100")

                if st.button("Apply Custom Binning"):
                    try:
                        thresh_values = [float(x.strip()) for x in thresholds.split(',')]
                        bin_name = f"{selected_col}_binned"
                        data[bin_name] = pd.cut(data[selected_col], bins=thresh_values, labels=False, include_lowest=True)

                        st.session_state.processed_data = data
                        st.success(f"Created {bin_name} with custom thresholds")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error creating bins: {e}")

    elif engineering_type == "Log Transformation":
        st.markdown("### Log Transformation")
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        # Exclude target variable if it's set
        if st.session_state.target_variable and st.session_state.target_variable in numeric_cols:
            numeric_cols = numeric_cols.drop(st.session_state.target_variable)

        if len(numeric_cols) == 0:
            st.warning("No numeric columns available for log transformation!")
        else:
            selected_cols = st.multiselect(
                "Select columns for log transformation:",
                numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))]
            )

            log_base = st.radio("Log base:", ["Natural log (e)", "Base 10", "Base 2"])

            if st.button("Apply Log Transformation"):
                for col in selected_cols:
                    # Check if all values are positive
                    if (data[col] <= 0).any():
                        # Add a small constant to make all values positive
                        transformed_col = np.log(data[col] - data[col].min() + 1)
                    else:
                        if log_base == "Natural log (e)":
                            transformed_col = np.log(data[col])
                        elif log_base == "Base 10":
                            transformed_col = np.log10(data[col])
                        else:  # Base 2
                            transformed_col = np.log2(data[col])

                    data[f"{col}_log"] = transformed_col

                st.session_state.processed_data = data
                st.success(f"Applied log transformation to {len(selected_cols)} columns")
                st.rerun()

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Previous Step", type="secondary"):
            st.session_state.current_step = 7
            st.rerun()
    with col2:
        if st.button("Next Step → Visualize Results", type="primary"):
            if "Feature Engineering" not in st.session_state.steps_completed:
                st.session_state.steps_completed.append("Feature Engineering")
            st.session_state.current_step = 9
            st.rerun()

# Step 10: Visualize Results
def visualize_results_step():
    st.markdown('<h1 class="main-header">Visualize Results</h1>', unsafe_allow_html=True)

    if st.session_state.processed_data is None:
        st.warning("Please complete the previous steps first!")
        return

    original_data = st.session_state.original_data
    processed_data = st.session_state.processed_data

    # Dataset comparison
    st.markdown('<h2 class="sub-header">Dataset Comparison</h2>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Original Rows", original_data.shape[0])
    with col2:
        st.metric("Processed Rows", processed_data.shape[0])
    with col3:
        row_change = processed_data.shape[0] - original_data.shape[0]
        st.metric("Row Change", f"{row_change:+d}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Original Columns", original_data.shape[1])
    with col2:
        st.metric("Processed Columns", processed_data.shape[1])
    with col3:
        col_change = processed_data.shape[1] - original_data.shape[1]
        st.metric("Column Change", f"{col_change:+d}")

    # Missing values comparison
    st.markdown('<h2 class="sub-header">Missing Values Comparison</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Before")
        original_missing = original_data.isnull().sum().sum()
        st.write(f"Total missing values: {original_missing}")

        # Visualize missing values
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(original_data.isnull(), cbar=False, cmap='viridis', ax=ax)
        plt.title('Missing Values Before Processing')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("After")
        processed_missing = processed_data.isnull().sum().sum()
        st.write(f"Total missing values: {processed_missing}")

        # Visualize missing values
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(processed_data.isnull(), cbar=False, cmap='viridis', ax=ax)
        plt.title('Missing Values After Processing')
        plt.tight_layout()
        st.pyplot(fig)

    # Data distribution comparison
    st.markdown('<h2 class="sub-header">Data Distribution Comparison</h2>', unsafe_allow_html=True)

    # Find common numeric columns
    original_numeric = original_data.select_dtypes(include=[np.number]).columns
    processed_numeric = processed_data.select_dtypes(include=[np.number]).columns
    common_numeric = original_numeric.intersection(processed_numeric)

    if len(common_numeric) > 0:
        # Select columns to compare
        selected_cols = st.multiselect(
            "Select columns to compare:",
            common_numeric,
            default=common_numeric[:min(3, len(common_numeric))]
        )

        if selected_cols:
            for col in selected_cols:
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))

                # Before
                sns.histplot(original_data[col].dropna(), kde=True, color='#FFA500', ax=axes[0])
                axes[0].set_title(f'{col} Distribution (Before)')

                # After
                sns.histplot(processed_data[col].dropna(), kde=True, color='#1f77b4', ax=axes[1])
                axes[1].set_title(f'{col} Distribution (After)')

                plt.tight_layout()
                st.pyplot(fig)
    else:
        st.warning("No common numeric columns found for comparison.")

    # Target variable comparison if set
    if st.session_state.target_variable and st.session_state.target_variable in processed_data.columns:
        st.markdown('<h2 class="sub-header">Target Variable Comparison</h2>', unsafe_allow_html=True)

        target = st.session_state.target_variable

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Before")
            if target in original_data.columns:
                if pd.api.types.is_numeric_dtype(original_data[target]):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(original_data[target].dropna(), kde=True, color='#FFA500', ax=ax)
                    plt.title(f'{target} Distribution (Before)')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.countplot(y=original_data[target], color='#FFA500', ax=ax)
                    plt.title(f'{target} Distribution (Before)')
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.warning("Target variable not found in original data.")

        with col2:
            st.subheader("After")
            if pd.api.types.is_numeric_dtype(processed_data[target]):
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(processed_data[target].dropna(), kde=True, color='#1f77b4', ax=ax)
                plt.title(f'{target} Distribution (After)')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.countplot(y=processed_data[target], color='#1f77b4', ax=ax)
                plt.title(f'{target} Distribution (After)')
                plt.tight_layout()
                st.pyplot(fig)

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Previous Step", type="secondary"):
            st.session_state.current_step = 8
            st.rerun()
    with col2:
        if st.button("Next Step → Export Data", type="primary"):
            if "Visualize Results" not in st.session_state.steps_completed:
                st.session_state.steps_completed.append("Visualize Results")
            st.session_state.current_step = 10
            st.rerun()

# Step 11: Export Data
def export_data_step():
    st.markdown('<h1 class="main-header">Export Processed Data</h1>', unsafe_allow_html=True)

    if st.session_state.processed_data is None:
        st.warning("Please complete the previous steps first!")
        return

    data = st.session_state.processed_data

    # Data preview
    st.markdown('<h2 class="sub-header">Data Preview</h2>', unsafe_allow_html=True)

    st.dataframe(data.head())

    # Data summary
    st.markdown('<h2 class="sub-header">Data Summary</h2>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", data.shape[0])
    with col2:
        st.metric("Columns", data.shape[1])
    with col3:
        st.metric("Missing Values", data.isnull().sum().sum())

    # Export options
    st.markdown('<h2 class="sub-header">Export Options</h2>', unsafe_allow_html=True)

    export_format = st.radio("Choose export format:", ["CSV", "Excel", "JSON"])

    if export_format == "CSV":
        filename = st.text_input("Filename:", "processed_data.csv")

        if st.button("Download CSV"):
            csv = data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
            st.markdown(href, unsafe_allow_html=True)

    elif export_format == "Excel":
        filename = st.text_input("Filename:", "processed_data.xlsx")

        if st.button("Download Excel"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                data.to_excel(writer, index=False, sheet_name='Processed Data')

                # Get the workbook and the worksheet
                workbook = writer.book
                worksheet = writer.sheets['Processed Data']

                # Add some formatting
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#FFA500',
                    'border': 1
                })

                # Write the column headers with the defined format
                for col_num, value in enumerate(data.columns.values):
                    worksheet.write(0, col_num, value, header_format)

            b64 = base64.b64encode(output.getvalue()).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download {filename}</a>'
            st.markdown(href, unsafe_allow_html=True)

    else:  # JSON
        filename = st.text_input("Filename:", "processed_data.json")

        if st.button("Download JSON"):
            json = data.to_json(orient='records')
            b64 = base64.b64encode(json.encode()).decode()
            href = f'<a href="data:application/json;base64,{b64}" download="{filename}">Download {filename}</a>'
            st.markdown(href, unsafe_allow_html=True)

    # ML pipeline preparation
    st.markdown('<h2 class="sub-header">ML Pipeline Preparation</h2>', unsafe_allow_html=True)

    st.info("Your data is now ready for ML pipelines! Here's a summary of what was done:")

    # Create a summary of steps completed
    steps_summary = []
    if "Handle Missing Values" in st.session_state.steps_completed:
        steps_summary.append("✅ Missing values handled")
    if "Encode Categorical Variables" in st.session_state.steps_completed:
        steps_summary.append("✅ Categorical variables encoded")
    if "Handle Outliers" in st.session_state.steps_completed:
        steps_summary.append("✅ Outliers handled")
    if "Address Data Imbalance" in st.session_state.steps_completed:
        steps_summary.append("✅ Data imbalance addressed")
    if "Feature Scaling" in st.session_state.steps_completed:
        steps_summary.append("✅ Features scaled")
    if "Correlation Analysis" in st.session_state.steps_completed:
        steps_summary.append("✅ Correlation analysis completed")
    if "Feature Engineering" in st.session_state.steps_completed:
        steps_summary.append("✅ Features engineered")

    for step in steps_summary:
        st.write(step)

    # Code snippet for loading data in ML pipeline
    st.markdown('<h2 class="sub-header">Code for ML Pipeline</h2>', unsafe_allow_html=True)

    code_snippet = f"""
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the processed data
data = pd.read_csv('{filename if export_format == "CSV" else "processed_data.csv"}')

# Define features and target
X = data.drop(columns=['{st.session_state.target_variable if st.session_state.target_variable else "target"}'])
y = data['{st.session_state.target_variable if st.session_state.target_variable else "target"}']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now you can train your ML model
# model = YourModel()
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)
"""

    st.code(code_snippet, language='python')

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Previous Step", type="secondary"):
            st.session_state.current_step = 9
            st.rerun()
    with col2:
        if st.button("Start New Project", type="primary"):
            for key in st.session_state.keys():
                del st.session_state[key]
            init_session_state()
            st.rerun()

# Main function to run the app
def main():
    # Initialize session state
    init_session_state()

    # Render sidebar
    render_sidebar()

    # Display current step
    st.markdown(f'<h2 class="sub-header">Step {st.session_state.current_step + 1}: {steps[st.session_state.current_step]}</h2>', unsafe_allow_html=True)

    # Render current step content
    if st.session_state.current_step == 0:
        upload_data_step()
    elif st.session_state.current_step == 1:
        explore_data_step()
    elif st.session_state.current_step == 2:
        handle_missing_values_step()
    elif st.session_state.current_step == 3:
        encode_categorical_step()
    elif st.session_state.current_step == 4:
        handle_outliers_step()
    elif st.session_state.current_step == 5:
        address_imbalance_step()
    elif st.session_state.current_step == 6:
        feature_scaling_step()
    elif st.session_state.current_step == 7:
        correlation_analysis_step()
    elif st.session_state.current_step == 8:
        feature_engineering_step()
    elif st.session_state.current_step == 9:
        visualize_results_step()
    elif st.session_state.current_step == 10:
        export_data_step()

# Run the app
if __name__ == "__main__":
    main()

