import streamlit as st
import pandas as pd
import numpy as np

# TensorFlow import with error handling for Streamlit Cloud
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    st.error(f"TensorFlow import failed: {e}")
    TENSORFLOW_AVAILABLE = False
    # Fallback: create dummy load_model function
    def load_model(path, compile=True):
        st.error("TensorFlow not available. Model loading disabled.")
        return None

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import hashlib
import warnings
import os
import time
import base64

warnings.filterwarnings('ignore')

# =============================================================================
# ⚙️ KONFIGURASI APLIKASI GLOBAL (HARUS DI BAGIAN ATAS DAN HANYA SEKALI)
# =============================================================================
st.set_page_config(
    page_title="Industrial Gas Removal Monitoring System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 🔧 KONFIGURASI FILE CSV DAN AUTO-UPDATE
# =============================================================================

CSV_FILE_NAME = "data2parfull_cleaned.csv"
CSV_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), CSV_FILE_NAME)
DEFAULT_UPDATE_INTERVAL = 10800  # 3 hours

# =============================================================================
# 🖼️ BACKGROUND IMAGE CONFIGURATION
# =============================================================================

def get_background_image_base64(image_path):
    """Convert image to base64 string for CSS background"""
    try:
        # Check multiple possible paths for cloud deployment
        possible_paths = [
            image_path,
            os.path.join(os.getcwd(), "foto.jpg"),
            "foto.jpg",
            "./foto.jpg"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, "rb") as img_file:
                    encoded = base64.b64encode(img_file.read()).decode()
                    return encoded
        
        return None
    except Exception as e:
        return None

# Background image path - CHANGED TO foto.jpg
BACKGROUND_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "foto.jpg")

# =============================================================================
# 🔐 SECURE AUTHENTICATION SYSTEM
# =============================================================================

def hash_password(password):
    """Secure password hashing using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

USER_CREDENTIALS = {
    "engineer": hash_password("engineer123"),
    "supervisor": hash_password("supervisor123"),
    "admin": hash_password("admin123"),
}

USER_ROLES = {
    "engineer": {
        "name": "Plant Engineer", 
        "role": "Engineer",
        "department": "Process Engineering",
        "permissions": ["view", "analyze"]
    },
    "supervisor": {
        "name": "Operations Supervisor", 
        "role": "Supervisor",
        "department": "Operations",
        "permissions": ["view", "analyze", "export"]
    },
    "admin": {
        "name": "System Administrator", 
        "role": "Administrator",
        "department": "IT & Maintenance",
        "permissions": ["view", "analyze", "export", "configure"]
    },
}

def check_authentication():
    return st.session_state.get('authenticated', False)

def authenticate_user(username, password):
    if username in USER_CREDENTIALS:
        return USER_CREDENTIALS[username] == hash_password(password)
    return False

# =============================================================================
# 🔄 AUTO-UPDATE FUNCTIONS
# =============================================================================

def get_current_time():
    """Get current time"""
    return datetime.now()

def init_session_state():
    """Initialize session state variables for auto-update functionality"""
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = get_current_time()
    
    if 'update_interval' not in st.session_state:
        st.session_state.update_interval = DEFAULT_UPDATE_INTERVAL
    
    if 'auto_update_enabled' not in st.session_state:
        st.session_state.auto_update_enabled = True
    
    if 'csv_data' not in st.session_state:
        st.session_state.csv_data = None
    
    if 'selected_interval_label' not in st.session_state:
        st.session_state.selected_interval_label = '3 hours'

@st.cache_data(ttl=DEFAULT_UPDATE_INTERVAL)
def load_csv_automatically():
    """Load CSV automatically from local file with caching"""
    try:
        if not os.path.exists(CSV_FILE_PATH):
            st.error(f"CSV file not found: {CSV_FILE_PATH}")
            return None
        
        delimiters = [',', ';', '\t']
        df = None
        
        for delimiter in delimiters:
            try:
                df = pd.read_csv(CSV_FILE_PATH, sep=delimiter)
                if len(df.columns) > 1:
                    break
            except Exception:
                continue
        
        if df is None:
            st.error("Failed to read CSV file with any delimiter")
            
        return df
        
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

def check_and_update():
    """Check if it's time to update data"""
    current_time = get_current_time()
    time_diff = (current_time - st.session_state.last_update_time).total_seconds()
    
    if time_diff >= st.session_state.update_interval and st.session_state.auto_update_enabled:
        st.session_state.last_update_time = current_time
        load_csv_automatically.clear() 
        st.rerun()

def format_time_remaining():
    """Format time remaining until next update"""
    current_time = get_current_time()
    time_diff = (current_time - st.session_state.last_update_time).total_seconds()
    time_remaining = st.session_state.update_interval - time_diff
    
    if time_remaining <= 0:
        return "Update pending..."
    
    hours = int(time_remaining // 3600)
    minutes = int((time_remaining % 3600) // 60)
    seconds = int(time_remaining % 60)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def login_page():
    """Professional login page with background image - FIXED VERSION"""
    
    # Get background image
    background_base64 = get_background_image_base64(BACKGROUND_IMAGE_PATH)
    background_opacity = 0.5
    
    # Debug info for development
    if background_base64:
        st.success("✅ Background image loaded successfully!")
    else:
        st.error(f"❌ Could not load background image: foto.jpg")
        st.info("Please make sure foto.jpg exists in the same folder as your Python script")
    
    # Create background CSS - FIXED to match working version
    if background_base64:
        main_background = f"""
            background: 
                linear-gradient(135deg, rgba(30, 60, 114, {background_opacity}) 0%, rgba(42, 82, 152, {background_opacity}) 100%),
                url('data:image/jpeg;base64,{background_base64}') center/cover no-repeat fixed;
            min-height: 100vh;
        """
    else:
        # Fallback to gradient if image not found
        main_background = """
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
        """
    
    # Professional login styling - BASED ON WORKING VERSION
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        .stApp > div:first-child {{
            {main_background}
            font-family: 'Inter', sans-serif;
        }}
        
        .main {{
            {main_background}
            font-family: 'Inter', sans-serif;
        }}
        
        div[data-testid="stAppViewContainer"] {{
            {main_background}
            font-family: 'Inter', sans-serif;
        }}
        
        .login-container {{
            max-width: 500px;
            margin: 3% auto;
            padding: 50px;
            background: rgba(255, 255, 255, 0.85);
            border-radius: 25px;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(15px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}
        
        .company-header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        
        .company-logo {{
            font-size: 3rem;
            color: #1e3c72;
            margin-bottom: 10px;
        }}
        
        .company-title {{
            color: #1e3c72 !important;
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 5px;
        }}
        
        .system-subtitle {{
            color: #666;
            font-size: 1rem;
            font-weight: 400;
        }}
        
        .login-form {{
            margin-top: 30px;
        }}
        
        .stForm {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
        }}
        
        .stTextInput > div > div > input {{
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 12px;
            padding: 15px 20px;
            font-size: 1rem;
            background: rgba(255, 255, 255, 0.9);
            color: #333;
            transition: all 0.3s ease;
        }}
        
        .stTextInput > div > div > input:focus {{
            border-color: #1e3c72;
            box-shadow: 0 0 0 3px rgba(30, 60, 114, 0.2);
            background: rgba(255, 255, 255, 1);
        }}
        
        .stButton > button {{
            width: 100%;
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            border: none;
            border-radius: 12px;
            color: white;
            font-weight: 600;
            font-size: 1.1rem;
            padding: 15px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(30, 60, 114, 0.3);
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(30, 60, 114, 0.4);
            background: linear-gradient(135deg, #2a5298, #1e3c72);
        }}
        
        .stForm h3 {{
            color: white;
            text-align: center;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }}
        
        .stForm p {{
            color: rgba(255, 255, 255, 0.9);
            text-align: center;
            margin-bottom: 20px;
        }}
        
        .stTextInput label {{
            color: white !important;
            font-weight: 500;
            margin-bottom: 5px;
        }}
        
        .security-footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #e1e5e9;
            color: #666;
            font-size: 0.9rem;
        }}
        
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        
        .login-container ~ div {{
            display: none;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    # Login container - EXACT STRUCTURE FROM WORKING VERSION
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="login-container">
            <div class="company-header">
                <div class="company-logo">🏭</div>
                <h1 class="company-title" style="color: #1e3c72 !important;">Industrial Monitoring System</h1>
                <p class="system-subtitle">Gas Removal Predictive Maintenance</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("secure_login"):
            st.markdown("### 🔐 Secure Access")
            st.markdown("Please authenticate to access the industrial monitoring system.")
            
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
            
            login_button = st.form_submit_button("🚀 Access System")
            
            if login_button:
                if username and password:
                    if authenticate_user(username, password):
                        st.session_state['authenticated'] = True
                        st.session_state['username'] = username
                        st.session_state['user_info'] = USER_ROLES.get(username, {})
                        st.success("✅ Authentication successful! Loading system...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials. Access denied.")
                else:
                    st.warning("⚠️ Please provide both username and password.")
        
        st.markdown("""
        <div class="security-footer">
            <p>🔒 Secure Industrial System Access</p>
            <p><strong>Demo Credentials:</strong><br>
            engineer/engineer123 | supervisor/supervisor123 | admin/admin123</p>
            <p style="font-size: 0.8rem; margin-top: 0.5rem; color: #888;">
                Required files: foto.jpg, best_lstm_model.h5, data2parfull_cleaned.csv
            </p>
        </div>
        """, unsafe_allow_html=True)

def logout():
    """Logout user and clear session state"""
    for key in ['authenticated', 'username', 'user_info', 'csv_data', 'last_update_time']:
        if key in st.session_state:
            del st.session_state[key]
    load_csv_automatically.clear() 
    st.rerun()

def show_user_panel():
    """Professional user information panel with auto-update controls"""
    if 'user_info' in st.session_state:
        user_info = st.session_state['user_info']
        st.sidebar.markdown("### 👤 User Profile")
        
        st.sidebar.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <div style="color: white; font-weight: 600; font-size: 1.1rem;">
                {user_info.get('name', 'User')}
            </div>
            <div style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                {user_info.get('role', 'User')} • {user_info.get('department', 'General')}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        permissions = user_info.get('permissions', [])
        st.sidebar.markdown("**Access Level:**")
        for perm in permissions:
            st.sidebar.markdown(f"✅ {perm.title()}")
        
        st.sidebar.markdown("---")
        
        st.sidebar.markdown("### 🔄 Auto-Update Settings")
        
        st.session_state.auto_update_enabled = st.sidebar.checkbox(
            "Enable Auto-Update",
            value=st.session_state.auto_update_enabled,
            help="Automatically refresh data at specified intervals"
        )
        
        update_options = {
            "1 minute": 60,
            "5 minutes": 300,
            "15 minutes": 900,
            "30 minutes": 1800,
            "1 hour": 3600,
            "3 hours": 10800,
            "6 hours": 21600,
            "12 hours": 43200,
            "24 hours": 86400
        }
        
        current_interval_value = st.session_state.get('update_interval', DEFAULT_UPDATE_INTERVAL)
        
        default_index = 0
        for idx, (label, value) in enumerate(update_options.items()):
            if value == current_interval_value:
                default_index = idx
                break
        
        selected_interval = st.sidebar.selectbox(
            "Update Interval",
            options=list(update_options.keys()),
            index=default_index,
            key="update_interval_selector",
            help="How often to refresh the data"
        )
        
        if update_options[selected_interval] != st.session_state.update_interval:
            st.session_state.update_interval = update_options[selected_interval]
            load_csv_automatically.clear() 
            st.rerun()

        st.session_state.selected_interval_label = selected_interval
        
        st.sidebar.markdown("**Auto-Update Status:**")
        if st.session_state.auto_update_enabled:
            st.sidebar.info(f"🕐 Next update in: {format_time_remaining()}")
        else:
            st.sidebar.warning("⏸️ Auto-update disabled")
        
        if st.sidebar.button("🔄 Refresh Now", type="secondary"):
            load_csv_automatically.clear()
            st.session_state.last_update_time = get_current_time()
            st.rerun()
        
        st.sidebar.markdown(f"**Last Data Updated:** {st.session_state.last_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        st.sidebar.markdown("---")
        
        st.sidebar.markdown("### 📄 Data Source")
        st.sidebar.info(f"**CSV File:** {CSV_FILE_NAME}")
        
        if os.path.exists(CSV_FILE_PATH):
            file_size = os.path.getsize(CSV_FILE_PATH) / 1024
            file_modified = datetime.fromtimestamp(os.path.getmtime(CSV_FILE_PATH))
            st.sidebar.markdown(f"**Size:** {file_size:.2f} KB")
            st.sidebar.markdown(f"**Modified:** {file_modified.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.sidebar.error("⚠️ CSV file not found!")
        
        st.sidebar.markdown("---")
        
        if st.sidebar.button("🚪 Secure Logout", type="primary"):
            logout()

def process_timestamp_silent(df):
    """Process timestamp column with enhanced format detection"""
    timestamp_cols = [c for c in df.columns if any(keyword in c.lower() 
                     for keyword in ['time', 'date', 'timestamp', 'waktu', 'tanggal'])]
    
    pressure_cols = [c for c in df.columns if any(keyword in c.lower() 
                    for keyword in ['tekanan', 'pressure', 'kondensor', 'condenser'])]
    
    if not pressure_cols:
        return None, None, None
    
    pressure_col = pressure_cols[0]
    successful_format = "auto-detected"
    
    if timestamp_cols:
        timestamp_col = timestamp_cols[0]
        
        # Enhanced date formats for Indonesia data
        date_formats = [
            '%d/%m/%Y %H:%M:%S',
            '%d/%m/%Y %H:%M',
            '%d-%m-%Y %H:%M:%S',
            '%d-%m-%Y %H:%M',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%m/%d/%Y %H:%M:%S',
            '%m/%d/%Y %H:%M',
            '%d/%m/%Y',
            '%d-%m-%Y',
            '%Y-%m-%d',
            '%m/%d/%Y'
        ]
        
        parsed_dates = None
        
        for date_format in date_formats:
            try:
                temp_dates = pd.to_datetime(df[timestamp_col], format=date_format, errors='coerce')
                if not temp_dates.isna().all():
                    parsed_dates = temp_dates
                    successful_format = date_format
                    break
            except Exception:
                continue
        
        if parsed_dates is None or parsed_dates.isna().all():
            try:
                temp_dates = pd.to_datetime(df[timestamp_col], errors='coerce', infer_datetime_format=True)
                if not temp_dates.isna().all():
                    parsed_dates = temp_dates
                    successful_format = "auto-inferred"
                else:
                    parsed_dates = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
                    successful_format = "fallback (2024-01-01 hourly)"
            except Exception:
                parsed_dates = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
                successful_format = "fallback (2024-01-01 hourly)"
        
        df['timestamp'] = parsed_dates
    else:
        df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
    
    return df, pressure_col, successful_format

def main_dashboard():
    """Professional Industrial Dashboard with Auto-Update"""
    
    init_session_state()
    check_and_update()
    
    # Check if required files exist
    MODEL_PATH = "best_lstm_model.h5"
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        st.error("Please ensure the model file is uploaded to your repository.")
        st.stop()
    
    if not os.path.exists(CSV_FILE_PATH):
        st.error(f"❌ CSV file not found: {CSV_FILE_PATH}")
        st.error("Please ensure the CSV file is uploaded to your repository.")
        st.stop()
    
    # Professional industrial styling
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        :root {{
            --main-header-start: #1e3c72;
            --main-header-end: #2a5298;
            --alert-critical-start: #ff7e7e;
            --alert-critical-end: #dc3545;
            --alert-warning-start: #ffea99;
            --alert-warning-end: #ffc107;
            --alert-normal-start: #c3e6cb;
            --alert-normal-end: #28a745;
        }}

        .main {{
            background-color: var(--background-color);
            font-family: 'Inter', sans-serif;
        }}
        
        .main-header {{
            background: linear-gradient(90deg, var(--main-header-start) 0%, var(--main-header-end) 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            color: white;
            text-align: center;
        }}
        
        .alert-critical {{
            background: linear-gradient(90deg, var(--alert-critical-start) 0%, var(--alert-critical-end) 100%);
            border-left: 4px solid var(--alert-critical-end);
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            color: white;
        }}
        
        .alert-warning {{
            background: linear-gradient(90deg, var(--alert-warning-start) 0%, var(--alert-warning-end) 100%);
            border-left: 4px solid var(--alert-warning-end);
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            color: #333;
        }}
        
        .alert-normal {{
            background: linear-gradient(90deg, var(--alert-normal-start) 0%, var(--alert-normal-end) 100%);
            border-left: 4px solid var(--alert-normal-end);
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            color: white;
        }}
        
        .metric-card {{
            background: var(--secondary-background-color);
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .status-operational {{
            color: var(--alert-normal-end);
            font-weight: 600;
        }}
        
        .status-warning {{
            color: var(--alert-warning-end);
            font-weight: 600;
        }}
        
        .status-critical {{
            color: var(--alert-critical-end);
            font-weight: 600;
        }}
        
        .stPlotlyChart {{
            background: var(--secondary-background-color);
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.2rem;">🏭 Industrial Gas Removal Monitoring System</h1>
        <p style="margin: 10px 0 0 0; font-size: 1.1rem; opacity: 0.9;">
            Predictive Maintenance & Real-time Process Monitoring
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    show_user_panel()
    
    st.sidebar.markdown("## ⚙️ System Configuration")
    
    # Model loading with error handling
    MODEL_PATH = "best_lstm_model.h5"
    model = None
    
    if not TENSORFLOW_AVAILABLE:
        st.sidebar.error("❌ TensorFlow not available")
        st.error("⚠️ TensorFlow compatibility issue detected. Dashboard will run in limited mode.")
        st.info("📝 You can still view the data analysis features, but predictions are disabled.")
        # Continue without model for data visualization
    elif not os.path.exists(MODEL_PATH):
        st.sidebar.error(f"❌ Model file not found: {MODEL_PATH}")
        st.error("Please ensure the model file is uploaded to your repository.")
    else:
        try:
            model = load_model(MODEL_PATH, compile=False)
            st.sidebar.success("✅ LSTM Model Loaded")
        except Exception as e:
            st.sidebar.error(f"❌ Model Loading Failed: {str(e)}")
            st.error(f"Model loading error: {str(e)}")
            model = None
    
    st.sidebar.markdown("### 🎛️ System Parameters")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        default_start_date = get_current_time().date() - timedelta(days=7)
        start_date = st.date_input(
            "Start Date", 
            value=default_start_date,
            help="Data collection start date"
        )
    
    with col2:
        data_frequency = st.selectbox(
            "Data Frequency",
            options=["Hourly", "Daily", "Weekly", "Monthly"],
            index=0,
            help="Data sampling frequency"
        )
    
    threshold = st.sidebar.slider(
        "Critical Threshold", 
        min_value=0.05, 
        max_value=0.30, 
        value=0.14, 
        step=0.01,
        help="Critical pressure threshold for maintenance alerts"
    )
    
    sequence_length = st.sidebar.slider(
        "Prediction Sequence Length", 
        min_value=20, 
        max_value=120, 
        value=80, 
        step=10,
        help="Number of historical points used for prediction"
    )
    
    show_detailed_table = st.sidebar.checkbox("Show Detailed Data Table", value=False)
    
    # Load CSV using the cached function
    df = load_csv_automatically()
    
    if df is not None:
        with st.spinner("🔄 Processing sensor data..."):
            # Process timestamp silently
            df, pressure_col, successful_format = process_timestamp_silent(df)
            
            if df is None or pressure_col is None:
                st.error("❌ Pressure column not found. Please ensure your data contains pressure measurements.")
                st.stop()
            
            # Clean and prepare data
            data = df[[pressure_col]].copy()
            data = data.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce'))
            data = data.dropna()
            
            if (data < 0).any().any():
                data = data.clip(lower=0)
            
            ground_truth_all = data.values.flatten()
            timestamps_all = df['timestamp'].iloc[:len(ground_truth_all)].tolist()
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data_all = scaler.fit_transform(data)
            
            train_size = int(len(scaled_data_all) * 0.8)
            
            train_data = scaled_data_all[:train_size]
            test_data = scaled_data_all[train_size:]
            
            def create_sequences(data, seq_length):
                sequences = []
                targets = []
                for i in range(len(data) - seq_length):
                    sequences.append(data[i:i+seq_length])
                    targets.append(data[i+seq_length])
                return np.array(sequences), np.array(targets)

            # Model predictions (only if model is available)
            if model is not None:
                X_train, y_train = create_sequences(train_data, sequence_length)
                X_test, y_test = create_sequences(test_data, sequence_length)

                if len(X_test) > 0:
                    predictions_on_test = model.predict(X_test)
                    predictions_on_test_inv = scaler.inverse_transform(predictions_on_test).flatten()
                    actual_test_inv = scaler.inverse_transform(y_test).flatten()
                    
                    test_timestamps = timestamps_all[train_size + sequence_length:]
                    
                    if len(actual_test_inv) != len(predictions_on_test_inv):
                        min_len_test = min(len(actual_test_inv), len(predictions_on_test_inv))
                        actual_test_inv = actual_test_inv[:min_len_test]
                        predictions_on_test_inv = predictions_on_test_inv[:min_len_test]
                        test_timestamps = test_timestamps[:min_len_test]

                    mse = mean_squared_error(actual_test_inv, predictions_on_test_inv)
                    mae = mean_absolute_error(actual_test_inv, predictions_on_test_inv)
                    r2 = r2_score(actual_test_inv, predictions_on_test_inv)
                    
                    accuracy_tolerance = 0.01
                    accuracy = np.mean(np.abs(actual_test_inv - predictions_on_test_inv) <= accuracy_tolerance) * 100
                    
                else:
                    mse, mae, r2, accuracy = 0, 0, 0, 0
                    predictions_on_test_inv = np.array([])
                    actual_test_inv = np.array([])
                    test_timestamps = []
            else:
                # No model available - set default values
                mse, mae, r2, accuracy = 0, 0, 0, 0
                predictions_on_test_inv = np.array([])
                actual_test_inv = np.array([])
                test_timestamps = []
                st.warning("⚠️ Running in data visualization mode only. Predictions disabled.")
            
            # System status analysis
            current_pressure = ground_truth_all[-1] if len(ground_truth_all) > 0 else 0
            predicted_pressure_now = predictions_on_test_inv[-1] if len(predictions_on_test_inv) > 0 else current_pressure
            
            system_status = "OPERATIONAL"
            status_color = "status-operational"
            alert_class = "alert-normal"
            
            if model is None:
                system_status = "LIMITED_MODE"
                status_color = "status-warning"
                alert_class = "alert-warning"
            elif current_pressure > threshold or predicted_pressure_now > threshold:
                system_status = "CRITICAL"
                status_color = "status-critical"
                alert_class = "alert-critical"
            elif current_pressure > threshold * 0.8 or predicted_pressure_now > threshold * 0.8:
                system_status = "WARNING"
                status_color = "status-warning"
                alert_class = "alert-warning"

            # System status alert
            if system_status == "LIMITED_MODE":
                st.markdown(f"""
                <div class="{alert_class}">
                    <h3>⚠️ LIMITED MODE</h3>
                    <p><strong>Running in data visualization mode only.</strong><br>
                    TensorFlow/Model not available. Predictions disabled.<br>
                    Current pressure: {current_pressure:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
            elif system_status == "CRITICAL":
                st.markdown(f"""
                <div class="{alert_class}">
                    <h3>🚨 CRITICAL ALERT</h3>
                    <p><strong>Immediate maintenance required!</strong><br>
                    Current pressure: {current_pressure:.4f}<br>
                    Predicted pressure: {predicted_pressure_now:.4f}<br>
                    Threshold: {threshold:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
            elif system_status == "WARNING":
                st.markdown(f"""
                <div class="{alert_class}">
                    <h3>⚠️ WARNING</h3>
                    <p><strong>System approaching critical levels.</strong><br>
                    Schedule maintenance within 24 hours.<br>
                    Current pressure: {current_pressure:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="{alert_class}">
                    <h3>✅ SYSTEM OPERATIONAL</h3>
                    <p>All systems operating within normal parameters.<br>
                    Current pressure: {current_pressure:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # KPI Dashboard
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>System Status</h4>
                    <p class="{status_color}">{system_status}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.metric(
                    "Current Pressure", 
                    f"{current_pressure:.4f}",
                    delta=f"{(current_pressure - threshold):.4f}" if current_pressure > threshold else "0.0000"
                )
            
            with col3:
                st.metric(
                    "Model Accuracy (R²)", 
                    f"{r2*100:.1f}%",
                    delta=f"{(r2-0.8)*100:.1f}%" if r2 > 0.8 else None
                )
            
            with col4:
                st.metric(
                    "Prediction Error (MAE)", 
                    f"{mae:.4f}",
                    delta=f"{(mae-0.01):.4f}" if mae > 0.01 else None
                )
            
            with col5:
                maintenance_hours = 24 if system_status == "WARNING" else (0 if system_status == "CRITICAL" else 168)
                st.metric(
                    "Maintenance Window", 
                    f"{maintenance_hours}h",
                    delta="URGENT" if maintenance_hours == 0 else None
                )
            
            # Visualization
            st.markdown("### 📈 Process Monitoring" + (" (Data Visualization Only)" if model is None else ""))
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps_all, 
                    y=ground_truth_all,
                    mode='lines',
                    name='Historical Data',
                    line=dict(color='#2E86AB', width=2),
                    hovertemplate='<b>Time:</b> %{x}<br><b>Pressure:</b> %{y:.4f}<extra></extra>'
                )
            )
            
            # Only add predictions if model is available
            if model is not None and len(predictions_on_test_inv) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=test_timestamps, 
                        y=predictions_on_test_inv,
                        mode='lines',
                        name='Predictions',
                        line=dict(color='#A23B72', width=2, dash='dash'),
                        hovertemplate='<b>Time:</b> %{x}<br><b>Predicted:</b> %{y:.4f}<extra></extra>'
                    )
                )
            
            fig.add_hline(
                y=threshold,
                line_dash="dot",
                line_color="red",
                annotation_text=f"Critical Threshold ({threshold})",
                annotation_position="top right"
            )
            
            chart_title = "Industrial Gas Removal System - Process Monitoring"
            if model is None:
                chart_title += " (Limited Mode)"
            
            fig.update_layout(
                height=600,
                showlegend=True,
                title=chart_title,
                title_x=0.5,
                title_font_size=20,
                template="plotly_white",
                xaxis_title="Time",
                yaxis_title="Pressure",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance metrics
            st.markdown("### 📊 Model Performance" + (" (Not Available - Limited Mode)" if model is None else ""))
            
            col1, col2 = st.columns(2)
            
            with col1:
                if model is not None:
                    metrics_df = pd.DataFrame({
                        'Metric': ['Mean Squared Error', 'Mean Absolute Error', 'R² Score', f'Accuracy (±{accuracy_tolerance})'],
                        'Value': [f"{mse:.6f}", f"{mae:.6f}", f"{r2:.4f}", f"{accuracy:.2f}%"],
                        'Status': ['Good' if mse < 0.001 else 'Poor',
                                  'Good' if mae < 0.01 else 'Poor',
                                  'Excellent' if r2 > 0.9 else 'Good',
                                  'Excellent' if accuracy > 90 else 'Good'] 
                    })
                    st.dataframe(metrics_df, use_container_width=True)
                else:
                    st.warning("⚠️ Model performance metrics not available in limited mode.")
                    st.info("📊 Data statistics available below.")
            
            with col2:
                st.markdown("#### 🔍 Data Processing Info")
                if model is not None:
                    st.info(f"""
                    **Dataset Info:**
                    - Total records: {len(ground_truth_all):,}
                    - Training data: {train_size:,} records ({train_size/len(ground_truth_all)*100:.1f}%)
                    - Test data: {len(ground_truth_all)-train_size:,} records ({(len(ground_truth_all)-train_size)/len(ground_truth_all)*100:.1f}%)
                    - Pressure column: {pressure_col}
                    - Timestamp format: {successful_format}
                    """)
                else:
                    st.info(f"""
                    **Dataset Info (Limited Mode):**
                    - Total records: {len(ground_truth_all):,}
                    - Pressure column: {pressure_col}
                    - Timestamp format: {successful_format}
                    - Min pressure: {ground_truth_all.min():.4f}
                    - Max pressure: {ground_truth_all.max():.4f}
                    - Mean pressure: {ground_truth_all.mean():.4f}
                    """)
            
            # Detailed data table
            if show_detailed_table:
                st.markdown("### 📋 Detailed Data Analysis")
                
                detailed_df = pd.DataFrame({
                    'Timestamp': timestamps_all,
                    'Actual_Pressure': ground_truth_all
                })
                
                if len(predictions_on_test_inv) > 0:
                    detailed_df['Predicted_Pressure'] = np.nan
                    start_idx = len(detailed_df) - len(predictions_on_test_inv)
                    detailed_df.iloc[start_idx:, detailed_df.columns.get_loc('Predicted_Pressure')] = predictions_on_test_inv
                
                detailed_df['Status'] = detailed_df['Actual_Pressure'].apply(
                    lambda x: 'CRITICAL' if x > threshold else 'WARNING' if x > threshold * 0.8 else 'NORMAL'
                )
                
                st.dataframe(detailed_df, use_container_width=True)
                
                # Download functionality
                if 'export' in st.session_state.get('user_info', {}).get('permissions', []):
                    csv_data = detailed_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Analysis Results",
                        data=csv_data,
                        file_name=f"gas_removal_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    
    else:
        st.error("❌ Failed to load CSV data from repository. Please check if the file exists and is properly formatted.")

def main():
    """Main application flow with authentication"""
    if not check_authentication():
        login_page()
    else:
        main_dashboard()

if __name__ == "__main__":
    main()