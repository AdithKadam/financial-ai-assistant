import os
import sys

# Add 'src' folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import json

# Add src directory to path
sys.path.append('src')

# Import our custom modules
try:
    from src.database.database import FinanceDB
    from src.utils.data_generator import FinancialDataGenerator
    from src.utils.api_client import APIClient
    from src.utils.rag_system import FinancialRAGSystem
    from src.agents.financial_agents import FinancialAgentCoordinator
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all modules are in the correct directories")
    st.stop()

# --- NEW THEME/COLOR/FONT CHANGE START ---

# Configure Streamlit page
st.set_page_config(
    page_title="Financial Freedom Awaits", # New Page Title
    page_icon="💸", # More vibrant icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state (No changes here)
if 'user_id' not in st.session_state:
    st.session_state.user_id = 1
if 'db' not in st.session_state:
    st.session_state.db = FinanceDB()
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = FinancialRAGSystem()
    st.session_state.rag_system.initialize_knowledge_base()
if 'agent_coordinator' not in st.session_state:
    st.session_state.agent_coordinator = FinancialAgentCoordinator()
if 'api_client' not in st.session_state:
    st.session_state.api_client = APIClient()

# Custom CSS for a vibrant, elegant, and formal look (Inspired by Owncap)
st.markdown("""
<style>
    /* Import elegant Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Montserrat:wght@400;500;600;700&display=swap');

    :root {
        /* Owncap inspired palette */
        --primary-accent: #00C896; /* Vibrant Teal */
        --dark-primary: #0A343D;   /* Dark Teal/Blue (for backgrounds, text) */
        --light-background: #F8F9FA; /* Off-white for main content */
        --white: #FFFFFF;
        --text-color: #333333; /* Darker text for readability */
        --subtle-gray: #ADB5BD; /* Light gray for borders/hints */
    }

    /* Apply custom fonts to the body and headings */
    body {
        font-family: 'Poppins', sans-serif;
        color: var(--text-color);
        background-color: var(--white);
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Montserrat', sans-serif;
        color: var(--dark-primary);
        font-weight: 600;
    }

    /* Main header styling (Financial Freedom Awaits) */
    .main-header {
        font-size: 4.5rem; /* Larger, more impactful */
        font-weight: 700;
        color: var(--dark-primary);
        text-align: left;
        margin-top: 3rem;
        margin-bottom: 1rem;
        line-height: 1.1;
        letter-spacing: -0.03em;
    }
    
    .stApp {
        background-color: var(--white);
    }
    .stApp [data-testid="stHeader"] {
        background-color: transparent;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Custom Header for Owncap-like branding */
    .owncap-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 2rem;
        background-color: var(--dark-primary); /* Dark teal header background */
        color: var(--white);
        margin: -1rem -1rem 2rem -1rem; /* Adjust to full width */
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .owncap-header .logo {
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        font-size: 1.8rem;
        color: var(--primary-accent); /* Vibrant Teal logo */
        display: flex;
        align-items: center;
    }
    .owncap-nav a {
        color: var(--white);
        text-decoration: none;
        margin-left: 2rem;
        font-weight: 500;
        font-family: 'Poppins', sans-serif;
        transition: color 0.3s;
    }
    .owncap-nav a:hover {
        color: var(--primary-accent);
    }
    .contact-button {
        background-color: var(--primary-accent);
        color: var(--dark-primary);
        padding: 0.7rem 1.5rem;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
        transition: background-color 0.3s;
    }

    /* Metric Cards - Sleek and modern */
    .metric-card {
        background-color: var(--white);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid var(--subtle-gray);
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin: 0.7rem 0;
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-3px);
    }
    
    /* FIX: Allow metric values to fit */
    [data-testid="stMetricValue"] {
        font-family: 'Montserrat', sans-serif;
        color: var(--dark-primary) !important;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        /* Ensure numbers don't get truncated */
        overflow: visible !important; 
        word-break: break-word;
        white-space: normal;
        line-height: 1.1;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Poppins', sans-serif;
        color: var(--text-color) !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        /* Ensure label fits */
        white-space: normal; 
    }
    
    /* Chat Messages - Clean and distinct */
    .chat-message {
        padding: 1.2rem;
        border-radius: 15px;
        margin: 0.7rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        font-family: 'Poppins', sans-serif;
    }
    .user-message {
        background-color: #E6F7F2; /* Light vibrant teal for user */
        border-left: 5px solid var(--primary-accent);
        margin-left: 3rem;
    }
    .ai-message {
        background-color: var(--light-background); /* Off-white for AI */
        border-right: 5px solid var(--subtle-gray);
        margin-right: 3rem;
    }

    /* Sidebar - Cleaner and integrated look */
    .css-1d391kg { /* Targets the sidebar main container */
        background-color: var(--dark-primary) !important;
        color: var(--white) !important;
        font-family: 'Poppins', sans-serif;
    }
    .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4 {
        color: var(--primary-accent) !important;
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
    }
    .css-1d391kg .st-bh, .css-1d391kg .st-ci { /* Selectbox and input labels in sidebar */
        color: var(--white) !important;
    }
    .sidebar-section {
        background-color: rgba(255,255,255,0.08); /* Slightly lighter dark teal */
        border-radius: 10px;
        margin: 0rem 0;
        border: 0px solid rgba(255,255,255,0.1);
    }

    /* Buttons - Sleek, primary accent */
    .stButton > button {
        background-color: var(--primary-accent);
        color: var(--dark-primary);
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s, transform 0.2s;
        box-shadow: 0 2px 5px rgba(0,200,150,0.3);
    }
    .stButton > button:hover {
        background-color: #00A67C;
        transform: translateY(-2px);
    }

    /* Expander/Info boxes */
    .stAlert div[data-baseweb="button"] {
        background-color: #E6F7F2;
        color: var(--dark-primary);
        border-left: 5px solid var(--primary-accent);
        border-radius: 8px;
        padding: 1rem;
    }
    .stExpander div[data-baseweb="accordion-item"] {
        background-color: var(--light-background);
        border-radius: 10px;
        border: 1px solid var(--subtle-gray);
        margin-bottom: 0.5rem;
        padding: 0.5rem;
    }
    
    p, li, .st-ay, .st-az { /* General text and table cells */
        font-family: 'Poppins', sans-serif;
        color: var(--text-color);
    }
</style>
""", unsafe_allow_html=True)

# Define primary colors for Plotly consistency
PRIMARY_COLOR = '#00C896' # Vibrant Teal
SECONDARY_COLOR = '#0A343D' # Dark Teal/Blue
ACCENT_COLOR = '#88D1AA' # Lighter Teal for subtle contrasts

# --- NEW THEME/COLOR/FONT CHANGE END ---

def load_sample_data():
    """Load or generate sample data"""
    try:
        # Check if sample data exists
        if os.path.exists('data/synthetic/sample_transactions.csv'):
            transactions_df = pd.read_csv('data/synthetic/sample_transactions.csv')
            users_df = pd.read_csv('data/synthetic/sample_users.csv')
            return users_df, transactions_df
        else:
            # Generate sample data
            with st.spinner("Generating sample data..."):
                generator = FinancialDataGenerator()
                users_df, transactions_df = generator.save_sample_data()
            return users_df, transactions_df
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return pd.DataFrame(), pd.DataFrame()

def create_spending_overview_chart(transactions_df):
    """Create spending overview visualizations"""
    if transactions_df.empty:
        return None, None, None
    
    # Filter expenses only
    expenses_df = transactions_df[transactions_df['transaction_type'] == 'expense'].copy()
    expenses_df['date'] = pd.to_datetime(expenses_df['date'])
    
    # Monthly spending trend
    monthly_spending = expenses_df.groupby(expenses_df['date'].dt.to_period('M'))['amount'].sum().reset_index()
    monthly_spending['date'] = monthly_spending['date'].astype(str)
    
    trend_fig = px.line(
        monthly_spending, 
        x='date', 
        y='amount',
        title='Monthly Spending Trend',
        markers=True,
        color_discrete_sequence=[PRIMARY_COLOR] # Use Vibrant Teal
    )
    trend_fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Amount ($)",
        template="plotly_white",
        font_family="Poppins, sans-serif", # Apply new font
        title_font_family="Montserrat, sans-serif",
        title_font_color=SECONDARY_COLOR
    )
    
    # Category breakdown
    category_spending = expenses_df.groupby('category')['amount'].sum().reset_index()
    category_spending = category_spending.sort_values('amount', ascending=False)
    
    # Pie chart
    pie_fig = px.pie(
        category_spending,
        values='amount',
        names='category',
        title='Spending by Category',
        color_discrete_sequence=px.colors.qualitative.Pastel # Softer, elegant palette
    )
    pie_fig.update_traces(textposition='inside', textinfo='percent+label')
    pie_fig.update_layout(
        font_family="Poppins, sans-serif",
        title_font_family="Montserrat, sans-serif",
        title_font_color=SECONDARY_COLOR
    )
    
    # Bar chart
    bar_fig = px.bar(
        category_spending.head(10),
        x='amount',
        y='category',
        orientation='h',
        title='Top 10 Spending Categories',
        text='amount',
        color_discrete_sequence=[SECONDARY_COLOR] # Use Dark Teal/Blue
    )
    bar_fig.update_traces(texttemplate='$%{text:.0f}', textposition='outside')
    bar_fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        font_family="Poppins, sans-serif",
        title_font_family="Montserrat, sans-serif",
        title_font_color=SECONDARY_COLOR
    )
    
    return trend_fig, pie_fig, bar_fig

def create_budget_comparison_chart(budget_data):
    """Create budget vs actual spending comparison"""
    if not budget_data:
        return None
    
    categories = list(budget_data.keys())
    actual_amounts = [data['current_spending'] for data in budget_data.values()]
    budget_amounts = [data['recommended_limit'] for data in budget_data.values()]
    
    fig = go.Figure()
    
    # Add budget limits (Lighter accent color)
    fig.add_trace(go.Bar(
        name='Budget Limit',
        x=categories,
        y=budget_amounts,
        marker_color=ACCENT_COLOR, 
        opacity=0.7
    ))
    
    # Add actual spending (Primary vibrant color)
    fig.add_trace(go.Bar(
        name='Actual Spending',
        x=categories,
        y=actual_amounts,
        marker_color=PRIMARY_COLOR # Vibrant Teal
    ))
    
    fig.update_layout(
        title='Budget vs Actual Spending',
        xaxis_title='Category',
        yaxis_title='Amount ($)',
        barmode='group',
        template='plotly_white',
        font_family="Poppins, sans-serif",
        title_font_family="Montserrat, sans-serif",
        title_font_color=SECONDARY_COLOR
    )
    
    return fig

def create_financial_health_gauge(health_score):
    """Create financial health gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = health_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Financial Health Score", 'font': {'family': 'Montserrat', 'color': SECONDARY_COLOR}},
        delta = {'reference': 70},
        gauge = {
            'axis': {'range': [None, 100], 'tickfont': {'family': 'Poppins'}},
            'bar': {'color': PRIMARY_COLOR}, # Vibrant Teal bar
            'steps': [
                {'range': [0, 50], 'color': "#DDF3EB"}, # Lightest teal
                {'range': [50, 80], 'color': "#BEE7DA"} # Mid teal
            ],
            'threshold': {
                'line': {'color': '#FF6347', 'width': 4}, # Tomato red for warning
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, font_family="Poppins, sans-serif")
    return fig

def create_spending_velocity_chart(transactions_df):
    """Create spending velocity visualization"""
    if transactions_df.empty:
        return None
    
    expenses_df = transactions_df[transactions_df['transaction_type'] == 'expense'].copy()
    expenses_df['date'] = pd.to_datetime(expenses_df['date'])
    expenses_df = expenses_df.sort_values('date')
    
    # Calculate daily spending
    daily_spending = expenses_df.groupby(expenses_df['date'].dt.date)['amount'].sum().reset_index()
    daily_spending.columns = ['date', 'amount']
    
    # Calculate 7-day moving average
    daily_spending['moving_avg'] = daily_spending['amount'].rolling(window=7, min_periods=1).mean()
    
    fig = go.Figure()
    
    # Daily spending markers
    fig.add_trace(go.Scatter(
        x=daily_spending['date'],
        y=daily_spending['amount'],
        mode='markers',
        name='Daily Spending',
        marker=dict(color=PRIMARY_COLOR, opacity=0.6) # Vibrant Teal markers
    ))
    
    # Moving average line
    fig.add_trace(go.Scatter(
        x=daily_spending['date'],
        y=daily_spending['moving_avg'],
        mode='lines',
        line=dict(color=SECONDARY_COLOR, width=2), # Dark Teal/Blue line
        name='7-Day Avg'
    ))
    
    fig.update_layout(
        title='Daily Spending Pattern',
        xaxis_title='Date',
        yaxis_title='Amount ($)',
        template='plotly_white',
        font_family="Poppins, sans-serif",
        title_font_family="Montserrat, sans-serif",
        title_font_color=SECONDARY_COLOR
    )
    
    return fig

def display_ai_insights(insights_data):
    """Display AI-generated insights"""
    if 'summary' in insights_data:
        summary = insights_data['summary']

        # Financial Health Score
        col1, col2, col3 = st.columns([1, 3, 1 ])
        with col2:
            health_score = summary.get('overall_health_score', 70)
            gauge_fig = create_financial_health_gauge(health_score)
            st.plotly_chart(gauge_fig, use_container_width=True)
        
         # Top Priorities
        if 'top_priorities' in summary and summary['top_priorities']:
            st.subheader("Top Priorities")
            for i, priority in enumerate(summary['top_priorities'], 1):
                st.info(f"**{i}.** {priority}")

       
def key_metrics(insights_data):

 if 'summary' in insights_data:
        summary = insights_data['summary']

        if 'key_metrics' in summary:
            st.subheader("Key Metrics")
            metrics = summary['key_metrics']
            
            # FIX: Use equal columns to ensure large numbers fit
            col1, col2, col3, col4 = st.columns(4) 
            
            with col1:
                monthly_spending = metrics.get('monthly_spending', 0)
                st.metric("Monthly Spending", f"${monthly_spending:,.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                monthly_income = st.session_state.get('current_user_income', 5000)
                savings_rate = ((monthly_income - monthly_spending) / monthly_income * 100) if monthly_income > 0 else 0
                st.metric("Savings Rate", f"{savings_rate:,.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                num_transactions = len(st.session_state.get('current_transactions', []))
                st.metric("Transactions", num_transactions)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                avg_transaction = monthly_spending / num_transactions if num_transactions > 0 else 0
                st.metric("Avg Transaction", f"${avg_transaction:,.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
              

def chat_interface():
    """AI Chat Interface"""
    st.subheader("Ask Your AI Financial Advisor")
    
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat input
    user_input = st.chat_input("Ask me about your finances...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get AI response
        with st.spinner("Thinking..."):
            try:
                # Prepare user context
                user_context = {
                    'monthly_income': st.session_state.get('current_user_income', 5000),
                    'transactions': st.session_state.get('current_transactions', pd.DataFrame()).to_dict(orient="records")
                }

                # Get response from RAG system
                chat_response = st.session_state.rag_system.chat_with_context(user_input, user_context)
                ai_response = chat_response['response']
                
                # Add contextual advice if available
                if chat_response.get('contextual_advice', {}).get('recommendations'):
                    ai_response += "\n\n**Additional Recommendations:**\n"
                    for rec in chat_response['contextual_advice']['recommendations'][:3]:
                        ai_response += f"• {rec}\n"
                
                # Add AI response to history
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                
            except Exception as e:
                error_response = f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."
                st.session_state.chat_history.append({"role": "assistant", "content": error_response})
    
    # Display chat history (using custom CSS classes via st.markdown)
    for message in st.session_state.chat_history[-10:]:  # Show last 10 messages
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message ai-message">{message["content"]}</div>', unsafe_allow_html=True)

def file_upload_section():
    """Handle file upload and processing"""
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file with your transaction data",
        type=['csv'],
        help="Upload a CSV file with columns: date, amount, category, description, merchant, transaction_type"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            transactions_df = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_columns = ['date', 'amount', 'category']
            missing_columns = [col for col in required_columns if col not in transactions_df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                return None
            
            # Add default values for missing optional columns
            if 'description' not in transactions_df.columns:
                transactions_df['description'] = ''
            if 'merchant' not in transactions_df.columns:
                transactions_df['merchant'] = ''
            if 'transaction_type' not in transactions_df.columns:
                transactions_df['transaction_type'] = transactions_df['amount'].apply(lambda x: 'income' if x > 0 else 'expense')
            
            # Validate and process data
            transactions_df['date'] = pd.to_datetime(transactions_df['date'])
            transactions_df['amount'] = pd.to_numeric(transactions_df['amount'], errors='coerce').abs() # Use absolute amount for expenses/income
            
            # Remove invalid rows
            transactions_df = transactions_df.dropna(subset=['date', 'amount'])
            
            st.success(f"Successfully loaded {len(transactions_df)} transactions!")
            
            # Show preview
            with st.expander("Preview Data"):
                st.dataframe(transactions_df.head(10))
            
            # Store in session state
            st.session_state.current_transactions = transactions_df
            st.session_state.uploaded_data = True
            
            return transactions_df
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return None
    
    return None

def main():
    """Main Streamlit application"""
    
    # Custom Header (Mimicking Owncap)
    st.markdown(f"""
        <div class="owncap-header">
            <div class="logo" style="font-size: 3.5em;">
                Financial AI Assistant
            </div>
        </div>
    """, unsafe_allow_html=True)
    # st.markdown("Innovate, Prosper, Own: Empowering Your Financial Future", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Navigation")
        
        # Page selection
        page = st.selectbox(
            "Choose a view:",
            ["Upload Data", "Dashboard Overview", "Spending Analysis", "Budget Planning", "AI Chat Assistant", "Market Context"]
        )
        
        # User settings
        st.markdown('<div class="sidebar-section"><h3>User Settings</h3>', unsafe_allow_html=True)
        # Check if income is already set from session state, otherwise use default
        default_income = st.session_state.get('current_user_income', 5000)
        monthly_income = st.number_input("Monthly Income ($)", min_value=0, value=default_income, step=100)
        st.session_state.current_user_income = monthly_income
        
        financial_goal = st.selectbox(
            "Primary Financial Goal",
            ["Emergency Fund", "House Down Payment", "Retirement", "Debt Payoff", "Vacation"]
        )
        # Store goal in session state
        st.session_state.financial_goal = financial_goal 
        st.markdown('</div>', unsafe_allow_html=True) # Close sidebar section div
        
        # Quick stats
        if 'current_transactions' in st.session_state:
            st.markdown('<div class="sidebar-section"><h3>Quick Stats</h3>', unsafe_allow_html=True)
            transactions = st.session_state.current_transactions
            if not transactions.empty:
                expenses = transactions[transactions['transaction_type'] == 'expense']
                total_spending = expenses['amount'].sum()
                st.metric("Total Spending", f"${total_spending:,.2f}")
                st.metric("Number of Transactions", len(expenses))
            st.markdown('</div>', unsafe_allow_html=True) # Close sidebar section div
    
    # Load data
    if 'current_transactions' not in st.session_state or st.session_state.get('uploaded_data') is None:
        users_df, transactions_df = load_sample_data()
        if not transactions_df.empty:
            # Use first user's data as default
            st.session_state.current_transactions = transactions_df[transactions_df['user_id'] == 1].copy()
            # Set default income if not set by user
            if 'current_user_income' not in st.session_state:
                st.session_state.current_user_income = monthly_income 
    
    # Main content based on selected page
    if page == "Dashboard Overview":
        dashboard_overview()
    elif page == "Spending Analysis":
        spending_analysis()
    elif page == "Budget Planning":
        budget_planning()
    elif page == "AI Chat Assistant":
        chat_interface()
    elif page == "Upload Data":
        upload_data_page()
    elif page == "Market Context":
        market_context_page()

def dashboard_overview():
    """Dashboard overview page"""
    st.header("Financial Dashboard Overview")
    
    # Get current transactions
    transactions_df = st.session_state.get('current_transactions', pd.DataFrame())
    
    if transactions_df.empty:
        st.warning("No transaction data available. Please upload your data or use sample data.")
        return
    
    # Run comprehensive analysis
    with st.spinner("Analyzing your financial data..."):
        user_data = {
            'monthly_income': st.session_state.current_user_income,
            'transactions': transactions_df
        }
        
        # Get market data
        market_data = st.session_state.api_client.get_comprehensive_market_context()
        
        # Run analysis
        analysis_results = st.session_state.agent_coordinator.comprehensive_analysis(user_data, market_data)
    
    key_metrics(analysis_results)
    # Display results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Spending charts
        trend_fig, pie_fig, bar_fig = create_spending_overview_chart(transactions_df)
        
        if trend_fig:
            st.plotly_chart(trend_fig, use_container_width=True)
        
        col1a, col1b = st.columns(2)
        with col1a:
            if pie_fig:
                st.plotly_chart(pie_fig, use_container_width=True)
        with col1b:
            if bar_fig:
                st.plotly_chart(bar_fig, use_container_width=True)
    
    with col2:
        # AI Insights
        display_ai_insights(analysis_results) 

    

    
    # Additional analysis sections
    st.markdown("---")
    
    # Budget comparison
    if 'budget_recommendations' in analysis_results:
        st.subheader("Budget vs Actual")
        budget_data = analysis_results['budget_recommendations'].get('current_vs_recommended')
        if budget_data:
            budget_fig = create_budget_comparison_chart(budget_data)
            if budget_fig:
                st.plotly_chart(budget_fig, use_container_width=True)
    
    # Market context
    if 'market_context' in analysis_results:
        st.subheader("Market Context")
        market_context = analysis_results['market_context']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            inflation_impact = market_context.get('inflation_impact', {})
            monthly_impact = inflation_impact.get('monthly_impact', 0)
            st.metric("Inflation Impact", f"+${monthly_impact:,.2f}/month")
        
        with col2:
            interest_context = market_context.get('interest_rate_impact', {})
            rate_environment = interest_context.get('rate_environment', 'moderate')
            st.metric("Interest Environment", rate_environment.title())
        
        with col3:
            employment_context = market_context.get('employment_context', {})
            employment_env = employment_context.get('employment_environment', 'stable')
            st.metric("Job Market", employment_env.title())

def spending_analysis():
    """Detailed spending analysis page"""
    st.header("Spending Analysis")
    
    transactions_df = st.session_state.get('current_transactions', pd.DataFrame())
    
    if transactions_df.empty:
        st.warning("No transaction data available.")
        return
    
    # Ensure date column is datetime and get min/max
    try:
        transactions_df['date'] = pd.to_datetime(transactions_df['date'])
        min_date = transactions_df['date'].min().date()
        max_date = transactions_df['date'].max().date()
    except Exception:
        st.error("Error processing date column.")
        return

    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=min_date)
    with col2:
        end_date = st.date_input("End Date", value=max_date)
    
    # Filter transactions
    filtered_df = transactions_df[
        (transactions_df['date'].dt.date >= start_date) &
        (transactions_df['date'].dt.date <= end_date)
    ].copy()
    
    # Analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Trends", "Categories", "Anomalies"])
    
    with tab1:
        # Overview metrics
        expenses_df = filtered_df[filtered_df['transaction_type'] == 'expense']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_spent = expenses_df['amount'].sum()
            st.metric("Total Spending", f"${total_spent:,.2f}")
        with col2:
            avg_transaction = expenses_df['amount'].mean() if not expenses_df.empty else 0
            st.metric("Average Transaction", f"${avg_transaction:,.2f}")
        with col3:
            num_transactions = len(expenses_df)
            st.metric("Number of Transactions", num_transactions)
        with col4:
            days_range = (end_date - start_date).days + 1
            daily_avg = total_spent / days_range if days_range > 0 else 0
            st.metric("Daily Average", f"${daily_avg:,.2f}")
        
        # Spending velocity chart
        velocity_fig = create_spending_velocity_chart(filtered_df)
        if velocity_fig:
            st.plotly_chart(velocity_fig, use_container_width=True)
    
    with tab2:
        # Trend analysis
        trend_fig, _, _ = create_spending_overview_chart(filtered_df)
        if trend_fig:
            st.plotly_chart(trend_fig, use_container_width=True)
        
        # Weekly pattern analysis
        if not expenses_df.empty:
            expenses_df['day_of_week'] = expenses_df['date'].dt.day_name()
            weekly_spending = expenses_df.groupby('day_of_week')['amount'].sum().reindex([
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
            ])
            
            weekly_fig = px.bar(
                x=weekly_spending.index,
                y=weekly_spending.values,
                title="Spending by Day of Week",
                labels={'x': 'Day', 'y': 'Amount ($)'},
                color_discrete_sequence=[SECONDARY_COLOR]
            )
            weekly_fig.update_layout(template='plotly_white', font_family="Poppins, sans-serif", title_font_family="Montserrat, sans-serif")
            st.plotly_chart(weekly_fig, use_container_width=True)
    
    with tab3:
        # Category analysis
        if not expenses_df.empty:
            category_stats = expenses_df.groupby('category').agg({
                'amount': ['sum', 'count', 'mean'],
                'date': ['min', 'max']
            }).round(2)
            
            category_stats.columns = ['Total', 'Count', 'Average', 'First Transaction', 'Last Transaction']
            category_stats = category_stats.reset_index()
            category_stats['Percentage'] = (category_stats['Total'] / category_stats['Total'].sum() * 100).round(2)
            
            st.dataframe(category_stats, use_container_width=True)
            
            # Category pie chart
            _, pie_fig, _ = create_spending_overview_chart(filtered_df)
            if pie_fig:
                st.plotly_chart(pie_fig, use_container_width=True)
    
    with tab4:
        # Anomaly detection
        with st.spinner("Detecting spending anomalies..."):
            spending_analyzer = st.session_state.agent_coordinator.spending_analyzer
            analysis = spending_analyzer.analyze_spending_patterns(filtered_df)
            
            anomalies = analysis.get('anomalies', {})
            
            if anomalies.get('anomalies'):
                st.subheader("Detected Anomalies")
                for anomaly in anomalies['anomalies'][:10]:
                    with st.expander(f"${anomaly['amount']:,.2f} - {anomaly['category']} on {anomaly['date']}"):
                        st.write(f"**Type:** {anomaly['type'].replace('_', ' ').title()}")
                        st.write(f"**Description:** {anomaly.get('description', 'N/A')}")
                        if 'deviation' in anomaly:
                            st.write(f"**Deviation:** {anomaly['deviation']:,.2f} standard deviations above average")
                        if 'category_average' in anomaly:
                            st.write(f"**Category Average:** ${anomaly['category_average']:,.2f}")
            else:
                st.info("No significant anomalies detected in your spending patterns.")

def budget_planning():
    """Budget planning and recommendations page"""
    st.header("Budget Planning & Recommendations")
    
    transactions_df = st.session_state.get('current_transactions', pd.DataFrame())
    monthly_income = st.session_state.current_user_income
    
    if transactions_df.empty:
        st.warning("No transaction data available for budget analysis.")
        return
    
    # Run budget analysis
    with st.spinner("Analyzing budget and generating recommendations..."):
        user_data = {
            'monthly_income': monthly_income,
            'age': 30,  # Default
            'financial_goals': st.session_state.get('financial_goal', 'Emergency Fund'),
            'risk_tolerance': 'Moderate',
            'transactions': transactions_df
        }
        
        # Get spending analysis first
        spending_analysis = st.session_state.agent_coordinator.spending_analyzer.analyze_spending_patterns(transactions_df)
        user_data['spending_analysis'] = spending_analysis
        
        # Get budget recommendations
        budget_recommendations = st.session_state.agent_coordinator.budget_advisor.generate_budget_recommendations(user_data)
    
    # Display budget overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Recommended vs Current Budget")
        
        if 'current_vs_recommended' in budget_recommendations:
            budget_comparison = budget_recommendations['current_vs_recommended']
            
            # Create comparison chart
            budget_fig = create_budget_comparison_chart(budget_comparison)
            if budget_fig:
                st.plotly_chart(budget_fig, use_container_width=True)
            
            # Detailed budget table
            budget_df = pd.DataFrame([
                {
                    'Category': category,
                    'Current ($)': data['current_spending'],
                    'Recommended ($)': data['recommended_limit'],
                    'Difference ($)': data['difference'],
                    'Status': '🔴 Over' if data['status'] == 'over' else '🟢 Under',
                    '% of Income': f"{data['percentage_of_income']:,.1f}%"
                }
                for category, data in budget_comparison.items()
            ])
            
            st.dataframe(budget_df, use_container_width=True, height=400)
    
    with col2:
        st.subheader("Budget Style")
        budget_style = budget_recommendations.get('budget_style', 'moderate')
        st.info(f"**Recommended Style:** {budget_style.title()}")
        
        # Budget style explanation
        style_explanations = {
            'conservative': "Prioritizes savings and emergency funds with lower discretionary spending.",
            'moderate': "Balanced approach between saving and enjoying life.",
            'aggressive': "Higher discretionary spending with moderate savings rate."
        }
        st.write(style_explanations.get(budget_style, "Balanced budget approach."))
    
    # Recommendations section
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Recommendations")
        recommendations = budget_recommendations.get('recommendations', [])
        
        for rec in recommendations:
            priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}
            priority_icon = priority_color.get(rec['priority'], "🟢")
            
            with st.expander(f"{priority_icon} {rec['category']} - {rec['priority'].title()} Priority"):
                st.write(rec['message'])
                if 'suggested_actions' in rec:
                    st.write("**Suggested Actions:**")
                    for action in rec['suggested_actions']:
                        st.write(f"• {action}")
    
    with col2:
        st.subheader("Savings Opportunities")
        savings_opportunities = budget_recommendations.get('savings_opportunities', [])
        
        total_potential_savings = sum(opp['potential_monthly_savings'] for opp in savings_opportunities)
        st.metric("Total Potential Monthly Savings", f"${total_potential_savings:,.2f}")
        
        for opp in savings_opportunities:
            difficulty_color = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}
            difficulty_icon = difficulty_color.get(opp['difficulty'], "🟡")
            
            st.write(f"{difficulty_icon} **{opp['category']}**: ${opp['potential_monthly_savings']:,.2f}/month")
            st.write(f"   Annual potential: ${opp['potential_annual_savings']:,.2f}")
            st.write(f"   Difficulty: {opp['difficulty'].title()}")
            st.write("---")

def upload_data_page():
    """Data upload page"""
    st.header("Upload Your Financial Data")
    
    # File upload section
    uploaded_transactions = file_upload_section()
    
    if uploaded_transactions is not None:
        st.success("Data uploaded successfully!")
        
        # Show analysis of uploaded data
        st.subheader("Quick Analysis")
        
        expenses = uploaded_transactions[uploaded_transactions['transaction_type'] == 'expense']
        income = uploaded_transactions[uploaded_transactions['transaction_type'] == 'income']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Expenses", f"${expenses['amount'].sum():,.2f}")
        with col2:
            st.metric("Total Income", f"${income['amount'].sum():,.2f}")
        with col3:
            st.metric("Net Amount", f"${income['amount'].sum() - expenses['amount'].sum():,.2f}")
        with col4:
            st.metric("Transactions", len(uploaded_transactions))
        
        # Quick visualization
        if not expenses.empty:
            category_spending = expenses.groupby('category')['amount'].sum().sort_values(ascending=False)
            
            fig = px.bar(
                x=category_spending.values,
                y=category_spending.index,
                orientation='h',
                title='Spending by Category',
                labels={'x': 'Amount ($)', 'y': 'Category'},
                color_discrete_sequence=[SECONDARY_COLOR]
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, template='plotly_white', font_family="Poppins, sans-serif", title_font_family="Montserrat, sans-serif")
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Show sample data format
        st.subheader("Required Data Format")
        st.write("Your CSV file should have the following columns:")
        
        sample_data = pd.DataFrame([
            {
                'date': '2024-09-01',
                'amount': 1200.00,
                'category': 'Housing',
                'description': 'Rent payment',
                'merchant': 'Property Manager',
                'transaction_type': 'expense'
            },
            {
                'date': '2024-09-01',
                'amount': 5000.00,
                'category': 'Salary',
                'description': 'Monthly salary',
                'merchant': 'Company',
                'transaction_type': 'income'
            },
            {
                'date': '2024-09-02',
                'amount': 45.50,
                'category': 'Food & Dining',
                'description': 'Grocery shopping',
                'merchant': 'Supermarket',
                'transaction_type': 'expense'
            }
        ])
        
        st.dataframe(sample_data)
        
        st.write("**Required columns:**")
        st.write("• `date` - Transaction date (YYYY-MM-DD format)")
        st.write("• `amount` - Transaction amount (positive number, for expense or income)")
        st.write("• `category` - Spending category")
        
        st.write("**Optional columns:**")
        st.write("• `description` - Transaction description")
        st.write("• `merchant` - Merchant/payee name")
        st.write("• `transaction_type` - 'income' or 'expense' (important for distinguishing cash flow)")

def market_context_page():
    """Market context and economic indicators page"""
    st.header("Market Context & Economic Indicators")
    
    # Get market data
    with st.spinner("Fetching latest market data..."):
        market_data = st.session_state.api_client.get_comprehensive_market_context()
    
    # Economic indicators
    st.subheader("Economic Indicators")
    
    economic_indicators = market_data.get('economic_indicators', {})
    
    if economic_indicators:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            inflation_data = economic_indicators.get('CPI_All_Items', {})
            if inflation_data:
                st.metric("Inflation Rate", f"{inflation_data.get('current_value', 'N/A')}")
        
        with col2:
            unemployment_data = economic_indicators.get('Unemployment_Rate', {})
            if unemployment_data:
                st.metric("Unemployment Rate", f"{unemployment_data.get('current_value', 'N/A')}%")
        
        with col3:
            fed_rate_data = economic_indicators.get('Federal_Funds_Rate', {})
            if fed_rate_data:
                st.metric("Federal Funds Rate", f"{fed_rate_data.get('current_value', 'N/A')}%")
        
        with col4:
            sentiment_data = economic_indicators.get('Consumer_Sentiment', {})
            if sentiment_data:
                st.metric("Consumer Sentiment", f"{sentiment_data.get('current_value', 'N/A')}")
    
    # Market sentiment
    st.subheader("Market Sentiment")
    
    market_sentiment = market_data.get('market_sentiment', {})
    
    if market_sentiment:
        sentiment_desc = market_sentiment.get('sentiment_description', 'Neutral')
        sentiment_score = market_sentiment.get('overall_sentiment', 0)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Sentiment gauge
            delta_color = 'inverse' if sentiment_score < 0 else 'normal'
            st.metric("Overall Sentiment", sentiment_desc, delta=f"{sentiment_score:,.3f}", delta_color=delta_color)
        
        with col2:
            # Recent headlines
            headlines = market_sentiment.get('recent_headlines', [])
            if headlines:
                st.write("**Recent Headlines:**")
                for headline in headlines[:3]:
                    st.write(f"• {headline}")
    
    # Context summary
    context_summary = market_data.get('context_summary', 'No context available')
    st.info(f"**Market Summary:** {context_summary}")
    
    # Personal impact analysis
    if 'current_transactions' in st.session_state:
        st.subheader("Impact on Your Finances")
        
        user_data = {
            'monthly_income': st.session_state.current_user_income,
            'transactions': st.session_state.current_transactions
        }
        
        # Run market impact analysis
        market_impact = st.session_state.agent_coordinator.market_context.analyze_market_impact(user_data, market_data)
        
        # Inflation impact
        if 'inflation_impact' in market_impact:
            inflation_impact = market_impact['inflation_impact']
            
            st.write("**Inflation Impact on Your Spending:**")
            col1, col2 = st.columns(2)
            
            with col1:
                monthly_impact = inflation_impact.get('monthly_impact', 0)
                annual_impact = inflation_impact.get('total_estimated_annual_increase', 0)
                st.metric("Additional Monthly Cost", f"${monthly_impact:,.2f}")
                st.metric("Additional Annual Cost", f"${annual_impact:,.2f}")
            
            with col2:
                percentage_impact = inflation_impact.get('percentage_of_income', 0)
                st.metric("% of Income Impact", f"{percentage_impact:,.1f}%")
                
                # Category impacts
                category_impacts = inflation_impact.get('category_impacts', {})
                if category_impacts:
                    st.write("**Most Affected Categories:**")
                    sorted_impacts = sorted(category_impacts.items(), 
                                            key=lambda x: x[1]['monthly_increase'], reverse=True)
                    for category, impact in sorted_impacts[:3]:
                        st.write(f"• {category}: +${impact['monthly_increase']:,.2f}/month")
        
        # Interest rate impact
        if 'interest_rate_impact' in market_impact:
            interest_impact = market_impact['interest_rate_impact']
            
            st.write("**Interest Rate Environment:**")
            rate_env = interest_impact.get('rate_environment', 'moderate')
            st.write(f"Current environment: **{rate_env.title()}**")
            
            impacts = interest_impact.get('impacts', {})
            if impacts:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    cc_impact = impacts.get('credit_cards', {})
                    st.write("**Credit Cards:**")
                    st.write(f"Est. Rate: {cc_impact.get('estimated_rate', 'N/A')}%")
                
                with col2:
                    savings_impact = impacts.get('savings', {})
                    st.write("**Savings:**")
                    st.write(f"Est. Rate: {savings_impact.get('estimated_rate', 'N/A')}%")
                
                with col3:
                    mortgage_impact = impacts.get('mortgage', {})
                    st.write("**Mortgage:**")
                    st.write(f"Est. Rate: {mortgage_impact.get('estimated_rate', 'N/A')}%")
        
        # Recommendations
        recommendations = market_impact.get('spending_recommendations', [])
        if recommendations:
            st.subheader("Market-Based Recommendations")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
    
    # Historical data if available
    st.subheader("Economic Trends")
    
    # Try to get historical economic data
    try:
        economic_data = st.session_state.api_client.get_economic_indicators(
            start_date='2024-01-01',
            end_date=datetime.now().strftime('%Y-%m-%d')
        )
        
        if not economic_data.empty:
            # Create trend charts for each indicator
            indicators = economic_data['indicator_name'].unique()
            
            # Using subplots to present charts formally
            num_indicators = min(len(indicators), 4)
            if num_indicators > 0:
                rows = (num_indicators + 1) // 2
                cols = 2 if num_indicators > 1 else 1
                
                fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f'{ind.replace("_", " ")} Trend' for ind in indicators[:num_indicators]])
                
                for i, indicator in enumerate(indicators[:num_indicators]):
                    row = i // cols + 1
                    col = i % cols + 1
                    indicator_data = economic_data[economic_data['indicator_name'] == indicator]
                    
                    if len(indicator_data) > 1:
                        fig.add_trace(go.Scatter(
                            x=indicator_data['date'],
                            y=indicator_data['value'],
                            mode='lines+markers',
                            name=indicator,
                            line=dict(color=PRIMARY_COLOR if i % 2 == 0 else SECONDARY_COLOR)
                        ), row=row, col=col)
                
                fig.update_layout(
                    height=400 * rows, 
                    title_text="Key Economic Indicators Trends", 
                    template='plotly_white', 
                    showlegend=False,
                    font_family="Poppins, sans-serif",
                    title_font_family="Montserrat, sans-serif"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.write(f"Historical data not available: {str(e)}")
        
        # Show sample economic data visualization
        import random
        dates = pd.date_range('2024-01-01', periods=12, freq='M')
        sample_data = pd.DataFrame({
            'date': dates,
            'inflation': [3.2 + random.uniform(-0.5, 0.8) for _ in range(12)],
            'unemployment': [3.8 + random.uniform(-0.3, 0.7) for _ in range(12)],
            'fed_rate': [4.5 + random.uniform(-0.25, 0.5) for _ in range(12)]
        })
        
        fig = px.line(
            sample_data.melt(id_vars=['date'], var_name='Indicator', value_name='Rate (%)'),
            x='date',
            y='Rate (%)',
            color='Indicator',
            title='Economic Indicators Trend (Sample Data)',
            labels={'value': 'Rate (%)', 'variable': 'Indicator'},
            color_discrete_sequence=[PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR]
        )
        fig.update_layout(template='plotly_white', font_family="Poppins, sans-serif", title_font_family="Montserrat, sans-serif")
        st.plotly_chart(fig, use_container_width=True)

# Run the app
if __name__ == "__main__":
    main()