import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

# Page Configuration
st.set_page_config(
    page_title="TrendPredictor | AI-Powered Crypto Insights",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# FETCH CRYPTO DATA
# -----------------------------
def load_data(symbol, days=365):
    df = yf.download(f"{symbol}-USD", period=f"{days}d", interval="1d")
    df = df.dropna()
    return df["Close"].values.reshape(-1, 1)


# -----------------------------
# CREATE WINDOWED SEQUENCES
# -----------------------------
def make_sequences(series, window=30):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    return np.array(X), np.array(y)


# -----------------------------
# LSTM MODEL
# -----------------------------
def train_lstm(X_train, y_train, X_val, y_val):
    model = Sequential([
        LSTM(64, activation="tanh", input_shape=(X_train.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[es],
        verbose=0
    )
    val_pred = model.predict(X_val, verbose=0)
    rmse = sqrt(mean_squared_error(y_val, val_pred))
    return model, rmse


# -----------------------------
# CNN1D MODEL
# -----------------------------
def train_cnn(X_train, y_train, X_val, y_val):
    model = Sequential([
        Conv1D(32, kernel_size=3, activation="relu", input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[es],
        verbose=0
    )
    val_pred = model.predict(X_val, verbose=0)
    rmse = sqrt(mean_squared_error(y_val, val_pred))
    return model, rmse


# -----------------------------
# STACK: RANDOM FOREST + XGBOOST
# -----------------------------
def create_tabular(series, lags=10):
    df = pd.DataFrame({"y": series.flatten()})
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df = df.dropna()
    X = df.drop("y", axis=1).values
    y = df["y"].values
    return X, y


def train_stack(X_train, y_train, X_val, y_val):
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_val)

    xgb_model = xgb.XGBRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_val)

    # Simple blend
    ensemble_pred = (rf_pred + xgb_pred) / 2
    rmse = sqrt(mean_squared_error(y_val, ensemble_pred))

    return (rf, xgb_model), rmse


# -----------------------------
# ARIMA MODEL
# -----------------------------
def train_arima(series):
    train_size = int(len(series) * 0.8)
    train, val = series[:train_size].flatten(), series[train_size:].flatten()
    model = ARIMA(train, order=(5,1,0)).fit()
    forecast = model.forecast(steps=len(val))
    rmse = sqrt(mean_squared_error(val, forecast))
    return model, rmse
def show_coin_detail(symbol, name):
    st.markdown(f"<h2 style='color:{NEON};'>{name} — Detailed Page</h2>", unsafe_allow_html=True)
    
    df = yf.Ticker(symbol).history(period="1y")

    # ---- SAFETY CHECK ----
    if df.empty:
        st.error(f"No data found for {name} ({symbol}). This may be a Yahoo Finance issue.")
        return

    current = df["Close"].iloc[-1]
    high_52 = df["High"].max()
    low_52 = df["Low"].min()
    volume = df["Volume"].iloc[-1]
    open_p = df["Open"].iloc[-1]
    prev = df["Close"].iloc[-2]
    change = current - prev
    pct = (change / prev) * 100

    # --- TOP METRICS ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"${current:,.2f}", f"{pct:.2f}%")
    c2.metric("Today Open", f"${open_p:,.2f}")
    c3.metric("52W High", f"${high_52:,.2f}")
    c4.metric("52W Low", f"${low_52:,.2f}")

    st.markdown("---")

    # --- PRICE CHART ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        line=dict(color=NEON, width=3),
        name=name
    ))
    fig.update_layout(
        height=350,
        paper_bgcolor=CARD_COLOR,
        plot_bgcolor=CARD_COLOR,
        xaxis_title="Date",
        yaxis_title="Price (USD)",
    )

    st.subheader("Price Chart (1 Year)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- SNAPSHOT ---
    st.subheader("Market Snapshot")
    left, right = st.columns(2)

    with left:
        st.write("**Last Close:**", f"${prev:,.2f}")
        st.write("**Current Price:**", f"${current:,.2f}")
        st.write("**Change:**", f"${change:,.2f} ({pct:.2f}%)")

    with right:
        st.write("**Volume:**", f"{volume:,}")
        st.write("**52-Week High:**", f"${high_52:,.2f}")
        st.write("**52-Week Low:**", f"${low_52:,.2f}")

    st.markdown("---")

    # --- MINI PREDICTOR USING YOUR AI ---
    with st.expander("AI Trend Prediction"):
        st.write("AI is calculating trend using your multi-model engine…")
        best_model, last_price, pred = ai_predict(symbol.replace("-USD", ""))

        trend = "Uptrend" if pred > last_price else "Downtrend"
        st.subheader(f"Predicted Trend: {trend}")
        st.write(f"Forecast Price: **${pred:.2f}** (by {best_model})")


# -----------------------------
# MAIN PREDICTION FUNCTION
# -----------------------------
def ai_predict(symbol="BTC", days=365):
    series = load_data(symbol, days)

    # LSTM / CNN dataset
    X, y = make_sequences(series, window=30)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Train deep models
    lstm_model, rmse_lstm = train_lstm(X_train, y_train, X_val, y_val)
    cnn_model, rmse_cnn = train_cnn(X_train, y_train, X_val, y_val)

    # RF + XGBoost Stack
    X_tab, y_tab = create_tabular(series, lags=10)
    X_train_tab, X_val_tab, y_train_tab, y_val_tab = train_test_split(
        X_tab, y_tab, test_size=0.2, shuffle=False
    )
    (rf, xgb_model), rmse_stack = train_stack(
        X_train_tab, y_train_tab, X_val_tab, y_val_tab
    )

    # ARIMA
    arima_model, rmse_arima = train_arima(series)

    # Determine best model
    rmse_scores = {
        "LSTM": rmse_lstm,
        "CNN1D": rmse_cnn,
        "Stack": rmse_stack,
        "ARIMA": rmse_arima,
    }

    best_model = min(rmse_scores, key=rmse_scores.get)

    # Predict next value
    last_price = float(series[-1][0])

    if best_model == "LSTM":
        seq = series[-30:].reshape(1, 30, 1)
        pred = float(lstm_model.predict(seq, verbose=0)[0][0])

    elif best_model == "CNN1D":
        seq = series[-30:].reshape(1, 30, 1)
        pred = float(cnn_model.predict(seq, verbose=0)[0][0])

    elif best_model == "Stack":
        row = np.array([series[-i][0] for i in range(1, 11)]).reshape(1, -1)
        pred = (rf.predict(row)[0] + xgb_model.predict(row)[0]) / 2

    else:
        pred = float(arima_model.forecast(steps=1)[0])

    return best_model, last_price, pred

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

# Colors and Styling
BG_COLOR = "#0D0D0D"
ACCENT_COLOR = "hsl(43, 80%, 47%)"  # Golden/Neon Yellow
CARD_COLOR = "rgba(30, 30, 30, 0.6)"
TEXT_COLOR = "#E0E0E0"
NEON = "#ebab34"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {BG_COLOR};
    color: {TEXT_COLOR};
}}

.stApp {{
    background: radial-gradient(circle at 50% 50%, #1a1a1a 0%, #0d0d0d 100%);
}}

/* Glassmorphism Cards */
.card, .metric-card, .about-card, .big-card {{
    background: {CARD_COLOR};
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}}

.card:hover, .metric-card:hover {{
    transform: translateY(-5px);
    box-shadow: 0 12px 48px 0 rgba(0, 0, 0, 0.5);
    border-color: {NEON};
}}

/* Metrics */
.metric-card h4 {{
    font-size: 0.9rem;
    color: rgba(255,255,255,0.6);
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

.metric-card h2 {{
    font-size: 1.8rem;
    font-weight: 700;
    margin: 5px 0;
}}

.neon-button {{
    background: linear-gradient(135deg, {NEON} 0%, #d49a2a 100%);
    border: none;
    padding: 12px 30px;
    font-weight: 600;
    border-radius: 12px;
    color: black;
    cursor: pointer;
    box-shadow: 0 4px 15px rgba(235, 171, 52, 0.3);
    transition: 0.3s;
}}

.neon-button:hover {{
    transform: scale(1.05);
    box-shadow: 0 6px 25px rgba(235, 171, 52, 0.5);
}}

/* Sidebar Styling */
[data-testid="stSidebar"] {{
    background-color: rgba(13, 13, 13, 0.95);
    border-right: 1px solid rgba(255,255,255,0.05);
}}

h1, h2, h3 {{
    font-weight: 700;
    background: linear-gradient(to right, #ffffff, {NEON});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

</style>
""", unsafe_allow_html=True)
if not st.session_state.logged_in:
    # Login form centered with glassmorphism
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div class='card' style='text-align:center; margin-top: 50px;'>
            <h1 style='font-size:3rem;'>Login</h1>
            <p style='color:rgba(255,255,255,0.5);'>Welcome back to TrendPredictor</p>
        </div>
        """, unsafe_allow_html=True)
        
        user = st.text_input("Username", placeholder="Enter your username")
        pwd = st.text_input("Password", type="password", placeholder="Enter your password")
        
        st.markdown("<br>", unsafe_allow_html=True)
        login_button = st.button("Access Dashboard", use_container_width=True)

        if login_button:
            # Using st.secrets for production-ready credentials
            # Fallback to hardcoded for local if secrets are missing
            target_user = st.secrets.get("credentials", {}).get("username", "admin")
            target_pwd = st.secrets.get("credentials", {}).get("password", "1234")
            
            if user == target_user and pwd == target_pwd:
                st.session_state.logged_in = True
                st.session_state.username = user
                st.success("Authentication Successful! Redirecting...")
                st.balloons()
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")

    st.stop()
with st.sidebar:

    selected = option_menu(
        menu_title="Crypto Dashboard",
        options=["Dashboard","Coins", "Settings", "About"],
        icons=["grid-fill", "pie-chart-fill", "gear-fill", "info-circle-fill"],
        menu_icon="coin",
        default_index=0,
        styles={
            "container": {"background-color": BG_COLOR},
            "icon": {"color": NEON},
            "nav-link": {"color": TEXT_COLOR, "font-size": "16px"},
            "nav-link-selected": {"background-color": NEON, "color": "black"},
        }
    )


if selected == "Dashboard":
    st.markdown(f"<h1 style='color:{NEON}; font-weight:700;'>Dashboard</h1>", unsafe_allow_html=True)


    # -----------------------------------------
    # 1. TOP STAT CARDS FOR ALL CRYPTOCURRENCIES
    # -----------------------------------------

    crypto_list = {
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
        "Tether": "USDT-USD",
        "BNB": "BNB-USD",
        "Solana": "SOL-USD",
        "XRP": "XRP-USD",
        "Dogecoin": "DOGE-USD",
        "Cardano": "ADA-USD",
        "Avalanche": "AVAX-USD",
        "TRON": "TRX-USD"
    }

    st.markdown("### Market Summary")
    
    # 5 cards per row → 2 rows for 10 coins
    rows = [list(crypto_list.items())[:5], list(crypto_list.items())[5:]]

    for row in rows:
        cols = st.columns(5)

        for (name, symbol), col in zip(row, cols):
            df = yf.Ticker(symbol).history(period="1mo")

            if df.empty or len(df) < 2:
                with col:
                    st.error(f"No data for {name}")
                continue

            current = df["Close"].iloc[-1]
            prev = df["Close"].iloc[-2]
            change = current - prev
            pct = (change / prev) * 100 if prev != 0 else 0
            vol = df["Volume"].iloc[-1]
            color = "#4CAF50" if pct >= 0 else "#FF5252"
            with col:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <h4>{name}</h4>
                        <h2 style='color:{NEON};'>${current:,.2f}</h2>
                        <p style='color:{color};'>Change: {pct:.2f}%</p>
                        <p style='font-size:13px;'>Volume: {vol:,}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown("---")
    
    btc_df = yf.Ticker("BTC-USD").history(period="1mo")
    # -----------------------------------------
    # 2. PORTFOLIO OVERVIEW (DONUT CHART + DETAILS)
    # -----------------------------------------
    left, right = st.columns([1.2, 2])

    import plotly.express as px

    donut_fig = px.pie(
        values=[37, 21, 18, 24],
        names=["Bitcoin", "Ethereum", "Litecoin", "Others"],
        hole=0.65,
        color_discrete_sequence=[NEON, "#f2e5dc", "#E57373", "#9c9919"]
    )
    donut_fig.update_layout(
        paper_bgcolor=CARD_COLOR,
        plot_bgcolor=CARD_COLOR,
        height=350,
        showlegend=True,
        annotations=[dict(text="$8,121", font_size=22, showarrow=False)]
    )

    with left:
        st.markdown("### Portfolio Overview")
        st.plotly_chart(donut_fig, use_container_width=True)

    # -----------------------------------------
    # 3. PORTFOLIO ANALYTICS LINE CHART
    # -----------------------------------------
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=btc_df.index,
        y=btc_df["Close"],
        mode="lines",
        line=dict(color=NEON, width=2.5)
    ))

    fig_line.update_layout(
        paper_bgcolor=CARD_COLOR,
        plot_bgcolor=CARD_COLOR,
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Date",
        yaxis_title="Price (USD)"
    )

    with right:
        st.markdown("### Portfolio Analytics")
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("---")

    # -----------------------------------------
    # 5. CANDLESTICK CHART (Premium UI)
    # -----------------------------------------
    st.markdown("### Candlestick Chart")

    fig_candle = go.Figure(data=[
        go.Candlestick(
            x=btc_df.index,
            open=btc_df["Open"],
            high=btc_df["High"],
            low=btc_df["Low"],
            close=btc_df["Close"],
            increasing_line_color="#55a64e",
            decreasing_line_color="#a64e4e",
        )
    ])

    fig_candle.update_layout(
        paper_bgcolor=CARD_COLOR,
        plot_bgcolor=CARD_COLOR,
        height=400,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Date",
        yaxis_title="Price"
    )

    st.plotly_chart(fig_candle, use_container_width=True)




elif selected == "Coins":
    st.markdown(f"<h1 style='color:{NEON};'>Crypto Details</h1>", unsafe_allow_html=True)

    coin = st.selectbox("Select a cryptocurrency", 
                        ["Bitcoin (BTC)", "Ethereum (ETH)", "BNB", "Solana (SOL)","XRP","Dogecoin (DOGE)","Cardano (ADA)","Avalanche (AVAX)","TRON (TRX)"])

    symbol_map = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Tether (USDT)": "USDT-USD",
    "BNB": "BNB-USD",
    "Solana (SOL)": "SOL-USD",
    "XRP": "XRP-USD",
    "Dogecoin (DOGE)": "DOGE-USD",
    "Cardano (ADA)": "ADA-USD",
    "Avalanche (AVAX)": "AVAX-USD",
    "TRON (TRX)": "TRX-USD"
}

    symbol = symbol_map[coin]
    show_coin_detail(symbol, coin)
# ============================================================================
# SETTINGS
# ============================================================================
elif selected == "Settings":
    st.markdown(f"<h1 style='color:{NEON};'>Settings</h1>", unsafe_allow_html=True)

    # Show logged in user
    st.markdown(f"Logged in as: **{st.session_state.username}**")

    # Editable Name
    new_name = st.text_input("User Name", value=st.session_state.username)

    auto = st.toggle("Auto Sync", value=True)
    notify = st.toggle("Enable Notifications", value=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<button class='neon-button'>Save Settings</button>", unsafe_allow_html=True)
    save_click = st.button("Apply Settings")

    if save_click:
        st.session_state.username = new_name
        st.success("Settings updated!")

    st.markdown("---")

    # LOGOUT BUTTON
    st.markdown("<button class='neon-button'>Logout</button>", unsafe_allow_html=True)
    logout_click = st.button("Logout Now")

    if logout_click:
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.success("Logged out successfully!")
        st.rerun()




# ============================================================================
# ABOUT
# ============================================================================
elif selected == "About":
    st.markdown(f"<h1 style='color:{NEON};'>About</h1>", unsafe_allow_html=True)

    st.markdown(
    f"""
    <div class='about-card'>
        <h2 style='color:{NEON};'>Crypto Dashboard v2.0 — User Guide</h2>

        <p>
            Welcome to the <b>AI-Powered Crypto Dashboard</b>.<br>
            This guide will help you understand every feature and how to use the system effectively.
        </p>

        <hr style="border:1px solid {NEON};">
        <h3 style='color:{NEON};'>1. Dashboard Overview</h3>
        <p>The dashboard provides real-time cryptocurrency insights including:</p>
        <ul style="margin-left:20px;">
            <li>Live prices for the top 10 cryptocurrencies</li>
            <li>Total portfolio balance</li>
            <li>Your crypto allocation with progress bars</li>
            <li>Interactive BTC chart (6-month price history)</li>
        </ul>
        <h3 style='color:{NEON};'>2. AI Trend Predictor</h3>
        <p>The AI predictor analyzes cryptocurrency trends using multiple machine-learning models. 
           The system automatically selects the <b>Best Model</b> based on RMSE performance.</p>

        <p>Prediction includes:</p>
        <ul style="margin-left:20px;">
            <li>Trend Direction — Uptrend or Downtrend</li>
            <li>Next-day forecasted price</li>
            <li>Confidence score (%)</li>
        </ul>

        <p>Models used:</p>
        <ul style="margin-left:20px;">
            <li><b>LSTM Neural Network</b></li>
            <li><b>CNN-1D Deep Learning</b></li>
            <li><b>Random Forest + XGBoost Stack</b></li>
            <li><b>ARIMA Time-Series Model</b></li>
        </ul>
        <h3 style='color:{NEON};'>3. Settings</h3>
        <p>Customize your experience:</p>
        <ul style="margin-left:20px;">
            <li>Update your username</li>
            <li>Enable or disable notifications</li>
            <li>Toggle auto-sync features</li>
            <li>Logout securely</li>
        </ul>
        <h3 style='color:{NEON};'>4. Login System</h3>
        <p>
            A secure login ensures that your personalized dashboard remains private.<br>
            Only authorized users can access predictions, history, and sensitive insights.
        </p>
        <h3 style='color:{NEON};'>5. Notes</h3>
        <ul style="margin-left:20px;">
            <li>Predictions are for educational purposes only.</li>
            <li>Crypto markets are highly volatile — always trade responsibly.</li>
            <li>Ensure a stable internet connection for real-time updates.</li>
        </ul>

        <hr style="border:1px solid {NEON};">

        <p style="text-align:center; font-size:18px;">
            Designed by <b>GROUP 18 (MEMBERS: ***)</b><br>
            Powered by AI • Streamlit • Python
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
