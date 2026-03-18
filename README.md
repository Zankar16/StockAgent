# 📈 TrendPredictor: AI-Powered Crypto Insights

TrendPredictor is a professional-grade cryptocurrency dashboard built with Streamlit. It leverages multiple machine learning models (LSTM, CNN-1D, Random Forest, XGBoost, ARIMA) to analyze and predict market trends.

## ✨ Features

- **Multi-Model Engine**: Automatically selects the best-performing model based on RMSE.
- **Deep Learning**: Utilizes LSTM and CNN-1D for sequential data analysis.
- **Real-time Data**: Fetches latest market data via `yfinance`.
- **Premium UI**: Glassmorphism aesthetic with a responsive dark theme.
- **Secure Access**: Protected by an authentication layer.

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/Zankar16/trend-predictor.git
cd trend-predictor
pip install -r requirements.txt
```

### 2. Configure Secrets
Create a `.streamlit/secrets.toml` file:
```toml
[credentials]
username = "admin"
password = "your_secure_password"
```

### 3. Run Locally
```bash
streamlit run dashboard.py
```

## 🌐 Deployment

### Streamlit Cloud (Recommended)
1. Push your code to GitHub.
2. Connect your repository to [Streamlit Cloud](https://share.streamlit.io/).
3. Add your `credentials` to the **Secrets** section in the Streamlit Cloud dashboard.

### Heroku
The included `Procfile` is pre-configured for Heroku:
```bash
heroku create your-app-name
git push heroku main
```

## 🛠️ Tech Stack
- **Frontend**: Streamlit, Plotly
- **Models**: TensorFlow/Keras, Scikit-Learn, XGBoost, Statsmodels
- **Data**: YFinance, Pandas, NumPy
