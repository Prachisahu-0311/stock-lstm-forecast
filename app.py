import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model

# ----------------- CONFIG -----------------
st.set_page_config(page_title="Stock LSTM Forecast", layout="wide")

SEQ_LEN = 30
FEATURE_COLS = [
    "Close", "Return", "SMA_10", "SMA_30",
    "EMA_10", "EMA_30", "Volatility_10", "RSI_14", "Volume"
]

# Load trained Keras LSTM model
MODEL_PATH = "model_lstm.h5"
model = load_model(MODEL_PATH, compile=False)


# ----------------- HELPERS -----------------
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


@st.cache_data
def load_data(ticker: str, start: str):
    df = yf.download(ticker, start=start)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # Returns
    data["Return"] = data["Close"].pct_change()

    # Moving averages
    data["SMA_10"] = data["Close"].rolling(10).mean()
    data["SMA_30"] = data["Close"].rolling(30).mean()
    data["EMA_10"] = data["Close"].ewm(span=10, adjust=False).mean()
    data["EMA_30"] = data["Close"].ewm(span=30, adjust=False).mean()

    # Volatility
    data["Volatility_10"] = data["Return"].rolling(10).std()

    # RSI (14)
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    data["RSI_14"] = 100 - (100 / (1 + rs))

    data = data.dropna()
    return data


def create_sequences(features, target, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i : i + seq_len])
        y.append(target[i + seq_len])
    return np.array(X), np.array(y)


# ----------------- MAIN APP -----------------
st.title("📈 Stock Price Forecasting using LSTM")

st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2015-01-01"))
test_size = st.sidebar.slider("Test size (days)", min_value=100, max_value=400, value=200, step=50)

if st.sidebar.button("Run Forecast"):
    with st.spinner("Downloading data and running forecasts..."):
        # 1. Load data
        df_raw = load_data(ticker, start_date.strftime("%Y-%m-%d"))

        if df_raw.shape[0] <= SEQ_LEN + test_size:
            st.error("Not enough data for selected test size. Choose a smaller test size or earlier start date.")
        else:
            # 2. Feature engineering
            df_feat = add_features(df_raw)

            # 3. Scaling
            df_model = df_feat[FEATURE_COLS].copy()

            feature_scaler = MinMaxScaler()
            target_scaler = MinMaxScaler()

            features_scaled = feature_scaler.fit_transform(df_model.values)
            target_scaled = target_scaler.fit_transform(df_feat[["Close"]].values)

            # 4. Sequences
            X_all, y_all = create_sequences(features_scaled, target_scaled, SEQ_LEN)

            # Align dates for sequences (drop first SEQ_LEN)
            seq_dates = df_feat.index[SEQ_LEN:]

            # Train/Test split on sequences
            X_train, X_test = X_all[:-test_size], X_all[-test_size:]
            y_train, y_test = y_all[:-test_size], y_all[-test_size:]
            test_dates = seq_dates[-test_size:]

            # 5. LSTM prediction
            y_pred_scaled = model.predict(X_test, verbose=0)
            y_test_inv = target_scaler.inverse_transform(y_test)
            y_pred_inv = target_scaler.inverse_transform(y_pred_scaled)

            y_test_inv = y_test_inv.flatten()
            y_pred_inv = y_pred_inv.flatten()

            # 6. Metrics
            rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
            mae = mean_absolute_error(y_test_inv, y_pred_inv)
            mape_val = mape(y_test_inv, y_pred_inv)

            st.subheader("📊 LSTM Performance (Test Set)")
            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{rmse:.2f}")
            col2.metric("MAE", f"{mae:.2f}")
            col3.metric("MAPE", f"{mape_val:.2f}%")

            # 7. Plot actual vs predicted
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(test_dates, y_test_inv, label="Actual", color="black")
            ax.plot(test_dates, y_pred_inv, label="LSTM Forecast", color="orange")
            ax.set_title(f"{ticker} – LSTM Forecast vs Actual (Last {test_size} days)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)

            # 8. Next-day forecast
            last_window = X_all[-1][np.newaxis, ...]
            next_pred_scaled = model.predict(last_window, verbose=0)
            next_price = target_scaler.inverse_transform(next_pred_scaled)[0, 0]

            last_known_date = df_feat.index[-1]
            next_date = last_known_date + pd.Timedelta(days=1)

            st.subheader("🔮 Next-Day Price Prediction")
            st.write(f"Last available date: **{last_known_date.date()}**")
            st.write(f"Predicted closing price for **{next_date.date()}**: **${next_price:.2f}**")

            st.success("Forecast complete ✅")

else:
    st.info("Set your options in the sidebar and click **Run Forecast** to start.")
