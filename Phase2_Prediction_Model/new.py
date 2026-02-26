import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import joblib
import os


# =====================================================
# 1️⃣ Load and Preprocess Data
# =====================================================
def load_and_preprocess_data(file_path):
    """
    Loads data from CSV, sets datetime index, selects OHLCV columns, and drops NA.
    """
    data = pd.read_csv(file_path)
    data["open_time"] = pd.to_datetime(data["open_time"])
    data.set_index("open_time", inplace=True)
    data = data[["open", "high", "low", "close", "volume"]]
    data.dropna(inplace=True)
    return data


# =====================================================
# 2️⃣ Add Technical Indicators
# =====================================================
def add_technical_indicators(data):
    """
    Adds SMA, RSI, Bollinger Upper, and ATR indicators to data.
    """
    data["SMA"] = data["close"].rolling(window=14).mean()

    delta = data["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    data["Bollinger_Upper"] = (
        data["close"].rolling(window=20).mean()
        + 2 * data["close"].rolling(window=20).std()
    )

    high_low = data["high"] - data["low"]
    high_close = np.abs(data["high"] - data["close"].shift())
    low_close = np.abs(data["low"] - data["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data["ATR"] = tr.rolling(window=14).mean()

    data.dropna(inplace=True)
    return data


# =====================================================
# 3️⃣ Create Sequences with horizon_steps
# =====================================================
def create_sequences(data, seq_length, horizon_steps=1):
    """
    Creates sequences for LSTM/GRU input.
    horizon_steps determines how many steps ahead to predict.
    """
    X, y = [], []
    for i in range(seq_length, len(data) - horizon_steps + 1):
        X.append(data[i - seq_length : i])
        y.append(data[i + horizon_steps - 1, 0])  # target is 'close' column
    return np.array(X), np.array(y)


# =====================================================
# 4️⃣ Build LSTM Model
# =====================================================
def build_lstm_model(input_shape):
    """
    Builds an LSTM model with two layers and tanh activations.
    """
    lstm_model = Sequential(
        [
            LSTM(64, activation="tanh", return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, activation="tanh", return_sequences=False),
            Dropout(0.2),
            Dense(32, activation="tanh"),
            Dense(1),  # Linear activation for regression
        ]
    )
    lstm_model.compile(optimizer="adam", loss="mse")
    return lstm_model


# =====================================================
# 5️⃣ Build GRU Model
# =====================================================
def build_gru_model(input_shape):
    """
    Builds a GRU model with two layers and tanh activations.
    """
    gru_model = Sequential(
        [
            GRU(64, activation="tanh", return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            GRU(64, activation="tanh", return_sequences=False),
            Dropout(0.2),
            Dense(32, activation="tanh"),
            Dense(1),
        ]
    )
    gru_model.compile(optimizer="adam", loss="mse")
    return gru_model


# =====================================================
# 6️⃣ Train Model
# =====================================================
def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=100,
    batch_size=64,
    model_save_path="best_model.keras",
):
    """
    Trains model with early stopping, LR reduction, and model checkpoint.
    """
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
    )
    model_checkpoint = ModelCheckpoint(
        model_save_path, monitor="val_loss", save_best_only=True, verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1,
    )
    return model, history


# =====================================================
# 7️⃣ Evaluate Model
# =====================================================
def evaluate_model(model, X_test, y_test, scaler):
    """
    Evaluates model and returns metrics.
    """
    predictions = model.predict(X_test)
    num_features = scaler.n_features_in_

    dummy_preds = np.zeros((predictions.shape[0], num_features))
    dummy_preds[:, 0] = predictions.reshape(-1)
    predictions_rescaled = scaler.inverse_transform(dummy_preds)[:, 0]

    dummy_y_test = np.zeros((y_test.shape[0], num_features))
    dummy_y_test[:, 0] = y_test.reshape(-1)
    y_test_rescaled = scaler.inverse_transform(dummy_y_test)[:, 0]

    mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
    r2 = r2_score(y_test_rescaled, predictions_rescaled)
    next_predicted_price = (
        predictions_rescaled[-1] if len(predictions_rescaled) > 0 else np.nan
    )

    return (
        y_test_rescaled,
        predictions_rescaled,
        mse,
        rmse,
        mae,
        r2,
        next_predicted_price,
    )


# =====================================================
# 8️⃣ Save Model
# =====================================================
def save_trained_model(model, model_type, timeframe):
    """
    Saves trained model to file.
    """
    model_filename = f"model_{model_type}_{timeframe}.keras"
    model.save(model_filename)
    print(f"Model for {model_type} at {timeframe} saved as {model_filename}")


# =====================================================
# 9️⃣ Plot Results
# =====================================================
def plot_results(y_test_rescaled, predictions_rescaled, model_type, timeframe, dates):
    """
    Plots predicted vs actual prices.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_test_rescaled, label="Actual Price")
    plt.plot(dates, predictions_rescaled, linestyle="--", label="Predicted Price")
    plt.title(f"{model_type} - {timeframe} Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{model_type}_{timeframe}_prediction.png")
    plt.show()


# =====================================================
# 10️⃣ Plot Metrics Comparison
# =====================================================
def plot_metrics_comparison(results):
    """
    Plots bar chart of metrics comparison across models and timeframes.
    """
    timeframes = list(results.keys())
    if not timeframes:
        print("No results to plot for metrics comparison.")
        return

    metrics = ["MSE", "RMSE", "MAE", "R²"]
    model_types = list(results[timeframes[0]].keys())
    metric_values = {
        metric: {model_type: [] for model_type in model_types} for metric in metrics
    }

    for tf in timeframes:
        for model_type in model_types:
            if model_type in results[tf]:
                for metric in metrics:
                    metric_values[metric][model_type].append(
                        results[tf][model_type].get(metric, np.nan)
                    )
            else:
                for metric in metrics:
                    metric_values[metric][model_type].append(np.nan)

    num_models = len(model_types)
    bar_width = 0.8 / num_models
    plt.figure(figsize=(15, 12))
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        x_pos_base = np.arange(len(timeframes))
        for j, model_type in enumerate(model_types):
            offset = (j - (num_models - 1) / 2) * bar_width
            bars = plt.bar(
                x_pos_base + offset,
                metric_values[metric][model_type],
                bar_width,
                label=model_type,
            )
            for bar in bars:
                yval = bar.get_height()
                if not np.isnan(yval):
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        yval,
                        f"{yval:.2f}",
                        ha="center",
                        va="bottom",
                    )
        plt.title(f"{metric} Comparison")
        plt.xlabel("Timeframe")
        plt.ylabel(metric)
        plt.xticks(x_pos_base, timeframes)
        plt.legend()
        plt.grid(axis="y", linestyle="--")
    plt.tight_layout()
    plt.savefig("metrics_comparison.png")
    plt.show()


# =====================================================
# 11️⃣ Plot Combined Predictions
# =====================================================
def plot_combined_predictions(models_predictions, actual_prices, dates):
    """
    Plots predictions from multiple models on the same timeline with actual prices.
    """
    plt.figure(figsize=(14, 7))
    for label, preds in models_predictions.items():
        plt.plot(
            dates[: len(preds)], preds, linestyle="--", label=f"Predicted ({label})"
        )
    plt.plot(
        dates[: len(actual_prices)],
        actual_prices,
        color="black",
        linewidth=2,
        label="Actual Price",
    )
    plt.title("Combined Model Predictions Comparison")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("combined_predictions_comparison.png")
    plt.show()


# =====================================================
# 12️⃣ Main Function
# =====================================================
def main():
    """
    Main function to train and evaluate models on multiple datasets.
    """
    # Paths to three separate datasets (5min, 15min, 1h)
    dataset_paths = {
        "5min": "/content/drive/MyDrive/binance_5min.csv",
        "15min": "/content/drive/MyDrive/binance_15min.csv",
        "1h": "/content/drive/MyDrive/binance_1h.csv",
    }

    seq_length = 50
    horizon_steps_mapping = {
        "5min": 3,
        "15min": 4,
        "1h": 4,
    }  # Example: steps ahead for each timeframe
    results = {}

    data_dict = {}  # store processed data for combined plot

    for tf_key, file_path in dataset_paths.items():
        print(f"\nProcessing dataset: {tf_key}")
        if not os.path.exists(file_path):
            print(f"Error: file {file_path} not found, skipping {tf_key}")
            continue

        data = load_and_preprocess_data(file_path)
        data = add_technical_indicators(data)
        feature_cols = [
            "close",
            "open",
            "high",
            "low",
            "volume",
            "SMA",
            "RSI",
            "Bollinger_Upper",
            "ATR",
        ]
        data = data[feature_cols].dropna()

        features_np = data.values
        train_split = int(0.8 * len(features_np))
        val_split = train_split + int(0.1 * len(features_np))

        train_features = features_np[:train_split]
        val_features = features_np[train_split:val_split]
        test_features = features_np[val_split:]

        # Scale features
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_features)
        val_scaled = scaler.transform(val_features)
        test_scaled = scaler.transform(test_features)

        # Save scaler
        scaler_path = f"scaler_{tf_key}.pkl"
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved for {tf_key}")

        # Create sequences
        horizon_steps = horizon_steps_mapping.get(tf_key, 1)
        X_train, y_train = create_sequences(train_scaled, seq_length, horizon_steps)
        X_val, y_val = create_sequences(val_scaled, seq_length, horizon_steps)
        X_test, y_test = create_sequences(test_scaled, seq_length, horizon_steps)

        # Dates for plotting
        test_dates_start_index = val_split + seq_length
        dates_for_plot = data.index[
            test_dates_start_index : test_dates_start_index + len(y_test)
        ]

        results[tf_key] = {}

        # --- LSTM ---
        print(f"\n--- Training LSTM for {tf_key} ---")
        lstm_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        lstm_model, _ = train_model(
            lstm_model,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=100,
            model_save_path=f"best_lstm_{tf_key}.keras",
        )
        y_test_r, preds_r, mse, rmse, mae, r2, next_p = evaluate_model(
            lstm_model, X_test, y_test, scaler
        )
        results[tf_key]["LSTM"] = {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R²": r2,
            "Next Predicted Price": next_p,
        }
        save_trained_model(lstm_model, "LSTM", tf_key)
        plot_results(y_test_r, preds_r, "LSTM", tf_key, dates_for_plot)

        data_dict[tf_key] = {
            "dates": dates_for_plot,
            "y_test": y_test_r,
            "preds": preds_r,
        }

        # --- GRU ---
        print(f"\n--- Training GRU for {tf_key} ---")
        gru_model = build_gru_model((X_train.shape[1], X_train.shape[2]))
        gru_model, _ = train_model(
            gru_model,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=100,
            model_save_path=f"best_gru_{tf_key}.keras",
        )
        y_test_r, preds_r, mse, rmse, mae, r2, next_p = evaluate_model(
            gru_model, X_test, y_test, scaler
        )
        results[tf_key]["GRU"] = {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R²": r2,
            "Next Predicted Price": next_p,
        }
        save_trained_model(gru_model, "GRU", tf_key)
        plot_results(y_test_r, preds_r, "GRU", tf_key, dates_for_plot)

    # Display metrics table
    print("\n--- Comparison of Evaluation Metrics ---")
    for tf, metrics_dict in results.items():
        print(f"\nTimeframe: {tf}")
        for model_name, metric_vals in metrics_dict.items():
            print(f"  Model: {model_name}")
            for key, val in metric_vals.items():
                print(f"    {key}: {val:.2f}")

    # Plot metrics comparison
    plot_metrics_comparison(results)

    # Combined plot (e.g., LSTM only)
    combined_preds = {tf: data_dict[tf]["preds"] for tf in data_dict}
    combined_actual = data_dict["5min"][
        "y_test"
    ]  # use first dataset's actuals as reference
    combined_dates = data_dict["5min"]["dates"]
    plot_combined_predictions(combined_preds, combined_actual, combined_dates)


if __name__ == "__main__":
    main()
