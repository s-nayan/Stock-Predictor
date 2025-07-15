from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from keras.models import load_model
import os
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model once when the server starts
# Load the model once when the server starts
# Use MODEL_PATH env var if set, otherwise default to local file
model_path = os.environ.get("MODEL_PATH", "Latest_stock_price_model.keras")
model = load_model(model_path)

@app.route("/api/home", methods=['GET'])
def home():
    return jsonify({"message": "Stock Prediction API is running"})

@app.route("/api/stock-data", methods=['GET'])
def get_stock_data():
    try:
        stock = request.args.get('stock', 'GOOG')
        years = int(request.args.get('years', 10))
        
        # Get stock data
        end = datetime.now()
        start = datetime(end.year - years, end.month, end.day)
        
        stock_data = yf.download(stock, start, end)
        
        if stock_data.empty:
            return jsonify({"error": "Invalid stock symbol or no data available"}), 400
        
        # Handle MultiIndex columns if they exist
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = [col[0] for col in stock_data.columns]
        
        # Calculate moving averages
        stock_data['MA_100'] = stock_data['Close'].rolling(100).mean()
        stock_data['MA_200'] = stock_data['Close'].rolling(200).mean()
        stock_data['MA_250'] = stock_data['Close'].rolling(250).mean()
        
        # Convert to JSON format - handle NaN values and ensure proper conversion
        def safe_to_list(series):
            """Safely convert a pandas Series to list, handling NaN values"""
            return series.fillna(0).values.tolist()
        
        data = {
            'dates': stock_data.index.strftime('%Y-%m-%d').tolist(),
            'close_prices': safe_to_list(stock_data['Close']),
            'ma_100': safe_to_list(stock_data['MA_100']),
            'ma_200': safe_to_list(stock_data['MA_200']),
            'ma_250': safe_to_list(stock_data['MA_250']),
            'volume': safe_to_list(stock_data['Volume']),
            'high': safe_to_list(stock_data['High']),
            'low': safe_to_list(stock_data['Low']),
            'open': safe_to_list(stock_data['Open'])
        }
        
        return jsonify({
            "stock": stock,
            "data": data,
            "info": {
                "total_records": len(stock_data),
                "date_range": f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict", methods=['POST'])
def predict_stock():
    try:
        data = request.get_json()
        stock = data.get('stock', 'GOOG')
        years = data.get('years', 10)
        
        # Get stock data
        end = datetime.now()
        start = datetime(end.year - years, end.month, end.day)
        
        stock_data = yf.download(stock, start, end)
        
        if stock_data.empty:
            return jsonify({"error": "Invalid stock symbol or no data available"}), 400
        
        # Handle MultiIndex columns if they exist
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = [col[0] for col in stock_data.columns]
        
        # Prepare test data (same as in main.py)
        splitting_len = int(len(stock_data) * 0.7)
        x_test = stock_data[['Close']].iloc[splitting_len:].copy()
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(x_test[['Close']])
        
        # Prepare sequences for prediction
        x_data = []
        y_data = []
        
        for i in range(100, len(scaled_data)):
            x_data.append(scaled_data[i-100:i])
            y_data.append(scaled_data[i])
        
        x_data, y_data = np.array(x_data), np.array(y_data)
        
        # Make predictions
        predictions = model.predict(x_data, verbose=0)
        
        # Inverse transform predictions
        inv_predictions = scaler.inverse_transform(predictions)
        inv_y_test = scaler.inverse_transform(y_data)
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((inv_predictions - inv_y_test) ** 2))
        
        # Create results DataFrame
        prediction_dates = stock_data.index[splitting_len + 100:]
        
        result = {
            "stock": stock,
            "predictions": {
                "dates": prediction_dates.strftime('%Y-%m-%d').tolist(),
                "actual_prices": inv_y_test.reshape(-1).tolist(),
                "predicted_prices": inv_predictions.reshape(-1).tolist()
            },
            "metrics": {
                "rmse": float(rmse),
                "accuracy": float(max(0, 100 - (rmse / np.mean(inv_y_test) * 100)))
            },
            "training_data_split": f"{splitting_len}/{len(stock_data)} records used for training"
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/forecast", methods=['POST'])
def forecast_stock():
    try:
        data = request.get_json()
        stock = data.get('stock', 'GOOG')
        days = data.get('days', 30)
        years = data.get('years', 10)
        
        # Get stock data
        end = datetime.now()
        start = datetime(end.year - years, end.month, end.day)
        
        stock_data = yf.download(stock, start, end)
        
        if stock_data.empty:
            return jsonify({"error": "Invalid stock symbol or no data available"}), 400
        
        # Handle MultiIndex columns if they exist
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = [col[0] for col in stock_data.columns]
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(stock_data[['Close']])
        
        # Get last 100 days for forecasting
        last_100_scaled = scaled_data[-100:]
        future_input = last_100_scaled.reshape(1, 100, 1)
        
        future_predictions_scaled = []
        
        # Generate forecasts
        for _ in range(days):
            pred = model.predict(future_input, verbose=0)[0]
            future_predictions_scaled.append(pred)
            future_input = np.append(future_input[:, 1:, :], [[pred]], axis=1)
        
        # Inverse transform to actual prices
        future_predictions = scaler.inverse_transform(future_predictions_scaled)
        
        # Create future dates
        last_date = stock_data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
        
        # Calculate trend and confidence metrics
        current_price = float(stock_data['Close'].iloc[-1])
        forecasted_prices = future_predictions.reshape(-1)
        
        trend = "upward" if forecasted_prices[-1] > current_price else "downward"
        price_change = forecasted_prices[-1] - current_price
        percentage_change = (price_change / current_price) * 100
        
        result = {
            "stock": stock,
            "current_price": current_price,
            "forecast": {
                "dates": future_dates.strftime('%Y-%m-%d').tolist(),
                "predicted_prices": [float(x) for x in forecasted_prices.tolist()]
            },
            "analysis": {
                "trend": trend,
                "price_change": float(price_change),
                "percentage_change": float(percentage_change),
                "forecast_period": f"{days} days"
            },
            "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/chart", methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        stock = data.get('stock', 'GOOG')
        chart_type = data.get('type', 'prediction')  # 'prediction' or 'forecast'
        
        # Get stock data
        end = datetime.now()
        start = datetime(end.year - 10, end.month, end.day)
        
        stock_data = yf.download(stock, start, end)
        
        if stock_data.empty:
            return jsonify({"error": "Invalid stock symbol or no data available"}), 400
        
        # Handle MultiIndex columns if they exist
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = [col[0] for col in stock_data.columns]
        
        plt.figure(figsize=(15, 6))
        
        if chart_type == 'moving_averages':
            # Moving averages chart
            stock_data['MA_100'] = stock_data['Close'].rolling(100).mean()
            stock_data['MA_200'] = stock_data['Close'].rolling(200).mean()
            stock_data['MA_250'] = stock_data['Close'].rolling(250).mean()
            
            plt.plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue')
            plt.plot(stock_data.index, stock_data['MA_100'], label='MA 100', color='red')
            plt.plot(stock_data.index, stock_data['MA_200'], label='MA 200', color='green')
            plt.plot(stock_data.index, stock_data['MA_250'], label='MA 250', color='orange')
            
            plt.title(f'{stock} - Close Price with Moving Averages')
            
        elif chart_type == 'forecast':
            # Forecast chart (simplified version)
            plt.plot(stock_data.index, stock_data['Close'], label='Historical Close', color='blue')
            plt.title(f'{stock} - Historical Close Price')
        
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        
        plot_base64 = base64.b64encode(plot_data).decode('utf-8')
        plt.close()
        
        return jsonify({
            "chart": plot_base64,
            "type": chart_type,
            "stock": stock
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
