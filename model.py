from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

app = Flask(__name__)

# Sample dataset
periods_dates = ["2024-09-26", "2024-10-10", "2024-11-15", "2024-12-12", "2025-01-04"]
base_date = datetime.strptime(periods_dates[0], "%Y-%m-%d")
days_since_start = [(datetime.strptime(date, "%Y-%m-%d") - base_date).days for date in periods_dates]

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(np.array(days_since_start).reshape(-1, 1))

# Prepare training data
x, y = [], []
for i in range(1, len(scaled_data)):
    x.append(scaled_data[i-1:i, 0])
    y.append(scaled_data[i, 0])

x = np.array(x).reshape((len(x), 1, 1))
y = np.array(y)

# Build and train LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(1, 1)),
    LSTM(units=50),
    Dense(units=1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x, y, epochs=200, batch_size=1, verbose=1)  # Increased epochs and batch size of 1

# Save & load the model
model.save("lstm_model.h5")
loaded_model = load_model("lstm_model.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_date = request.form['date']
        input_days = (datetime.strptime(input_date, "%Y-%m-%d") - base_date).days
        input_scaled = scaler.transform(np.array([[input_days]]))
        input_scaled = np.array(input_scaled).reshape((1, 1, 1))
        
        prediction_scaled = loaded_model.predict(input_scaled)
        prediction_days = scaler.inverse_transform(prediction_scaled)[0][0]

        # Convert predicted days back to a date
        predicted_date = base_date + timedelta(days=int(np.round(prediction_days)))  # Round the prediction
        predicted_date_str = predicted_date.strftime("%Y-%m-%d")

        return jsonify({'prediction': predicted_date_str})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)