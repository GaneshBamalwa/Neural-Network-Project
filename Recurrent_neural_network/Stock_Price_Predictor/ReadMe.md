# Stock Price Prediction using LSTM 📈

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project predicts the **closing price of the Bank Nifty index** using a **Long Short-Term Memory (LSTM) Recurrent Neural Network**.  
The model captures sequential dependencies in time-series data and forecasts future prices based on historical data.  

**Key Features:**
- Visualization of historical stock trends.  
- Sliding window preprocessing for LSTM input sequences.  
- LSTM network with dropout for regularization.  
- Prediction plots for both full dataset and month-specific analysis.  

---

## 📂 Dataset

The dataset used is **`bank_nifty.csv`**, which contains daily historical stock data.

| Column | Description |
|--------|-------------|
| Date   | Trading date (dd-MMM-yyyy) |
| Open   | Opening price of the index |
| High   | Highest price of the day |
| Low    | Lowest price of the day |
| Close  | Closing price of the day |
| Volume | Number of shares traded |

> ⚠️ Place the CSV file in the project directory before running the notebook.  


---

## 🏗 Project Structure
StockPricePrediction/<br>
│
├── model.py<br>
├── dataset/<br>
│ └── bank_nifty.csv<br>
├── README.md<br>



---

## 🔬 Model Architecture

1. **LSTM Layer 1** → Extracts sequential patterns.  
2. **LSTM Layer 2** → Processes deeper temporal features.  
3. **Dense Layer** → Interprets extracted features.  
4. **Dropout Layer** → Reduces overfitting.  
5. **Output Layer** → Predicts the closing price.  

**Training Setup:**  
- **Loss Function:** Mean Absolute Error (MAE)  
- **Metrics:** RMSE, MAE, MSE  
- **Epochs:** 25  
- **Batch Size:** 32  

---

## 📊 Visualizations

### Opening vs Closing Price Trends
<img width="1003" height="528" alt="opening_v_closing_price" src="https://github.com/user-attachments/assets/2e7d82e6-c641-4ff1-a282-5968640f98fa" />

### Feature Correlation Heatmap
<img width="979" height="528" alt="correlation_between_different_features" src="https://github.com/user-attachments/assets/0868fe5c-b98c-4940-b007-8924655373ce" />

### Month-Specific Predictions (Lookback Window)
- **60 Days**
  <img width="1178" height="613" alt="60_days_prediction" src="https://github.com/user-attachments/assets/6255681d-a614-41c4-960d-71fce1b5a25a" />

- **90 Days**
  <img width="1178" height="613" alt="90_days" src="https://github.com/user-attachments/assets/64448ff3-698d-4963-8fe8-0aa38ff6fc29" />

- **120 Days**
  <img width="1178" height="613" alt="120_days" src="https://github.com/user-attachments/assets/d13163f4-1eda-425f-8ea4-905fd99d40d9" />

- **180 Days**
  <img width="1178" height="613" alt="180_days" src="https://github.com/user-attachments/assets/f2ede948-fef0-4da2-b0c3-7b213f37322c" />

---

##  Inference

- The model achieves a prediction accuracy within **100–200 points** of the actual Bank Nifty prices when using a **120-day lookback window**.  
- This shows that the LSTM successfully captures medium-term sequential patterns, but accuracy can still vary depending on market volatility and lookback choice.  
- While LSTMs are powerful for sequence modeling, they struggle with **long-term dependencies** due to vanishing gradients and limited memory capacity.  

###  Future Directions
- **Transformer-based Models (e.g., BERT, GPT, Time Series Transformers)**: Unlike RNNs/LSTMs, Transformers rely on *attention mechanisms*, which allow them to learn relationships across **entire sequences at once**, making them highly effective for time-series forecasting.  
- **Hybrid Approaches**: Combining CNNs (for feature extraction) with Transformers or LSTMs (for temporal modeling).  
- **Advanced Architectures**: Experimenting with GRU, Bidirectional LSTM, and Attention-based LSTMs.  
- **Deployment & Real-Time Prediction**: Creating a **Streamlit dashboard** for live Bank Nifty predictions.  

This project laid the foundation for understanding sequential modeling with RNNs and LSTMs. The next step is to explore **attention-based architectures and Transformers**, which represent the state-of-the-art in modeling sequential and financial time-series data.


