# Stock Price Prediction using LSTM

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This project predicts the **closing price of the Bank Nifty index** using a **Long Short-Term Memory (LSTM) Recurrent Neural Network**. The model learns sequential patterns from historical stock data to forecast future prices.  

Key features include:

- Data visualization of stock trends.
- Sliding window preprocessing for LSTM input.
- LSTM model with dropout to prevent overfitting.
- Prediction plots for full dataset and month-specific analysis.

---

## Dataset

The dataset used is `bank_nifty.csv`, containing daily historical stock data:

| Column | Description |
|--------|-------------|
| Date   | Trading date (dd-MMM-yyyy) |
| Open   | Opening price of the index |
| High   | Highest price of the day |
| Low    | Lowest price of the day |
| Close  | Closing price of the day |
| Volume | Number of shares traded |

> **Note:** Ensure the CSV file is placed in the project directory before running the notebook.

---

## Project Structure

StockPriceRNN/
│
├── model.py # Main Jupyter Notebook with LSTM model
├── bank_nifty.csv # Dataset (not included if large)
├── README.md # Project documentation
├── .gitignore # Files to ignore in GitHub


---

## Model Architecture

1. **First LSTM Layer**: Captures sequential patterns from input sequences.  
2. **Second LSTM Layer**: Further processes the features extracted from the first layer.  
3. **Dense Layer**: Interprets features to make predictions.  
4. **Dropout Layer**: Prevents overfitting by randomly dropping neurons during training.  
5. **Output Layer**: Predicts the closing price.

**Training Metrics**:

- Loss function: Mean Absolute Error (MAE)  
- Metrics: Root Mean Squared Error (RMSE), MAE, Mean Squared Error (MSE)  
- Epochs: 25  
- Batch size: 32  

---

## Visualizations

## Opening vs Closing price trends

<img width="1003" height="528" alt="opening_v_closing_price" src="https://github.com/user-attachments/assets/2e7d82e6-c641-4ff1-a282-5968640f98fa" />


## Feature correlation heatmap

<img width="979" height="528" alt="correlation_between_different_features" src="https://github.com/user-attachments/assets/0868fe5c-b98c-4940-b007-8924655373ce" />

# Month-specific predictions (e.g., September)[Lookback days specified]
## 60 Days:
<img width="1178" height="613" alt="60_days_prediction" src="https://github.com/user-attachments/assets/6255681d-a614-41c4-960d-71fce1b5a25a" />

## 90 Days:

<img width="1178" height="613" alt="90_days" src="https://github.com/user-attachments/assets/64448ff3-698d-4963-8fe8-0aa38ff6fc29" />

## 120 Days:

<img width="1178" height="613" alt="120_days" src="https://github.com/user-attachments/assets/d13163f4-1eda-425f-8ea4-905fd99d40d9" />


## 180 Days:

<img width="1178" height="613" alt="180_das" src="https://github.com/user-attachments/assets/f2ede948-fef0-4da2-b0c3-7b213f37322c" />



---

## Requirements

- Python 3.8+
- TensorFlow 2.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn



