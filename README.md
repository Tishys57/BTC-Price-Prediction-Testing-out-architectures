# Bitcoin Trend Prediction Using Statistical & Deep Learning Models

A hands-on time-series forecasting project exploring Bitcoin price prediction through a combination of **statistical analysis, classical time-series models, technical indicators, and recurrent neural networks**.

The project uses historical BTC-USD daily closing-price data and experiments with both traditional statistical forecasting methods (**ARIMA/SARIMA**) and deep learning approaches (**LSTM/GRU**) to investigate how different modeling techniques perform on cryptocurrency price data.

> **Project type:** Machine Learning / Time-Series Forecasting
> **Primary language:** Python
> **Environment:** Jupyter Notebook
> **Data source:** Yahoo Finance (`BTC-USD`)

---

## Overview

Cryptocurrency markets produce highly volatile and non-stationary time-series data, making them an interesting environment for experimenting with forecasting techniques.

This project follows a practical machine-learning workflow:

1. Collect historical Bitcoin price data
2. Explore and visualize the time series
3. Test the data for stationarity
4. Decompose the time series into trend, seasonal, and residual components
5. Apply differencing to address non-stationarity
6. Analyze autocorrelation and partial autocorrelation
7. Experiment with moving-average-based features
8. Build classical ARIMA and SARIMA models
9. Normalize the data for neural-network training
10. Construct sliding-window sequences
11. Train LSTM and GRU recurrent neural networks
12. Evaluate predictions using MSE and MAE
13. Compare predicted and actual Bitcoin prices

The project was designed as a practical exploration of **multiple modeling paradigms rather than relying on a single forecasting algorithm**.

---

## Objectives

The main objectives of the project are to:

* Understand the statistical properties of cryptocurrency price data.
* Investigate whether the raw Bitcoin price series is stationary.
* Apply transformations appropriate for non-stationary time series.
* Explore temporal dependencies using ACF and PACF.
* Experiment with classical statistical forecasting models.
* Experiment with recurrent neural networks designed for sequential data.
* Compare different modeling approaches using quantitative evaluation metrics.
* Develop an end-to-end machine-learning workflow using real-world financial data.

---

## Dataset

Historical Bitcoin data is downloaded using the [`yfinance`](https://github.com/ranaroussi/yfinance) library.

The notebook retrieves:

* **Ticker:** `BTC-USD`
* **Frequency:** Daily
* **Requested period:** January 2011 – December 2023
* **Primary variable used for forecasting:** Daily closing price

The project focuses primarily on the historical **Close** price rather than attempting to build a multi-factor trading system.

### Data acquisition

```python
import yfinance as yf

btc_data = yf.download(
    "BTC-USD",
    start="2011-01-01",
    end="2023-12-31",
    progress=False
)
```

---

# Methodology

## 1. Exploratory Data Analysis

The first stage visualizes Bitcoin's historical daily closing price to understand its overall behavior and volatility.

The project examines:

* Long-term price trends
* Large price movements
* Changes in volatility
* Potential temporal patterns

The raw time series is then used as the basis for subsequent statistical analysis.

---

## 2. Stationarity Analysis

A key assumption behind many classical time-series techniques is that the underlying series is stationary or can be transformed into a stationary series.

The project uses the **Augmented Dickey-Fuller (ADF) test** to investigate this.

### Raw Bitcoin price

The ADF test produced:

| Statistic     |   Value |
| ------------- | ------: |
| ADF Statistic | -1.3123 |
| p-value       |  0.6235 |

At the 5% significance level, the null hypothesis of a unit root is not rejected, indicating that the raw Bitcoin closing-price series is **non-stationary**.

---

## 3. Time-Series Decomposition

The Bitcoin price series is decomposed into:

* Observed series
* Trend
* Seasonal component
* Residual component

A multiplicative decomposition with a period of 365 was used to investigate potential annual-scale structure in the daily data.

This provides a visual way to examine whether the observed series can be represented through systematic trend and seasonal components alongside residual variation.

---

## 4. First-Order Differencing

Because the raw price series is non-stationary, first-order differencing is applied:

```python
btc_data_diff = btc_data["Close"].diff().dropna()
```

The ADF test is then repeated on the differenced series.

### Differenced Bitcoin price

| Statistic     |        Value |
| ------------- | -----------: |
| ADF Statistic |      -9.7353 |
| p-value       | 8.78 × 10⁻¹⁷ |

The substantially smaller p-value indicates that the differenced series satisfies the stationarity criterion used in this experiment.

This transformation also provides the `d=1` component used by the ARIMA/SARIMA experiments.

---

# 5. Moving-Average Features

The project experiments with three 30-day moving-average techniques:

### Simple Moving Average (SMA)

The 30-day SMA calculates the arithmetic mean of the previous 30 observations.

### Exponential Moving Average (EMA)

The 30-day EMA gives greater weight to more recent observations.

### Weighted Moving Average (WMA)

The 30-day WMA applies linearly increasing weights to observations within the window.

These indicators are primarily used for exploratory analysis and visualization in the current implementation.

```python
SMA_30 = close.rolling(window=30).mean()

EMA_30 = close.ewm(
    span=30,
    adjust=False
).mean()
```

---

# 6. ACF and PACF Analysis

The project uses:

* **Autocorrelation Function (ACF)**
* **Partial Autocorrelation Function (PACF)**

to investigate temporal relationships within the Bitcoin price series.

Both the raw and differenced series are examined.

These plots provide information about how observations at different time lags relate to one another and are useful when investigating possible autoregressive and moving-average structures.

---

# 7. Classical Time-Series Models

Two classical forecasting approaches are implemented.

## ARIMA

The project fits:

```text
ARIMA(1, 1, 1)
```

where:

* `p = 1` → autoregressive component
* `d = 1` → first-order differencing
* `q = 1` → moving-average component

The model is fitted using `statsmodels`.

### ARIMA diagnostics

The notebook also examines:

* Model residuals
* Residual autocorrelation
* Ljung-Box statistics
* Information criteria such as AIC and BIC

The fitted model produced:

```text
AIC: 54801.990
BIC: 54820.376
```

---

## SARIMA

A seasonal extension is also tested:

```text
SARIMA(1, 1, 1)(1, 1, 1, 7)
```

The seasonal period of `7` represents a weekly cycle for daily observations.

The fitted model produced:

```text
AIC: 54731.785
BIC: 54762.419
```

The notebook also analyzes the residuals and their autocorrelation.

> **Note:** The SARIMA implementation produces a near-singular covariance warning in the recorded run, meaning some estimated standard errors may be unstable. This is retained as an explicit limitation rather than treating the model as fully validated.

---

# 8. Data Normalization

For the neural-network experiments, Bitcoin closing prices are scaled to the range `[0, 1]` using `MinMaxScaler`.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
btc_data_scaled = scaler.fit_transform(
    btc_data_close
)
```

Normalization helps keep the numerical scale of the input suitable for neural-network optimization.

---

# 9. Sliding-Window Sequence Generation

The recurrent neural networks use a **30-day look-back window**.

For every prediction:

```text
Previous 30 days → Next day's closing price
```

For example:

```text
Day 1 ... Day 30 → Day 31
Day 2 ... Day 31 → Day 32
Day 3 ... Day 32 → Day 33
...
```

The resulting input tensor has the structure:

```text
(samples, time_steps, features)
```

with:

```text
time_steps = 30
features = 1
```

This converts the raw time series into supervised learning sequences suitable for recurrent neural networks.

---

# 10. Train/Test Split

The generated sequences are split chronologically:

```text
80% → Training
20% → Testing
```

The split is performed without shuffling so that the temporal ordering of the observations is preserved.

This is important for time-series modeling because randomly mixing observations could allow information from later periods to enter the training set.

---

# 11. LSTM Model

The first deep-learning model is a Long Short-Term Memory network.

### Architecture

```text
Input
  │
  ▼
LSTM (50 units)
  │
  ▼
Dropout (20%)
  │
  ▼
Dense (1 unit)
  │
  ▼
Predicted Bitcoin Price
```

Configuration:

* LSTM units: `50`
* Dropout: `0.2`
* Optimizer: Adam
* Learning rate: `0.001`
* Loss function: Mean Squared Error
* Epochs: `10`
* Batch size: `32`
* Look-back window: `30` days

---

# 12. GRU Model

A GRU model is implemented as a second recurrent architecture.

### Architecture

```text
Input
  │
  ▼
GRU (50 units)
  │
  ▼
Dropout (20%)
  │
  ▼
Dense (1 unit)
  │
  ▼
Predicted Bitcoin Price
```

Configuration:

* GRU units: `50`
* Dropout: `0.2`
* Optimizer: Adam
* Learning rate: `0.001`
* Loss function: Mean Squared Error
* Epochs: `10`
* Batch size: `32`
* Look-back window: `30` days

Using both LSTM and GRU provides a practical comparison between two commonly used recurrent architectures for sequential data.

---

# 13. Evaluation

The neural-network predictions are transformed back to their original USD scale before evaluation.

Two metrics are calculated:

### Mean Squared Error (MSE)

MSE measures the average squared difference between predicted and actual values.

```text
MSE = mean((y_actual - y_predicted)²)
```

Because the errors are squared, larger prediction errors have a greater effect on the metric.

### Mean Absolute Error (MAE)

MAE measures the average absolute difference:

```text
MAE = mean(|y_actual - y_predicted|)
```

MAE is easier to interpret in the original price units.

---

# Results

The recorded notebook run produced the following results on the held-out test portion:

| Model    |          MSE |      MAE |
| -------- | -----------: | -------: |
| **LSTM** | 2,542,194.83 | 1,237.12 |
| **GRU**  | 1,406,149.25 |   949.51 |

The predictions are also plotted against the corresponding actual Bitcoin closing prices to provide a visual comparison.

### Important interpretation

These results indicate that, **within this particular experiment and test split**, the GRU produced lower MSE and MAE than the LSTM.

However, this should not be interpreted as evidence that GRU is universally superior to LSTM or that the model can reliably predict future Bitcoin prices.

The experiment is intended primarily as a study of different forecasting approaches.

---

# Technology Stack

### Programming Language

* Python

### Data Acquisition

* `yfinance`

### Data Processing

* `pandas`
* `NumPy`

### Statistical Analysis

* `statsmodels`
* Augmented Dickey-Fuller test
* ACF / PACF
* Seasonal decomposition
* ARIMA
* SARIMA

### Machine Learning

* `scikit-learn`
* Min-Max normalization
* MSE
* MAE

### Deep Learning

* TensorFlow / Keras
* LSTM
* GRU
* Dropout
* Adam optimizer

### Visualization

* `Matplotlib`

### Development Environment

* Jupyter Notebook

---

# Project Structure

```text
.
├── btc-trend-prediction.ipynb
└── README.md
```

The main notebook contains the complete workflow, including:

* Data acquisition
* Exploratory analysis
* Statistical testing
* Feature analysis
* Classical forecasting
* Neural-network construction
* Model training
* Evaluation
* Visualization

---

# Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/btc-trend-prediction.git
cd btc-trend-prediction
```

Install the required dependencies:

```bash
pip install yfinance pandas numpy matplotlib statsmodels scikit-learn tensorflow jupyter
```

Alternatively, install them individually if working in an existing Python environment.

---

# Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
btc-trend-prediction.ipynb
```

Run the notebook cells sequentially.

The notebook downloads the historical Bitcoin data directly through Yahoo Finance, so an active internet connection is required during the data-acquisition step.

---

# Experimental Workflow

The complete pipeline can be summarized as:

```text
                 ┌─────────────────────┐
                 │   Yahoo Finance     │
                 │      BTC-USD        │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Exploratory Data    │
                 │      Analysis       │
                 └──────────┬──────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
      ┌───────────────┐           ┌────────────────┐
      │ Statistical   │           │ Neural Network │
      │ Analysis      │           │ Preprocessing  │
      └───────┬───────┘           └───────┬────────┘
              │                           │
      ┌───────┴────────┐          ┌───────┴────────┐
      │                │          │                │
      ▼                ▼          ▼                ▼
    ARIMA           SARIMA      LSTM              GRU
      │                │          │                │
      └────────┬───────┘          └───────┬────────┘
               │                          │
               └────────────┬─────────────┘
                            ▼
                  ┌──────────────────┐
                  │ Model Evaluation │
                  │   MSE / MAE      │
                  └──────────────────┘
```

---

# Key Takeaways

This project provided practical experience with several aspects of applied machine learning:

### Time-Series Analysis

Working with real-world temporal data requires understanding properties such as:

* Non-stationarity
* Autocorrelation
* Trend
* Seasonality
* Residual behavior

### Statistical Modeling

ARIMA and SARIMA demonstrate how classical statistical approaches can be applied to sequential data after appropriate preprocessing.

### Deep Learning

LSTM and GRU models provide an opportunity to experiment with neural architectures designed to model sequential dependencies.

### Model Comparison

Rather than relying on a single algorithm, the project evaluates multiple approaches and compares their behavior using quantitative metrics.

### End-to-End Implementation

The notebook covers the complete workflow from:

```text
Raw Data
   ↓
Analysis
   ↓
Preprocessing
   ↓
Feature/Sequence Construction
   ↓
Model Development
   ↓
Training
   ↓
Evaluation
   ↓
Visualization
```

---

# Limitations

There are several important limitations in the current implementation.

## 1. Historical Price Only

The neural networks primarily use historical closing prices.

The models do not currently incorporate:

* Trading volume
* Market capitalization
* Order-book information
* On-chain metrics
* Macroeconomic indicators
* Sentiment data
* News
* Social-media activity
* Other financial assets

Adding these variables could make the forecasting problem more representative of real-world market modeling.

## 2. Cryptocurrency Volatility

Bitcoin prices are highly volatile and influenced by many external factors.

Historical price patterns alone may not contain sufficient information to reliably predict future market behavior.

## 3. Limited Hyperparameter Search

The LSTM and GRU architectures use manually selected configurations rather than systematic hyperparameter optimization.

Potential improvements include:

* Grid search
* Random search
* Bayesian optimization
* Automated experiment tracking

## 4. Scaling Procedure

In the current notebook, the `MinMaxScaler` is fitted before the chronological train/test split.

For a stricter time-series evaluation methodology, the scaler should be fitted **only on the training data** and then applied to the test data. This prevents information from the test period from influencing the preprocessing transformation.

## 5. Validation Strategy

The current neural-network training uses the test set as `validation_data`.

A more rigorous experimental design would separate the data into:

```text
Training set
Validation set
Test set
```

The final test set should ideally remain untouched until the end of model development.

## 6. Forecasting Scope

The current implementation predicts the next closing-price value based on a 30-day historical window.

It is not a complete automated trading strategy and does not account for:

* Transaction costs
* Slippage
* Position sizing
* Risk management
* Entry/exit rules
* Portfolio allocation
* Trading execution

Therefore, model prediction metrics should not be interpreted as trading returns.

---

# Future Improvements

Several extensions could make this project more robust and closer to a production-oriented forecasting system.

### Feature Engineering

Introduce additional features such as:

* Trading volume
* Volatility
* RSI
* MACD
* Bollinger Bands
* Additional moving averages
* Returns and log returns

### External Data

Incorporate:

* Market sentiment
* News sentiment
* Social-media activity
* Macroeconomic indicators
* Blockchain/on-chain metrics

### Improved Evaluation

Use:

* Walk-forward validation
* Expanding-window evaluation
* Rolling-window evaluation
* Separate validation and test sets
* Multiple forecasting horizons

### Model Development

Experiment with:

* Bidirectional LSTM
* Stacked LSTM/GRU networks
* 1D CNN + LSTM
* Transformer-based time-series models
* XGBoost / LightGBM
* Ensemble forecasting

### Experiment Tracking

A future version could use experiment-tracking tools to record:

* Hyperparameters
* Training runs
* Metrics
* Model versions
* Dataset versions

This would make comparisons between experiments more systematic and reproducible.

### Deployment

The trained model could eventually be exposed through:

* A REST API
* A lightweight web dashboard
* A scheduled prediction pipeline
* A containerized service

---

# Reproducibility

The project is implemented as a Jupyter Notebook so that each stage of the workflow can be inspected and reproduced.

The main variables controlling the neural-network experiment include:

```python
look_back = 30
train_size = int(len(X) * 0.8)

epochs = 10
batch_size = 32

learning_rate = 0.001
```

Because the data is retrieved dynamically from Yahoo Finance and the neural networks are trained from scratch, exact model outputs can vary between executions.

---

# Disclaimer

This project is an educational and experimental machine-learning project.

It is **not financial advice**, and the models should not be interpreted as reliable investment or trading signals.

Past price behavior does not guarantee future results, particularly in highly volatile markets such as cryptocurrency.

---

# Why This Project

This project was built to gain practical experience with the process of taking a real-world dataset and turning it into an end-to-end machine-learning experiment.

Rather than focusing exclusively on one algorithm, the project explores several approaches—from statistical time-series models to recurrent neural networks—and examines how preprocessing, temporal structure, architecture, and evaluation affect the forecasting task.

The broader goal is to develop the habit of **building, experimenting, evaluating, identifying limitations, and iterating**, rather than treating a machine-learning model as a black box.

---

## Author

**[Your Name]**

* GitHub: [Your GitHub Profile]
* LinkedIn: [Your LinkedIn Profile]
* Portfolio: [Your Portfolio Website]

---

## License

This project is provided for educational and research purposes. Add a license such as MIT if you intend to explicitly permit reuse and modification.
