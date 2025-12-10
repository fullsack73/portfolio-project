# Portfolio Analysis and Optimization Tool

This is a comprehensive financial tool that provides portfolio analysis, optimization, and hedging strategies through a user-friendly web interface. The application is built with a React frontend and a Flask backend.

## ⚠️ Disclaimer

**This program is intended for academic and demonstration purposes only. It should not be used for making real financial decisions. The models and data have limitations and do not guarantee future results. Use at your own risk.**

## Features

This application offers a suite of powerful features for investors and financial analysts:

### 1. Stock Data Analysis
- **Historical Data Visualization**: View historical price charts for any stock ticker.
- **Trend Analysis**: Apply a LightGBM regression model to visualize trends based on historical data.

### 2. Portfolio Optimization
- **Efficient Frontier**: Visualize the efficient frontier for a given set of stocks to understand the optimal risk-return trade-offs.
- **Key Portfolio Metrics**: Calculate essential metrics such as:
    - Expected Return
    - Volatility (Risk)
    - Sharpe Ratio
- **Customizable Optimization**: Find the optimal asset allocation based on:
    - A predefined group of tickers (e.g., Dow Jones, S&P 500).
    - A custom list of tickers.
    - A specific target return.
    - Your personal risk tolerance.

### 3. Hedge Analysis
- **Pairs Analysis**: Analyze the statistical relationship between two stocks to identify potential hedging or pairs trading opportunities. The tool provides a regression analysis of the pair's historical price movements.

### 4. Financial Statement Analysis
- **Key Ratios**: Instantly retrieve and display key financial ratios for any given stock, providing quick insights into its financial health and performance.

## Technology Stack

- **Frontend**: React, Vite, Chart.js
- **Backend**: Flask, Python
- **Data Source**: yfinance
- **Core Libraries**:
    - `pandas` for data manipulation.
    - `numpy` for numerical operations.
    - `scikit-learn` and `lightgbm` for machine learning models.

## Getting Started

### Prerequisites

- Node.js and npm
- Python 3.x and pip

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd portfolio-project
    ```

2.  **Install backend dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install frontend dependencies:**
    ```bash
    npm install
    ```

### Running the Application

1.  **Start the Flask backend server:**
    Open a terminal and run:
    ```bash
    python src/app.py
    ```
    The backend will be running on `http://127.0.0.1:5000`.

2.  **Start the React frontend development server:**
    In a separate terminal, run:
    ```bash
    npm run dev
    ```
    The frontend will be accessible at `http://localhost:5173`.

## API Endpoints

The Flask backend provides the following API endpoints:

- `GET /get-data`: Fetches historical stock data. 
  - **Params**: `ticker`, `start_date`, `end_date`, `regression`, `future_days`
- `GET /analyze-hedge`: Analyzes the relationship between two tickers.
  - **Params**: `ticker1`, `ticker2`, `start_date`, `end_date`
- `GET /portfolio-metrics`: Calculates metrics for a portfolio.
  - **Params**: `tickers`, `start_date`, `end_date`, `riskless_rate`
- `GET /financial-statement`: Retrieves financial ratios for a ticker.
  - **Params**: `ticker`
- `POST /api/optimize-portfolio`: Optimizes a portfolio based on user inputs.
  - **Body**: `ticker_group`, `tickers`, `start_date`, `end_date`, `risk_free_rate`, `target_return`, `risk_tolerance`