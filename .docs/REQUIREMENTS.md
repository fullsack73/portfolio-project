This document outlines the functional and non-functional requirements for the Portfolio Analysis and Optimization tool. The requirements are derived from the source code of the application.

### I. Core Features

The application is a comprehensive financial analysis tool that provides the following core functionalities:
- **Stock Data Analysis:** Visualization of historical stock data, regression analysis, and future price prediction.
- **Hedge Analysis:** Statistical analysis to determine if two assets can serve as a hedge for each other.
- **Portfolio Analysis:** Classic portfolio optimization based on Monte Carlo simulation to find the tangency portfolio (maximum Sharpe ratio).
- **Financial Statement Analysis:** Display of key financial ratios for a given company.
- **Advanced Portfolio Optimizer:** Sophisticated portfolio optimization based on user-defined constraints such as target return or risk tolerance.
- **Internationalization:** Full support for English and Korean languages throughout the user interface.

### II. User Interface (UI)

- **Navigation:** A collapsible sidebar allows users to switch between the different analysis modules.
- **Language Selection:** A floating button enables users to toggle the UI language between English and Korean.
- **Dynamic Views:** The main content area dynamically renders the component corresponding to the selected analysis module.
- **Data Visualization:** Interactive charts, powered by Plotly.js, are used to display all financial data and analysis results.
- **User Feedback:** The application provides clear loading and error states to keep the user informed during data-fetching and computation processes.

### III. Detailed Functional Requirements

#### 1. Stock Data Analysis (`stock` view)
- **1.1. Data Fetching & Display:**
  - Users can input a single stock ticker symbol (e.g., "AAPL").
  - Users can select a start and end date for the analysis period.
  - The application fetches and displays a line chart of the historical closing prices for the selected stock and period.
- **1.2. Regression Analysis:**
  - The application automatically performs a regression analysis on the historical data using a LightGBM model.
  - A regression line is overlaid on the stock price chart to show the trend.
- **1.3. Future Price Prediction:**
  - Users can input a number of future days for which to predict the stock price.
  - The application uses the trained regression model to forecast future prices and displays the prediction on a separate chart.

#### 2. Hedge Analysis (`hedge` view)
- **2.1. Input & Calculation:**
  - Users can input two different stock ticker symbols.
  - Users can optionally specify a date range for the analysis.
  - The backend calculates the Pearson correlation coefficient between the daily returns of the two stocks.
- **2.2. Results Display:**
  - The UI displays the company names for the two tickers.
  - It indicates whether the pair is considered a viable hedge (defined as a correlation < -0.5).
  - It shows the calculated correlation, the p-value, and the strength of the relationship (Strong, Moderate, or Weak).

#### 3. Portfolio Analysis (`portfolio` view)
- **3.1. Input & Calculation:**
  - Users can input a list of comma-separated stock tickers.
  - Users must provide a start date, end date, and a risk-free rate (as a percentage).
  - The backend uses a Monte Carlo simulation to find the optimal portfolio that maximizes the Sharpe ratio.
- **3.2. Results & Visualization:**
  - The UI displays the optimal asset weights, the portfolio's expected annual return, volatility (risk), and Sharpe ratio.
  - A graph visualizes the efficient frontier, the capital market line, the minimum volatility portfolio, and the tangency (optimal) portfolio.

#### 4. Financial Statement Analysis (`financial` view)
- **4.1. Input & Data Fetching:**
  - Users can input a single stock ticker.
  - The application fetches financial data from the backend.
- **4.2. Metrics Display:**
  - The UI displays the following key financial ratios for the selected company:
    - Price to Earnings (P/E)
    - Price to Book (P/B)
    - Price to Sales (P/S)
    - Debt Ratio
    - Liquidity Ratio (Current Ratio)
  - Each metric is accompanied by a tooltip explaining its significance.

#### 5. Portfolio Optimizer (`optimizer` view)
- **5.1. Input Configuration:**
  - Users can select a predefined group of tickers (S&P 500, Dow Jones) or upload a custom list from a CSV file.
  - Users must specify a start date, end date, and a risk-free rate.
  - Users can optionally provide a target annual return or a target annual volatility (risk tolerance). If neither is provided, the optimizer defaults to finding the portfolio with the maximum Sharpe ratio.
- **5.2. Optimization & Results:**
  - The backend uses the `PyPortfolioOpt` library to perform the optimization.
  - The UI displays the resulting asset weights, expected return, risk, and Sharpe ratio.
- **5.3. Investment Allocation:**
  - After optimization, users can input a total investment budget.
  - The application calculates and displays a table showing how the budget would be allocated across the assets, including the dollar amount and the number of shares for each ticker.

### IV. Non-Functional Requirements

- **Internationalization (i18n):** All text in the UI is translated into English and Korean.
- **API Backend:** A Python Flask server provides a RESTful API for all data fetching and financial calculations.

### V. Technology Stack

- **Frontend:**
  - **Framework:** React.js
  - **Charting:** Plotly.js
  - **Internationalization:** i18next
  - **HTTP Client:** Axios, Fetch API
- **Backend:**
  - **Framework:** Flask
  - **Data Source:** yfinance
  - **Core Libraries:** pandas, numpy, SciPy
  - **Machine Learning/Optimization:** scikit-learn, LightGBM, PyPortfolioOpt
- **Styling:**
  - Plain CSS with a modular structure.
