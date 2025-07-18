
# Project Tasks

This file outlines the tasks completed during the development of the portfolio analysis and optimization tool.

## Project Setup & Foundation
- [x] Initialize React frontend application.
- [x] Set up Python backend server.
- [x] Establish communication between frontend and backend.
- [x] Initialize Git for version control.
- [x] Define project structure and file organization.
- [x] Configure ESLint for code quality.

## Backend Development
- [x] Create API endpoint to fetch historical stock data.
- [x] Implement Monte Carlo simulation for portfolio projection.
- [x] Develop portfolio optimization logic using the Markowitz model.
- [x] Implement hedging analysis to find correlated assets.
- [x] Create endpoint to serve financial statement data.
- [x] Add utility to read stock tickers from CSV files.

## Frontend Development
- [x] Design the main application layout and UI components.
- [x] Create `TickerInput` for stock selection.
- [x] Develop `PortfolioInput` to define asset allocations.
- [x] Implement `StockChart` to visualize historical price data.
- [x] Create `PortfolioGraph` to display portfolio performance.
- [x] Build `FinancialStatement` component to show company financials.
- [x] Develop `Optimizer` component for portfolio optimization.
- [x] Create `Hedge` component for hedging analysis.
- [x] Implement `FutureChart` to display Monte Carlo simulation results.
- [x] Add date input components for selecting time ranges.

## Internationalization (i18n)
- [x] Integrate `i18next` for internationalization.
- [x] Create translation files for English (`en`) and Korean (`ko`).
- [x] Implement `LanguageSelector` component.
- [x] Wrap UI text with translation functions.

## Documentation
- [x] Write `README.md` with a project overview and setup instructions.
- [x] Create `REQUIREMENTS.md` to detail project requirements.
- [x] Write `DESIGN.md` to outline the technical design and architecture.
- [x] Create this `TASKS.md` file to track project development progress.

## ML-Based Portfolio Optimization
- [x] **Backend:** Integrate `Prophet` for time-series forecasting of asset returns.
- [x] **Backend:** Update `portfolio_optimization.py` to include a function for ML-based return forecasting.
- [x] **Backend:** Modify the `optimize_portfolio` function and the API endpoint in `app.py` to handle a `use_ml_forecast` flag.
- [x] **Frontend:** Add a checkbox to the `Optimizer.jsx` component to enable ML forecasting.
- [x] **Frontend:** Update `handleSubmit` in `Optimizer.jsx` to send the `use_ml_forecast` flag to the backend.
- [x] **i18n:** Add translations for the new UI element in both English and Korean.
- [x] **Documentation:** Update `REQUIREMENTS.md` and `DESIGN.md` to reflect the new feature.

## Default ML Portfolio Optimization
- [x] **Backend:** Modify `portfolio_optimization.py` to use ML forecasting by default.
- [x] **Backend:** Update the API endpoint in `app.py` to remove the ML-related flag.
- [x] **Frontend:** Remove the ML forecast checkbox and its state from `Optimizer.jsx`.
- [x] **i18n:** Remove translations for the obsolete UI element.
- [x] **Documentation:** Update `REQUIREMENTS.md` and `DESIGN.md` to reflect the new default behavior.

## Debugging Prophet Integration
- [x] **Logging:** Suppress verbose informational logs from `cmdstanpy` to improve debugging clarity.
- [x] **Backend:** Add enhanced error handling and logging within the `forecast_returns` function to catch potential failures.
- [x] **Backend:** Log the input data passed to the forecast function to ensure data integrity.
- [x] **Documentation:** Update `DESIGN.md` to document the new logging and error handling strategy.

## Robust Data Handling and Logging
- [x] **Logging:** Implement a more effective method to suppress `prophet` and `cmdstanpy` logs.
- [x] **Backend:** Refactor `get_stock_data` to fetch data per-ticker to handle individual download failures gracefully.
- [x] **Backend:** Add logging to report which tickers are skipped due to download errors.
- [x] **Documentation:** Update `DESIGN.md` to document the resilient data fetching mechanism.

## Final Data & Log Handling
- [x] **Backend:** Implement intelligent data filling (`ffill`, `bfill`) in `get_stock_data` to create a robust, complete dataset and prevent covariance errors.
- [x] **Backend:** Forcefully suppress `cmdstanpy` logs via monkey-patching its internal logging function.
- [x] **Documentation:** Update `DESIGN.md` to document the new data filling strategy and log suppression method.

## Advanced Data & Log Handling
- [x] **Backend:** Implement a ticker sanitization function to fix common `yfinance` symbol issues (e.g., `BRK.B` -> `BRK-B`).
- [x] **Backend:** Forcefully suppress `cmdstanpy` logs by redirecting `stdout`/`stderr` during model fitting.
- [x] **Backend:** Switch to `CovarianceShrinkage` risk model in `PyPortfolioOpt` for more robust calculations.
- [x] **Documentation:** Update `DESIGN.md` to reflect the new ticker sanitization and robust risk model.

## `NameError` Hotfix
- [x] **Backend:** Add missing `NullHandler` import to `portfolio_optimization.py`.

## Suppress `yfinance` Warning
- [x] **Backend:** Add `auto_adjust=True` to all `yf.download` calls to silence the `FutureWarning`.

## Fix Data Alignment Error
- [x] **Backend:** Synchronize tickers between the expected returns and covariance matrix to fix the `ValueError`.
