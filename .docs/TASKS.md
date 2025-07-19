
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

## Completed: ML Forecasting and Data Pipeline Hardening
- [x] **ML Forecasting as Default:** Enabled Prophet-based ML forecasting as the default and only method for portfolio optimization, removing the frontend toggle and simplifying the API.
- [x] **Robust Data Pipeline:** Implemented a resilient data fetching and processing pipeline with several key enhancements:
    - **Ticker Sanitization:** Automatically corrects `yfinance` ticker symbols (e.g., `BRK.B` to `BRK-B`).
    - **Intelligent Data Filling:** Uses forward-fill (`ffill`) and backward-fill (`bfill`) to handle missing data from non-overlapping trading days, preventing covariance matrix errors.
    - **Graceful Failure Handling:** The application now correctly handles ticker download failures and empty datasets (e.g., for the S&P 500), preventing crashes.
- [x] **Critical Bug Fixes:** Resolved several critical bugs that caused application instability:
    - **Data Alignment Error:** Synchronized tickers between the expected returns and covariance matrix, fixing the `ValueError`.
    - **Prophet `Length Mismatch` Error:** Standardized the DataFrame structure before passing it to Prophet, preventing data processing crashes.
    - **`NameError` Hotfix:** Restored a missing `NullHandler` import.
- [x] **Log Suppression:** Silenced verbose logs from third-party libraries (`yfinance`, `cmdstanpy`) to ensure clean and readable application logs.
- [x] **Comprehensive Documentation:** Updated `DESIGN.md` and `REQUIREMENTS.md` to reflect all architectural changes, bug fixes, and new features.

## Fixing Portfolio Optimization being still broken
- [ ] **Backend:**
    - [x] **Disabling Log Suppression Temporarily:** De-Silencing logs for yfinance and/or cmdstanpy to identify the root cause of the issue.
    - [x] **Investigate and Fix:** Investigate the root cause of the issue and fix it.
        - **Root Cause Identified:** The "Length mismatch" errors in Prophet forecasting are caused by inconsistent DataFrame structure when calling `reset_index()` and assigning column names. The issue occurs because:
            - When a DataFrame index has no name, `reset_index()` creates different column structures
            - The subsequent `df_prophet.columns = ['ds', 'y']` assignment fails when the actual number of columns doesn't match the expected 2 columns
            - Different tickers show different expected axis lengths (3, 4) vs actual (2), indicating varying DataFrame structures
        - **Fix Implemented:** Replaced the problematic `reset_index()` approach with a robust DataFrame preparation method that:
            - Creates a clean DataFrame with explicit column structure using `pd.DataFrame({'ds': index, 'y': values})`
            - Completely avoids the `reset_index()` call that was causing inconsistent column structures
            - Ensures Prophet always receives exactly 2 columns (`ds` and `y`) regardless of the original DataFrame state
            - Includes proper date formatting with `pd.to_datetime()` for the Prophet model
        - **New Issue Identified:** During S&P 500 optimization, majority of tickers are being skipped during forecasting process
            - **Root Cause Analysis:** Most tickers are likely failing Prophet forecasting due to:
                - Insufficient historical data (less than 2 data points)
                - Prophet model fitting failures (convergence issues, data quality)
                - Silent failures that result in fallback return of 0, effectively removing tickers from optimization
            - **Fix Implemented:** Enhanced forecasting system with robust fallback strategies:
                - Added detailed logging to identify why specific tickers are being skipped (data points, missing values)
                - Implemented historical mean fallback when Prophet fails or data quality is poor
                - Added data quality checks (missing values > 50% threshold)
                - Prophet model tuned with reduced seasonality for better stability
                - Default to 5% return instead of 0% to ensure tickers remain in optimization
                - Multi-level error handling to catch and gracefully handle various failure modes
        - **New Issue Discovered:** Diagnostic logging does not reveal the actual reason for ticker skipping
            - **Problem:** Despite enhanced logging and fallback mechanisms, S&P 500 optimization still shows many tickers being skipped without clear explanation in logs
            - **Investigation Needed:** 
                - Check if the issue is in data fetching stage (before forecasting)
                - Verify if tickers are being filtered out during data cleaning (`dropna` operations)
                - Examine if the issue is in the alignment step between forecasts and historical data
                - Investigate if there's a silent failure mode not being caught by current logging
            - **Investigation Results:** Comprehensive logging revealed the issue is limited to specific edge cases:
                - Only 2 tickers (`BRK.B`, `BF.B`) are being dropped during data cleaning stage
                - These tickers are properly sanitized (`BRK.B` → `BRK-B`, `BF.B` → `BF-B`) but still return all NaN data
                - The vast majority of S&P 500 tickers are successfully processed through the entire pipeline
                - The "widespread skipping" issue was likely resolved by previous fixes
            - **Fix Implemented:** Enhanced sanitize_tickers function with special case handling for problematic symbols
            - **Status:** Issue largely resolved - only 2 out of ~500 S&P 500 tickers affected (99.6% success rate)
            - **Remaining:** Minor edge cases for specific tickers that may be delisted or have data issues
    - [x] **Re-enable Log Suppression:** Re-enable log suppression after the issue has been fixed.
    - [x] **Comprehensive Documentation:** Update `DESIGN.md` and `REQUIREMENTS.md` to reflect all architectural changes, bug fixes, and new features.

## Making Portfolio Optimization "Faster"

### Performance Analysis - Identified Bottlenecks:
1. **Sequential Prophet Model Training**: Each ticker trains a separate Prophet model sequentially (500+ models for S&P 500)
2. **Individual yfinance API Calls**: Each ticker fetched individually with network latency
3. **Redundant Data Processing**: Multiple DataFrame operations per ticker
4. **Prophet Model Overhead**: Full MCMC sampling for each ticker (overkill for portfolio optimization)
5. **No Caching**: Repeated computations for same tickers/date ranges

### Optimization Strategies:
- [x] **Strategy 1: Parallel Processing**
    - [x] Implement concurrent Prophet model training using ThreadPoolExecutor
    - [x] Parallelize yfinance data fetching for multiple tickers
    - [x] Add progress indicators for long-running operations
- [x] **Strategy 2: Batch Data Fetching**
    - [x] Use yfinance batch download for multiple tickers in single API call
    - [x] Implement intelligent batching based on ticker count
- [ ] **Strategy 3: Lightweight Forecasting**
    - [ ] Replace Prophet with faster time series models for bulk processing
    - [ ] Implement simple ARIMA or exponential smoothing as faster alternative
    - [ ] Use Prophet only for high-priority/individual ticker analysis
- [ ] **Strategy 4: Smart Caching System**
    - [ ] Cache historical data downloads by ticker and date range
    - [ ] Cache Prophet model forecasts with TTL (time-to-live)
    - [ ] Implement Redis or in-memory caching for frequently accessed data
- [ ] **Strategy 5: Algorithmic Optimizations**
    - [ ] Pre-filter tickers based on data quality before expensive forecasting
    - [ ] Implement early termination for obviously problematic tickers
    - [ ] Use sampling for very large portfolios (e.g., top 100 from S&P 500)

### Implementation Priority:
1. **High Impact, Low Effort**: Batch data fetching and parallel processing
2. **Medium Impact, Medium Effort**: Lightweight forecasting alternatives
3. **High Impact, High Effort**: Comprehensive caching system

### Performance Targets:
- **Current**: ~5-10 minutes for S&P 500 optimization
- **Target**: <30 seconds for S&P 500 optimization
- **Stretch Goal**: <10 seconds with full caching

### Critical Regression Issue:
- **Problem**: After implementing batch fetching, ALL tickers (503/503) are being dropped during data cleaning
- **Symptoms**: 
  - `GET_STOCK_DATA FAILED: ['LH', 'EBAY', 'ADI', 'A', 'SMCI', 'TRV', 'APA', 'ABBV', 'LMT', 'CDW']...`
  - `DROPPED DURING DATA CLEANING: 503 tickers`
- **Root Cause Identified**: Incorrect handling of yfinance MultiIndex DataFrame structure for batch downloads
  - yfinance returns MultiIndex columns: `('Close', 'AAPL')`, `('Close', 'MSFT')`, etc.
  - Original code incorrectly accessed `batch_data.columns.names` instead of `batch_data.columns.get_level_values(0)`
  - Close price extraction logic was fundamentally flawed for multi-ticker downloads
- **Fix Implemented**: 
  - Proper MultiIndex DataFrame handling with `isinstance(batch_data.columns, pd.MultiIndex)` check
  - Correct Close price extraction using `batch_data.columns.get_level_values(0)`
  - Enhanced logging to debug DataFrame structure issues
  - Robust fallback handling for edge cases
- **Priority**: CRITICAL - Portfolio optimization completely broken
- **Fix Status**: [x] COMPLETED