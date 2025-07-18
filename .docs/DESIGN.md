# Design and Architecture Document

This document provides an overview of the design, architecture, and technology stack for the Portfolio Analysis and Optimization tool.

## 1. Architectural Overview

The application follows a classic **client-server architecture**.

-   **Client (Frontend):** A single-page application (SPA) built with React. It is responsible for rendering the user interface, managing user interactions, and communicating with the backend via API calls.
-   **Server (Backend):** A Python-based server built with the Flask framework. It exposes a RESTful API that the frontend consumes. The backend handles all business logic, including fetching financial data, performing statistical analysis, and running portfolio optimizations.

This separation of concerns allows for independent development and deployment of the frontend and backend.

## 2. Frontend Design

The frontend is structured as a modern React application, emphasizing a component-based architecture.

### 2.1. Component Structure

The UI is broken down into reusable components, each with a specific responsibility.

-   **`App.jsx`**: The main application component. It acts as the central controller, managing the overall application state, such as the currently active view (`activeView`), and orchestrating data fetching.
-   **View Components (`Hedge.jsx`, `Optimizer.jsx`, etc.):** Each major feature of the application has a corresponding top-level component that encapsulates its UI and logic. `App.jsx` conditionally renders one of these components based on the `activeView` state.
-   **UI Components (`TickerInput.jsx`, `DateInput.jsx`, `Selector.jsx`):** These are smaller, reusable components used across different views for common UI elements like forms, buttons, and navigation.
-   **Chart Components (`StockChart.jsx`, `PortfolioGraph.jsx`):** Specialized components that use the `react-plotly.js` library to render interactive data visualizations.

### 2.2. State Management

State is managed locally within components using React Hooks (`useState`, `useEffect`). For global concerns like the active view or language selection, state is lifted up to the nearest common ancestor, `App.jsx`, and passed down to child components via props. This approach is simple and effective for the current scale of the application.

### 2.3. Styling

Styling is handled with plain CSS (`App.css`, `index.css`). The styles are organized functionally, with specific class names for different components and layouts (e.g., `.hedge-analysis`, `.optimizer-container`). This keeps the styling straightforward and maintainable.

### 2.4. Internationalization (i18n)

The application supports both English and Korean.
-   **`i18next` and `react-i18next`**: These libraries are used to manage translations.
-   **Translation Files**: JSON files located in `src/locales/` store the translation strings for each language.
-   **`LanguageSelector.jsx`**: This component allows the user to switch the language, which is then managed by the `i18next` instance.

## 3. Backend Design

The backend is a modular Flask application designed to serve data and perform complex calculations efficiently.

### 3.1. API Server (`app.py`)

`app.py` is the entry point for the backend server. It defines the Flask application and its API endpoints. It uses Flask-CORS to handle Cross-Origin Resource Sharing, allowing the React development server to make requests to it.

### 3.2. API Endpoints

The server exposes several endpoints, each mapped to a specific function:

-   `GET /get-data`: Fetches historical stock data, performs regression, and predicts future prices.
-   `GET /analyze-hedge`: Analyzes the hedge relationship between two tickers.
-   `GET /portfolio-metrics`: Calculates portfolio metrics using a Monte Carlo simulation.
-   `GET /financial-statement`: Retrieves key financial ratios for a single ticker.
-   `POST /api/optimize-portfolio`: Runs advanced portfolio optimization based on various constraints.

### 3.3. Modular Business Logic

The core business logic is decoupled from the API routes and organized into separate Python modules:

-   **`hedge_analysis.py`**: Contains the logic for calculating correlation and determining hedge relationships.
-   **`montecarlo.py`**: Implements the Monte Carlo simulation for basic portfolio optimization.
-   **`financial_statement.py`**: Handles fetching and calculating financial ratios.
-   **`portfolio_optimization.py`**: Implements the advanced portfolio optimization using the `PyPortfolioOpt` library.
-   **`ticker_lists.py`**: Provides utility functions to get lists of tickers for major indices (S&P 500, Dow Jones) from local CSV files.

This modularity makes the code easier to understand, maintain, and test.

## 4. Technology Stack

### Frontend
-   **React**: Core library for building the user interface.
-   **Plotly.js / react-plotly.js**: For creating interactive charts and graphs.
-   **i18next / react-i18next**: For handling internationalization and translations.
-   **Axios / Fetch API**: For making HTTP requests to the backend.
-   **Vite**: Frontend build tool and development server.

### Backend
-   **Flask**: Micro web framework for the API server.
-   **yfinance**: For fetching historical market data from Yahoo Finance.
-   **pandas**: For data manipulation and analysis.
-   **NumPy**: For numerical operations, especially in financial calculations.
-g   **SciPy**: For scientific and technical computing, used here for optimization.
-   **LightGBM / scikit-learn**: For machine learning (regression model).
-   **PyPortfolioOpt**: For advanced portfolio optimization and analysis.

## 5. Data Flow Example (Stock Analysis)

1.  **User Interaction**: The user selects the "Stock Data" view, enters a ticker ("AAPL"), and selects a date range.
2.  **Component State**: The `TickerInput` and `DateInput` components update their local state. This state change is communicated up to the `App.jsx` component.
3.  **API Call**: `App.jsx` triggers the `fetchData` function, which makes a GET request to the `/api/get-data?ticker=AAPL&...` endpoint on the Flask server.
4.  **Backend Processing**:
    -   The Flask route handler receives the request.
    -   It calls the `generate_regression_data` function.
    -   This function uses `yfinance` to download the historical data for AAPL.
    -   It then uses `pandas` for feature engineering and `LightGBM` to train a model and generate regression/prediction data.
5.  **JSON Response**: The backend formats the results (prices, regression data, company name) into a JSON object and sends it back to the frontend.
6.  **Frontend State Update**: The `.then()` block of the fetch call in `App.jsx` receives the JSON data and updates the application's state (`data`, `regressionData`, etc.) using `useState`.
7.  **UI Re-render**: The state update triggers a re-render of the `AppContent` component. The `StockChart` and `RegressionChart` components receive the new data as props and display the visualizations.
