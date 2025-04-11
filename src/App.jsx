import React, { useState, useEffect } from 'react';
import StockChart from "./StockChart.jsx";
import DateInput from "./DateInput.jsx";
import TickerInput from "./TickerInput.jsx";
import RegressionChart from "./RegressionChart.jsx";
import Selector from './Selector.jsx';
import HedgeAnalysis from './Hedge.jsx';
import PortfolioInput from './PortfolioInput.jsx';
import './App.css';

function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [ticker, setTicker] = useState('AAPL');
  const [companyName, setCompanyName] = useState('Apple Inc.');
  const [showChart, setShowChart] = useState(false);
  const [regressionData, setRegressionData] = useState(null);
  const [formula, setFormula] = useState('');
  const [activeView, setActiveView] = useState('stock');
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const fetchData = (startDate = null, endDate = null, stockTicker = ticker) => {
    setLoading(true);
    // if this fails despite having right proxy settings, i'm fucked. but it should never fail
    let url = `/api/get-data?ticker=${stockTicker}&regression=true`;
    if (startDate && endDate) {
      url += `&start_date=${startDate}&end_date=${endDate}`;
    }

    console.log('Attempting to fetch data...');
    fetch(url, {
      method: 'GET',
      mode: 'cors',
      credentials: 'include',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
      }
    })
      .then((response) => {
        console.log('Response status:', response.status);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then((responseData) => {
        console.log('Raw data received:', responseData);
        setData(responseData.prices);
        setRegressionData(responseData.regression);
        setCompanyName(responseData.companyName);
        setFormula(responseData.formula);
        setLoading(false);
        setShowChart(true);
      })
      .catch((error) => {
        console.error('Fetch error:', error);
        setError(error.message);
        setLoading(false);
      });
  };

  // Initial data fetch
  useEffect(() => {
    fetchData();
  }, []);

  const handleDateRangeChange = (startDate, endDate) => {
    if (ticker) {
      fetchData(startDate, endDate);
    }
  };

  const handleTickerChange = (newTicker) => {
    setTicker(newTicker);
    fetchData(null, null, newTicker);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (ticker) {
      fetchData();
    }
  };

  return (
    <div className="app-container">
      <Selector 
        activeView={activeView} 
        onViewChange={setActiveView} 
        isOpen={isSidebarOpen}
        onToggle={() => setIsSidebarOpen(!isSidebarOpen)}
      />
      <main className="main-content">
        {activeView === 'stock' ? (
          <>
            <h1>Stock Data Visualization</h1>
            <div className="controls-container">
              <TickerInput onTickerChange={handleTickerChange} onSubmit={handleSubmit} initialTicker="AAPL" />
              <DateInput onDateRangeChange={handleDateRangeChange} />
            </div>

            {loading && <p className="loading">Loading...</p>}
            {error && <p className="error">Error: {error}</p>}

            {showChart && data && (
              <>
                <h2>{companyName} ({ticker})</h2>
                <div className="charts-container">
                  <div className="chart-wrapper">
                    <StockChart data={data} ticker={ticker} />
                  </div>
                  <div className="chart-wrapper">
                    <RegressionChart 
                      data={data} 
                      regression={regressionData} 
                      ticker={ticker}
                      formula={formula}
                    />
                  </div>
                </div>
              </>
            )}
          </>
        ) : activeView === 'hedge' ? (
          <HedgeAnalysis />
        ) : (
          <PortfolioInput />
        )}
      </main>
    </div>
  );
}

export default App;