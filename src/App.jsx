import React, { useState, useEffect } from 'react';
import { I18nextProvider, useTranslation } from 'react-i18next';
import i18n from './i18n';
import StockChart from "./StockChart.jsx";
import DateInput from "./DateInput.jsx";
import TickerInput from "./TickerInput.jsx";
import RegressionChart from "./RegressionChart.jsx";
import Selector from './Selector.jsx';
import HedgeAnalysis from './Hedge.jsx';
import PortfolioInput from './PortfolioInput.jsx';
import LanguageSelector from './LanguageSelector.jsx';
import FutureDateInput from './FutureDateInput.jsx';
import FutureChart from './FutureChart.jsx';
import FinancialStatement from './FinancialStatement.jsx';
import Optimizer from './Optimizer.jsx';
import './App.css';

function AppContent() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [ticker, setTicker] = useState('AAPL');
  const [companyName, setCompanyName] = useState('Apple Inc.');
  const [showChart, setShowChart] = useState(false);
  const [regressionData, setRegressionData] = useState(null);
  const [formula, setFormula] = useState('');
  const [appStartDate, setAppStartDate] = useState(null);
  const [appEndDate, setAppEndDate] = useState(null);
  const [futureDays, setFutureDays] = useState(30);
  const [futurePredictions, setFuturePredictions] = useState(null);
  const [activeView, setActiveView] = useState('stock');
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState('en');
  const { t } = useTranslation();

  const fetchData = (startDate = null, endDate = null, stockTicker = ticker) => {
    setLoading(true);
    let url = `/api/get-data?ticker=${stockTicker}&regression=true&future_days=${futureDays}`;
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
        setFuturePredictions(responseData.future_predictions);
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

  useEffect(() => {
    // Initialize appStartDate and appEndDate
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    const threeMonthsAgo = new Date(yesterday);
    threeMonthsAgo.setMonth(threeMonthsAgo.getMonth() - 3);

    const formatDate = (date) => date.toISOString().split('T')[0];

    setAppStartDate(formatDate(threeMonthsAgo));
    setAppEndDate(formatDate(yesterday));
  }, []);

  // Initial data fetch, now dependent on appStartDate and appEndDate
  useEffect(() => {
    if (appStartDate && appEndDate && ticker) {
      fetchData(appStartDate, appEndDate, ticker);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [appStartDate, appEndDate, ticker]); // Fetch when app dates are initialized or ticker changes initially.

  const handleDateRangeChange = (newStartDate, newEndDate) => {
    setAppStartDate(newStartDate);
    setAppEndDate(newEndDate);
    if (ticker) {
      fetchData(newStartDate, newEndDate, ticker);
    }
  };

  const handleTickerChange = (newTicker) => {
    setTicker(newTicker);
    // Use appStartDate and appEndDate if available, otherwise backend defaults (null, null)
    fetchData(appStartDate, appEndDate, newTicker);
  };

  const handleFutureDaysChange = (days) => {
    setFutureDays(days);
    // Use appStartDate and appEndDate if available, otherwise backend defaults (null, null)
    fetchData(appStartDate, appEndDate, ticker);
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
      <LanguageSelector 
        isOpen={isSidebarOpen}
        selectedLanguage={selectedLanguage} 
        onLanguageChange={setSelectedLanguage}
      />
      <main className="main-content">
        {activeView === 'stock' ? (
          <>
            <h1>{t('regression.title')}</h1>
            <div className="controls-container">
              <TickerInput onTickerChange={handleTickerChange} onSubmit={handleSubmit} initialTicker="AAPL" />
              <DateInput onDateRangeChange={handleDateRangeChange} />
              <FutureDateInput onFutureDaysChange={handleFutureDaysChange} initialDays={futureDays} />
            </div>

            {loading && <p className="loading">{t('common.loading')}</p>}
            {error && <p className="error">{t('common.error')}: {error}</p>}


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
                {futurePredictions && (
                  <div className="charts-container">
                    <div className="chart-wrapper">
                      <FutureChart data={futurePredictions} ticker={ticker} />
                    </div>
                  </div>
                )}
              </>
            )}
          </>
        ) : activeView === 'hedge' ? (
          <HedgeAnalysis />
        ) : activeView === 'financial' ? (
          <FinancialStatement />
        ) : activeView === 'optimizer' ? (
          <Optimizer />
        ) : (
          <PortfolioInput />
        )}
      </main>
    </div>
  );
}

export default function App() {
    return (
        <I18nextProvider i18n={i18n}>
            <AppContent />
        </I18nextProvider>
    );
}