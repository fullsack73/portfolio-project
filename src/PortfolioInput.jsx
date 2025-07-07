import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import axios from 'axios';
import PortfolioGraph from './PortfolioGraph';
import './App.css';

const PortfolioInput = () => {
    const [tickers, setTickers] = useState('');
    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');
    const [risklessRate, setRisklessRate] = useState('');
    const [metrics, setMetrics] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleTickersChange = (e) => {
        // convert to uppercase for lazy asses
        setTickers(e.target.value.toUpperCase());
    };

    const handleStartDateChange = (event) => {
        setStartDate(event.target.value);
    };

    const handleEndDateChange = (event) => {
        setEndDate(event.target.value);
    };

    const handleRisklessRateChange = (event) => {
        setRisklessRate(event.target.value);
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        setLoading(true);
        setError(null);

        try {
            // clean up tickers string by removing spaces and extra commas
            const cleanTickers = tickers.replace(/\s+/g, '').replace(/,+/g, ',');
            
            const response = await axios.get('/api/portfolio-metrics', {
                params: {
                    tickers: cleanTickers,
                    start_date: startDate,
                    end_date: endDate,
                    riskless_rate: parseFloat(risklessRate) / 100
                }
            });
            setMetrics(response.data);
        } catch (error) {
            console.error('Error fetching portfolio metrics:', error);
            setError('Failed to calculate portfolio metrics. Please check your inputs.');
        } finally {
            setLoading(false);
        }
    };

    const { t } = useTranslation();

    return (
        <div className="portfolio-analysis">
            <h2>{t('portfolio.optimization.title')}</h2>
            <p className="description">
                {t('portfolio.optimization.description')}
            </p>
            <form onSubmit={handleSubmit} className="portfolio-form">
                <div className="form-group">
                    <label htmlFor="tickers">
                        {t('portfolio.optimization.title')}
                        <span className="hint">{t('portfolio.optimization.description')}</span>
                    </label>
                    <input
                        type="text"
                        id="tickers"
                        value={tickers}
                        onChange={handleTickersChange}
                        placeholder="Enter stock tickers"
                        required
                    />
                </div>

                <div className="form-group">
                    <label htmlFor="risklessRate">
                        {t('portfolio.risklessRate')} (%)
                    </label>
                    <input
                        type="number"
                        id="risklessRate"
                        value={risklessRate}
                        onChange={handleRisklessRateChange}
                        placeholder="e.g., 2.5"
                        step="any"
                        min="0"
                        required
                    />
                </div>

                <div className="form-row">
                    <div className="form-group">
                        <label htmlFor="startDate">{t('date.start')}</label>
                        <input
                            type="date"
                            id="startDate"
                            value={startDate}
                            onChange={handleStartDateChange}
                            required
                        />
                    </div>

                    <div className="form-group">
                        <label htmlFor="endDate">{t('date.end')}</label>
                        <input
                            type="date"
                            id="endDate"
                            value={endDate}
                            onChange={handleEndDateChange}
                            required
                        />
                    </div>
                </div>

                <button type="submit" disabled={loading} className="submit-button">
                    {loading ? t('common.loading') : t('common.submit')}
                </button>
            </form>

            {error && <div className="error-message">{error}</div>}

            {metrics && (
                <div className="metrics-container">
                    <h3>{t('portfolio.results')}</h3>
                    <div className="metrics-grid">
                        <div className="metric-card">
                            <h4>{t('portfolio.weights')}</h4>
                            <p>{Object.entries(metrics.final_weights).map(([ticker, weight]) => `${ticker}: ${weight}`).join(', ')}</p>
                        </div>
                        <div className="metric-card">
                            <h4>{t('portfolio.expectedReturn')}</h4>
                            <p className={metrics.final_return >= 0 ? 'positive' : 'negative'}>
                                {(metrics.final_return * 100).toFixed(2)}%
                            </p>
                        </div>
                        <div className="metric-card">
                            <h4>{t('portfolio.portfolioRisk')}</h4>
                            <p>{(metrics.final_volatility * 100).toFixed(2)}%</p>
                        </div>
                        <div className="metric-card">
                            <h4>{t('portfolio.sharpeRatio')}</h4>
                            <p>{metrics.final_sharpe_ratio.toFixed(2)}</p>
                        </div>
                    </div>
                    <div className="graph-container">
                        <PortfolioGraph data={metrics.data} />
                    </div>
                </div>
            )}
        </div>
    );
};

export default PortfolioInput;
