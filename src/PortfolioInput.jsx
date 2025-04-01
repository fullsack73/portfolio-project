import React, { useState } from 'react';
import axios from 'axios';
import PortfolioGraph from './PortfolioGraph';
import './App.css';

const PortfolioInput = () => {
    const [tickers, setTickers] = useState('');
    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');
    const [metrics, setMetrics] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleTickersChange = (event) => {
        setTickers(event.target.value);
    };

    const handleStartDateChange = (event) => {
        setStartDate(event.target.value);
    };

    const handleEndDateChange = (event) => {
        setEndDate(event.target.value);
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        setLoading(true);
        setError(null);

        try {
            // Clean up tickers string by removing spaces and extra commas
            const cleanTickers = tickers.replace(/\s+/g, '').replace(/,+/g, ',');
            
            const response = await axios.get('http://127.0.0.1:5000/portfolio-metrics', {
                params: {
                    tickers: cleanTickers,
                    start_date: startDate,
                    end_date: endDate
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

    return (
        <div className="portfolio-analysis">
            <h2>Portfolio Optimization</h2>
            <p className="description">
                Enter your stock tickers and date range to find the optimal portfolio weights 
                that maximize the Sharpe ratio (risk-adjusted return).
            </p>
            <form onSubmit={handleSubmit} className="portfolio-form">
                <div className="form-group">
                    <label htmlFor="tickers">
                        Stock Tickers
                        <span className="hint">(comma-separated, e.g., AAPL,MSFT,GOOGL)</span>
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

                <div className="form-row">
                    <div className="form-group">
                        <label htmlFor="startDate">Start Date</label>
                        <input
                            type="date"
                            id="startDate"
                            value={startDate}
                            onChange={handleStartDateChange}
                            required
                        />
                    </div>

                    <div className="form-group">
                        <label htmlFor="endDate">End Date</label>
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
                    {loading ? 'Optimizing Portfolio...' : 'Find Optimal Portfolio'}
                </button>
            </form>

            {error && <div className="error-message">{error}</div>}

            {metrics && (
                <div className="metrics-container">
                    <h3>Optimal Portfolio Results</h3>
                    <div className="metrics-grid">
                        <div className="metric-card">
                            <h4>Optimal Weights</h4>
                            <p>{metrics.final_weights.map((w, i) => `${(w * 100).toFixed(1)}%`).join(', ')}</p>
                        </div>
                        <div className="metric-card">
                            <h4>Expected Return</h4>
                            <p className={metrics.final_return >= 0 ? 'positive' : 'negative'}>
                                {(metrics.final_return * 100).toFixed(2)}%
                            </p>
                        </div>
                        <div className="metric-card">
                            <h4>Portfolio Risk</h4>
                            <p>{(metrics.final_volatility * 100).toFixed(2)}%</p>
                        </div>
                        <div className="metric-card">
                            <h4>Sharpe Ratio</h4>
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
