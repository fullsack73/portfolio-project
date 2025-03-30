import React, { useState, useEffect } from 'react';
import './App.css';

const HedgeAnalysis = () => {
    const [ticker1, setTicker1] = useState('');
    const [ticker2, setTicker2] = useState('');
    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');
    const [hedgeData, setHedgeData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const fetchHedgeData = async () => {
        if (!ticker1 || !ticker2) {
            setError('Please enter both tickers');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const url = new URL('http://127.0.0.1:5000/analyze-hedge');
            url.searchParams.append('ticker1', ticker1.toUpperCase());
            url.searchParams.append('ticker2', ticker2.toUpperCase());
            if (startDate) url.searchParams.append('start_date', startDate);
            if (endDate) url.searchParams.append('end_date', endDate);

            const response = await fetch(url);
            const data = await response.json();

            if (data.error) {
                setError(data.error);
                return;
            }

            setHedgeData(data);
        } catch (err) {
            setError('Failed to fetch hedge analysis data');
            console.error('Error fetching hedge data:', err);
        } finally {
            setLoading(false);
        }
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        fetchHedgeData();
    };

    return (
        <div className="hedge-analysis">
            <h2>Hedge Relationship Analysis</h2>
            
            <form onSubmit={handleSubmit} className="hedge-form">
                <div className="form-group">
                    <label htmlFor="ticker1">First Ticker:</label>
                    <input
                        type="text"
                        id="ticker1"
                        value={ticker1}
                        onChange={(e) => setTicker1(e.target.value.toUpperCase())}
                        placeholder="e.g., AAPL"
                    />
                </div>

                <div className="form-group">
                    <label htmlFor="ticker2">Second Ticker:</label>
                    <input
                        type="text"
                        id="ticker2"
                        value={ticker2}
                        onChange={(e) => setTicker2(e.target.value.toUpperCase())}
                        placeholder="e.g., MSFT"
                    />
                </div>

                <div className="form-group">
                    <label htmlFor="startDate">Start Date:</label>
                    <input
                        type="date"
                        id="startDate"
                        value={startDate}
                        onChange={(e) => setStartDate(e.target.value)}
                    />
                </div>

                <div className="form-group">
                    <label htmlFor="endDate">End Date:</label>
                    <input
                        type="date"
                        id="endDate"
                        value={endDate}
                        onChange={(e) => setEndDate(e.target.value)}
                    />
                </div>

                <button type="submit" disabled={loading}>
                    {loading ? 'Analyzing...' : 'Analyze Hedge Relationship'}
                </button>
            </form>

            {error && <div className="error-message">{error}</div>}

            {hedgeData && (
                <div className="hedge-results">
                    <h3>Analysis Results</h3>
                    
                    <div className="result-card">
                        <h4>Companies</h4>
                        <p>{hedgeData.company1} ({hedgeData.ticker1})</p>
                        <p>{hedgeData.company2} ({hedgeData.ticker2})</p>
                    </div>

                    <div className="result-card">
                        <h4>Hedge Relationship</h4>
                        <p className={hedgeData.is_hedge ? 'hedge-positive' : 'hedge-negative'}>
                            {hedgeData.is_hedge ? 'Yes' : 'No'}
                        </p>
                        <p>Strength: {hedgeData.strength}</p>
                    </div>

                    <div className="result-card">
                        <h4>Statistical Analysis</h4>
                        <p>Correlation: {hedgeData.correlation.toFixed(3)}</p>
                        <p>P-value: {hedgeData.p_value.toFixed(4)}</p>
                    </div>

                    <div className="result-card">
                        <h4>Analysis Period</h4>
                        <p>Start: {hedgeData.period.start}</p>
                        <p>End: {hedgeData.period.end}</p>
                    </div>
                </div>
            )}
        </div>
    );
};

export default HedgeAnalysis;
