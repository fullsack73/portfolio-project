import React, { useState, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import './App.css';

// Define available metrics and operators
const METRICS = [
    'P/E',
    'P/B',
    'Debt/Equity',
    'ROE',
    'Price/Sales',
];
const OPERATORS = ['Under', 'Over', 'Avg'];

const StockScreener = () => {
    const { t } = useTranslation();
    const [filters, setFilters] = useState([
        { metric: 'P/E', operator: 'Under', value: '15' },
        { metric: 'P/B', operator: 'Under', value: '2' },
    ]);
    const [tickerGroup, setTickerGroup] = useState('S&P 500');
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [customTickers, setCustomTickers] = useState([]);

    const handleAddFilter = () => {
        setFilters([...filters, { metric: 'ROE', operator: 'Over', value: '15%' }]);
    };

    const handleRemoveFilter = (index) => {
        setFilters(filters.filter((_, i) => i !== index));
    };

    const handleFilterChange = (index, field, value) => {
        const newFilters = filters.map((filter, i) => 
            i === index ? { ...filter, [field]: value } : filter
        );
        setFilters(newFilters);
    };

    const handleFileUpload = (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                const text = event.target.result;
                const tickers = text.split('\n').map(t => t.trim()).filter(t => t);
                setCustomTickers(tickers);
            };
            reader.readAsText(file);
        }
    };

    const handleSearch = useCallback(async () => {
        setLoading(true);
        setError(null);
        setResults([]);

        // Translate frontend state to the format finvizfinance expects
        const apiFilters = {
            'Index': tickerGroup,
        };
        filters.forEach(filter => {
            // Example: { metric: 'P/E', operator: 'Under', value: '15' } -> { 'P/E': 'Under 15' }
            apiFilters[filter.metric] = `${filter.operator} ${filter.value}`;
        });

        try {
            const response = await fetch('http://127.0.0.1:5000/api/stock-screener', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ filters: apiFilters }),
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || 'Network response was not ok');
            }

            const data = await response.json();
            setResults(data);
        } catch (err) {
            setError(err.message || 'Failed to fetch screener results.');
            console.error(err);
        } finally {
            setLoading(false);
        }
    }, [filters, tickerGroup]);

    return (
        <div className="stock-screener-container card">
            <h3 className="card-header">{t('stock_screener')}</h3>
            
            <div className="screener-controls">
                <div className="control-group">
                    <label>{t('ticker_group')}</label>
                    <select className="optimizer-select" value={tickerGroup} onChange={(e) => setTickerGroup(e.target.value)}>
                        <option value="S&P 500">S&P 500</option>
                        <option value="Dow Jones">Dow Jones</option>
                        <option value="CUSTOM">Custom</option>
                    </select>
                    {tickerGroup === 'CUSTOM' && (
                        <input type="file" accept=".csv" onChange={handleFileUpload} />
                    )}
                </div>

                {filters.map((filter, index) => (
                    <div key={index} className="filter-row">
                        <select 
                            value={filter.metric} 
                            onChange={(e) => handleFilterChange(index, 'metric', e.target.value)}
                        >
                            {METRICS.map(m => <option key={m} value={m}>{m}</option>)}
                        </select>
                        <select 
                            value={filter.operator} 
                            onChange={(e) => handleFilterChange(index, 'operator', e.target.value)}
                        >
                            {OPERATORS.map(o => <option key={o} value={o}>{o}</option>)}
                        </select>
                        <input 
                            type="text" 
                            value={filter.value} 
                            onChange={(e) => handleFilterChange(index, 'value', e.target.value)}
                            placeholder="e.g., 15, 10%, etc."
                        />
                        <button className="remove-btn" onClick={() => handleRemoveFilter(index)}>✖</button>
                    </div>
                ))}
                <div className="action-buttons">
                    <button onClick={handleAddFilter}>{t('add_filter')}</button>
                    <button onClick={handleSearch} disabled={loading}>
                        {loading ? t('loading') : t('search')}
                    </button>
                </div>
            </div>

            {error && <div className="error-message">{error}</div>}

            {results.length > 0 && (
                <div className="results-container">
                    <h4>{t('screener_results')}</h4>
                    <table className="allocation-table">
                        <thead>
                            <tr>
                                {Object.keys(results[0]).map(key => <th key={key}>{key}</th>)}
                            </tr>
                        </thead>
                        <tbody>
                            {results.map((row, index) => (
                                <tr key={index}>
                                    {Object.values(row).map((val, i) => <td key={i}>{val}</td>)}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
};

export default StockScreener;
