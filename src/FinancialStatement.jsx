import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import StockScreener from './StockScreener'; // Import the new component

const FinancialStatement = () => {
    const { t } = useTranslation();
    const [data, setData] = useState(null);
    const [ticker, setTicker] = useState('AAPL');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [hoveredMetric, setHoveredMetric] = useState(null);

    const fetchData = async () => {
        setLoading(true);
        setError(null);
        setData(null);
        try {
            const response = await fetch(`/api/financial-statement?ticker=${ticker}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const result = await response.json();
            if (result.error) {
                throw new Error(result.error);
            }
            setData(result);
        } catch (error) {
            console.error("Fetch error:", error);
            setError(error.message);
        } finally {
            setLoading(false);
        }
    };

    const renderMetricCard = (title, value, tooltip) => (
        <div 
            className="metric-card"
            onMouseEnter={() => setHoveredMetric(title)}
            onMouseLeave={() => setHoveredMetric(null)}
        >
            <h4>{title}</h4>
            <p>{value}</p>
            <div className={`metric-tooltip ${hoveredMetric === title ? 'visible' : ''}`}>
                {tooltip}
            </div>
        </div>
    );

    return (
        <div className="financial-analysis">
            <h1>{t('financial.title')}</h1>
            <div className="input-form-wrapper">
                <label htmlFor="ticker-input">{t('ticker.label')}</label>
                <div className="input-form-container">
                    <input 
                        id="ticker-input"
                        type="text" 
                        value={ticker} 
                        onChange={(e) => setTicker(e.target.value.toUpperCase())} 
                        placeholder={t('ticker.placeholder')}
                    />
                    <button onClick={fetchData} className="submit-button" disabled={loading}>
                        {loading ? t('common.loading') : t('financial.fetch_data')}
                    </button>
                </div>
            </div>

            {error && <p className="error-message">{t('common.error')}: {error}</p>}

            {data && (
                <div className="metrics-container">
                    <div className="text-center mb-4">
                        <h2 className="text-2xl font-bold">{data.longName} ({data.ticker})</h2>
                    </div>
                    <div className="metrics-grid">
                        {renderMetricCard(t('financial.per'), data.per, t('financial.per_tooltip'))}
                        {renderMetricCard(t('financial.pbr'), data.pbr, t('financial.pbr_tooltip'))}
                        {renderMetricCard(t('financial.psr'), data.psr, t('financial.psr_tooltip'))}
                        {renderMetricCard(t('financial.debt_ratio'), data.debt_ratio, t('financial.debt_ratio_tooltip'))}
                        {renderMetricCard(t('financial.liquidity_ratio'), data.liquidity_ratio, t('financial.liquidity_ratio_tooltip'))}
                    </div>
                </div>
            )}

            {/* Add the Stock Screener component below the existing content */}
            <StockScreener />
        </div>
    );
};

export default FinancialStatement;
