import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';

const FinancialStatement = () => {
    const { t } = useTranslation();
    const [data, setData] = useState(null);
    const [ticker, setTicker] = useState('AAPL');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

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
        <div className="metric-card">
            <h4>
                {title}
                <div className="group relative ml-2 inline-block">
                    <span className="cursor-pointer">ⓘ</span>
                    <div className="absolute bottom-full mb-2 w-64 bg-gray-700 text-white text-xs rounded py-1 px-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                        {tooltip}
                    </div>
                </div>
            </h4>
            <p>{value}</p>
        </div>
    );

    return (
        <div className="financial-analysis">
            <h1>{t('financial.title')}</h1>
            <div className="centered-form-container">
                <div className="form-group">
                    <label htmlFor="ticker-input">{t('ticker.label')}</label>
                    <input 
                        id="ticker-input"
                        type="text" 
                        value={ticker} 
                        onChange={(e) => setTicker(e.target.value.toUpperCase())} 
                        placeholder={t('ticker.placeholder')}
                    />
                </div>
                <button onClick={fetchData} className="submit-button" disabled={loading}>
                    {loading ? t('common.loading') : t('financial.fetch_data')}
                </button>
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
        </div>
    );
};

export default FinancialStatement;
