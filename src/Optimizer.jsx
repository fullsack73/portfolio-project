import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import axios from 'axios';

const Optimizer = () => {
    const { t } = useTranslation();
    const [tickerGroup, setTickerGroup] = useState('SP500');
    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');
    const [riskFreeRate, setRiskFreeRate] = useState('');
    const [targetReturn, setTargetReturn] = useState('');
    const [riskTolerance, setRiskTolerance] = useState('');
    const [optimizedPortfolio, setOptimizedPortfolio] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setOptimizedPortfolio(null);

        try {
                        const response = await axios.post('http://127.0.0.1:5000/api/optimize-portfolio', {
                ticker_group: tickerGroup,
                start_date: startDate,
                end_date: endDate,
                risk_free_rate: parseFloat(riskFreeRate),
                target_return: targetReturn ? parseFloat(targetReturn) : null,
                risk_tolerance: riskTolerance ? parseFloat(riskTolerance) : null,
            });
            setOptimizedPortfolio(response.data);
        } catch (err) {
            setError(err.response ? err.response.data.error : 'An error occurred');
        }
        setLoading(false);
    };

    return (
        <div>
            <h2>{t('optimizer.title')}</h2>
            <form onSubmit={handleSubmit}>
                <div>
                    <label>{t('optimizer.tickerGroup')}</label>
                    <select value={tickerGroup} onChange={(e) => setTickerGroup(e.target.value)} required>
                        <option value="SP500">S&P 500</option>
                        {/* Add other ticker groups here in the future */}
                    </select>
                </div>
                <div>
                    <label>{t('optimizer.startDate')}</label>
                    <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} required />
                </div>
                <div>
                    <label>{t('optimizer.endDate')}</label>
                    <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} required />
                </div>
                <div>
                    <label>{t('optimizer.riskFreeRate')}</label>
                    <input type="number" value={riskFreeRate} onChange={(e) => setRiskFreeRate(e.target.value)} placeholder="0.02" required />
                </div>
                <div>
                    <label>{t('optimizer.targetReturn')}</label>
                    <input type="number" value={targetReturn} onChange={(e) => setTargetReturn(e.target.value)} placeholder="0.20" />
                </div>
                <div>
                    <label>{t('optimizer.riskTolerance')}</label>
                    <input type="number" value={riskTolerance} onChange={(e) => setRiskTolerance(e.target.value)} placeholder="0.15" />
                </div>
                <button type="submit" disabled={loading}>{loading ? t('common.loading') : t('optimizer.submit')}</button>
            </form>
            {error && <div style={{ color: 'red' }}>{error}</div>}
            {optimizedPortfolio && (
                <div>
                    <h3>{t('optimizer.results')}</h3>
                    <p>{t('optimizer.weights')}:</p>
                    <ul>
                        {Object.entries(optimizedPortfolio.weights).map(([ticker, weight]) => (
                            <li key={ticker}>{ticker}: {(weight * 100).toFixed(2)}%</li>
                        ))}
                    </ul>
                    <p>{t('optimizer.return')}: {(optimizedPortfolio.return * 100).toFixed(2)}%</p>
                    <p>{t('optimizer.risk')}: {(optimizedPortfolio.risk * 100).toFixed(2)}%</p>
                    <p>{t('optimizer.sharpeRatio')}: {optimizedPortfolio.sharpe_ratio.toFixed(2)}</p>
                </div>
            )}
        </div>
    );
};

export default Optimizer;
