import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import axios from 'axios';

const Optimizer = () => {
    const { t } = useTranslation();
    const [tickerGroup, setTickerGroup] = useState('SP500');
    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');
    const [riskFreeRate, setRiskFreeRate] = useState('2');
    const [targetReturn, setTargetReturn] = useState('');
    const [riskTolerance, setRiskTolerance] = useState('');
    const [optimizedPortfolio, setOptimizedPortfolio] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [investmentAmount, setInvestmentAmount] = useState('');
    const [allocation, setAllocation] = useState(null);

    const handleAllocation = () => {
        if (!investmentAmount || !optimizedPortfolio || !optimizedPortfolio.weights) return;

        const totalInvestment = parseFloat(investmentAmount);
        const allocated = Object.entries(optimizedPortfolio.weights).map(([ticker, weight]) => {
            const amount = totalInvestment * weight;
            // Assuming we can get price data, for now we'll use a placeholder
            // In a real scenario, you'd fetch current prices for each ticker
            const price = optimizedPortfolio.prices[ticker] || 1; // Placeholder price
            const shares = amount / price;
            return { ticker, amount, shares };
        });
        setAllocation(allocated);
    };

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
                risk_free_rate: parseFloat(riskFreeRate) / 100,
                target_return: targetReturn ? parseFloat(targetReturn) / 100 : null,
                risk_tolerance: riskTolerance ? parseFloat(riskTolerance) / 100 : null,
            });
            setOptimizedPortfolio(response.data);
        } catch (err) {
            setError(err.response ? err.response.data.error : 'An error occurred');
        }
        setLoading(false);
    };

    return (
        <div className="optimizer-container">
            <h2>{t('optimizer.title')}</h2>
            <form onSubmit={handleSubmit} className="optimizer-form">
                <div className="optimizer-form-grid">
                    <div className="optimizer-form-group">
                        <label>{t('optimizer.tickerGroup')}</label>
                        <select className="optimizer-select" value={tickerGroup} onChange={(e) => setTickerGroup(e.target.value)} required>
                            <option value="SP500">S&P 500</option>
                            {/* Add other ticker groups here */}
                        </select>
                    </div>
                    <div className="optimizer-form-group">
                        <label>{t('optimizer.startDate')}</label>
                        <input className="optimizer-input" type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} required />
                    </div>
                    <div className="optimizer-form-group">
                        <label>{t('optimizer.endDate')}</label>
                        <input className="optimizer-input" type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} required />
                    </div>
                                        <div className="optimizer-form-group">
                        <label>{t('optimizer.riskFreeRate')}</label>
                        <div class="input-with-symbol">
                            <input className="optimizer-input" type="number" value={riskFreeRate} onChange={(e) => setRiskFreeRate(e.target.value)} placeholder="e.g., 2" required />
                        </div>
                    </div>
                    <div className="optimizer-form-group">
                        <label>{t('optimizer.targetReturn')}</label>
                        <div class="input-with-symbol">
                            <input className="optimizer-input" type="number" value={targetReturn} onChange={(e) => setTargetReturn(e.target.value)} placeholder="e.g., 20" />
                        </div>
                    </div>
                    <div className="optimizer-form-group">
                        <label>{t('optimizer.riskTolerance')}</label>
                        <div class="input-with-symbol">
                            <input className="optimizer-input" type="number" value={riskTolerance} onChange={(e) => setRiskTolerance(e.target.value)} placeholder="e.g., 15" />
                        </div>
                    </div>
                    <button type="submit" className="optimizer-submit-button" disabled={loading}>
                        {loading ? t('common.loading') : t('optimizer.submit')}
                    </button>
                </div>
            </form>

            {error && <div className="optimizer-error">{error}</div>}

            {optimizedPortfolio && (
                <>
                    <div className="optimizer-results-container">
                        <h3>{t('optimizer.results')}</h3>
                        <div className="optimizer-results-grid">
                            <div className="optimizer-result-card">
                                <h4>{t('optimizer.return')}</h4>
                                <p>{(optimizedPortfolio.return * 100).toFixed(2)}%</p>
                            </div>
                            <div className="optimizer-result-card">
                                <h4>{t('optimizer.risk')}</h4>
                                <p>{(optimizedPortfolio.risk * 100).toFixed(2)}%</p>
                            </div>
                            <div className="optimizer-result-card">
                                <h4>{t('optimizer.sharpeRatio')}</h4>
                                <p>{optimizedPortfolio.sharpe_ratio.toFixed(2)}</p>
                            </div>
                        </div>
                        <div className="optimizer-weights-card">
                            <h4>{t('optimizer.weights')}</h4>
                            <ul className="optimizer-weights-list">
                                {Object.entries(optimizedPortfolio.weights).map(([ticker, weight]) => (
                                    <li key={ticker}>
                                        <span>{ticker}</span>
                                        <strong>{(weight * 100).toFixed(2)}%</strong>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </div>

                    <div className="investment-allocation-container">
                        <h3>{t('optimizer.investmentAllocation')}</h3>
                        <div className="investment-allocation-form">
                            <div className="optimizer-form-group">
                                <label>{t('optimizer.investmentBudget')}</label>
                                <input 
                                    className="optimizer-input"
                                    type="number" 
                                    value={investmentAmount} 
                                    onChange={(e) => setInvestmentAmount(e.target.value)} 
                                    placeholder={t('optimizer.enterBudget')} 
                                />
                            </div>
                            <button onClick={handleAllocation} className="optimizer-submit-button">
                                {t('optimizer.calculate')}
                            </button>
                        </div>

                        {allocation && (
                            <div className="allocation-results-container">
                                <h4>{t('optimizer.allocationResults')}</h4>
                                <table className="allocation-table">
                                    <thead>
                                        <tr>
                                            <th>{t('optimizer.ticker')}</th>
                                            <th>{t('optimizer.investmentAmount')}</th>
                                            <th>{t('optimizer.shares')}</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {allocation.map(({ ticker, amount, shares }) => (
                                            <tr key={ticker}>
                                                <td>{ticker}</td>
                                                <td>${amount.toFixed(2)}</td>
                                                <td>{shares.toFixed(4)}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        )}
                    </div>
                </>
            )}
                
        </div>
    );
};

export default Optimizer;
