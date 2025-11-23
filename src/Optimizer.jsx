import React, { useRef, useState } from 'react';
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
    const [customTickers, setCustomTickers] = useState([]);
    const [portfolioId, setPortfolioId] = useState('');
    const [persistResult, setPersistResult] = useState(false);
    const [loadIfAvailable, setLoadIfAvailable] = useState(false);
    const portfolioFileInputRef = useRef(null);

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

    const handleAllocation = () => {
        if (!investmentAmount || !optimizedPortfolio || !optimizedPortfolio.weights) return;

        const totalInvestment = parseFloat(investmentAmount);
        const allocated = Object.entries(optimizedPortfolio.weights).map(([ticker, weight]) => {
            const amount = totalInvestment * weight;
            // Assuming we can get price data, for now we'll use a placeholder
            // In a real scenario, you'd fetch current prices for each ticker
            const price = optimizedPortfolio.prices?.[ticker] ?? 1; // Placeholder price
            const shares = amount / price;
            return { ticker, amount, shares };
        });
        setAllocation(allocated);
    };

    const handleDownloadPortfolio = () => {
        if (!optimizedPortfolio) return;
        const payload = {
            ...optimizedPortfolio,
            portfolio_id: optimizedPortfolio.portfolio_id || portfolioId || `portfolio_${Date.now()}`
        };
        const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `${payload.portfolio_id}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    };

    const triggerPortfolioUpload = () => {
        portfolioFileInputRef.current?.click();
    };

    const handlePortfolioUpload = (event) => {
        const file = event.target.files?.[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const parsed = JSON.parse(e.target.result);
                if (!parsed || typeof parsed !== 'object' || !parsed.weights) {
                    throw new Error('Invalid portfolio file: missing weights');
                }
                setOptimizedPortfolio(parsed);
                setAllocation(null);
                setError(null);
                if (parsed.portfolio_id) {
                    setPortfolioId(parsed.portfolio_id);
                }
            } catch (uploadError) {
                setError(uploadError.message || 'Failed to load portfolio file');
            }
        };
        reader.onerror = () => setError('Failed to read portfolio file');
        reader.readAsText(file);

        // Reset input value to allow uploading the same file again if needed
        event.target.value = '';
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setOptimizedPortfolio(null);
        setAllocation(null);

        try {
            const payload = {
                start_date: startDate,
                end_date: endDate,
                risk_free_rate: parseFloat(riskFreeRate) / 100,
                target_return: targetReturn ? parseFloat(targetReturn) / 100 : null,
                risk_tolerance: riskTolerance ? parseFloat(riskTolerance) / 100 : null,
                persist_result: persistResult,
                load_if_available: loadIfAvailable,
            };

            if (tickerGroup === 'CUSTOM') {
                payload.tickers = customTickers;
            } else {
                payload.ticker_group = tickerGroup;
            }

            if (portfolioId.trim()) {
                payload.portfolio_id = portfolioId.trim();
            }

            const response = await axios.post('http://127.0.0.1:5000/api/optimize-portfolio', payload);

            if (response.data.error) {
                setError(response.data.error);
                setOptimizedPortfolio(null);
            } else {
                setOptimizedPortfolio(response.data);
                if (!portfolioId && response.data.portfolio_id) {
                    setPortfolioId(response.data.portfolio_id);
                }
            }
        } catch (err) {
            setError(err.response ? err.response.data.error : 'An error occurred');
            setOptimizedPortfolio(null);
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
                            <option value="DOW">Dow Jones</option>
                            <option value="CUSTOM">{t('optimizer.custom')}</option>
                        </select>
                        {tickerGroup === 'CUSTOM' && (
                            <input type="file" accept=".csv" onChange={handleFileUpload} />
                        )}
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
                        <div className="input-with-symbol">
                            <input className="optimizer-input" type="number" value={riskFreeRate} onChange={(e) => setRiskFreeRate(e.target.value)} placeholder="e.g., 2" required />
                        </div>
                    </div>
                    <div className="optimizer-form-group">
                        <label>{t('optimizer.targetReturn')}</label>
                        <div className="input-with-symbol">
                            <input className="optimizer-input" type="number" value={targetReturn} onChange={(e) => setTargetReturn(e.target.value)} placeholder="e.g., 20" />
                        </div>
                    </div>
                    <div className="optimizer-form-group">
                        <label>{t('optimizer.riskTolerance')}</label>
                        <div className="input-with-symbol">
                            <input className="optimizer-input" type="number" value={riskTolerance} onChange={(e) => setRiskTolerance(e.target.value)} placeholder="e.g., 15" />
                        </div>
                    </div>
                    <div className="optimizer-form-group">
                        <label>{t('optimizer.portfolioId', 'Portfolio ID')}</label>
                        <input
                            className="optimizer-input"
                            type="text"
                            value={portfolioId}
                            onChange={(e) => setPortfolioId(e.target.value)}
                            placeholder={t('optimizer.portfolioIdPlaceholder', 'Optional identifier to reuse')}
                        />
                        <div className="optimizer-checkbox-row">
                            <label>
                                <input
                                    type="checkbox"
                                    checked={persistResult}
                                    onChange={(e) => setPersistResult(e.target.checked)}
                                />
                                <span>{t('optimizer.persistResult', 'Save on server')}</span>
                            </label>
                            <label>
                                <input
                                    type="checkbox"
                                    checked={loadIfAvailable}
                                    onChange={(e) => setLoadIfAvailable(e.target.checked)}
                                />
                                <span>{t('optimizer.loadSaved', 'Load saved result')}</span>
                            </label>
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
                        <div className="optimizer-actions-row">
                            <button
                                className="optimizer-secondary-button"
                                onClick={handleDownloadPortfolio}
                                disabled={!optimizedPortfolio}
                                type="button"
                            >
                                {t('optimizer.downloadPortfolio', 'Download JSON')}
                            </button>
                            <button
                                className="optimizer-secondary-button"
                                onClick={triggerPortfolioUpload}
                                type="button"
                            >
                                {t('optimizer.loadPortfolio', 'Load JSON')}
                            </button>
                            <input
                                type="file"
                                accept="application/json"
                                ref={portfolioFileInputRef}
                                style={{ display: 'none' }}
                                onChange={handlePortfolioUpload}
                            />
                        </div>
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
                                        {allocation
                                            .filter(({ ticker }) => optimizedPortfolio.weights[ticker] > 0.0001)
                                            .map(({ ticker, amount, shares }) => (
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
