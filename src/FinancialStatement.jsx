import React, { useState, useEffect } from 'react';

const FinancialStatement = () => {
    const [data, setData] = useState(null);
    const [ticker, setTicker] = useState('AAPL');
    const [loading, setLoading] = useState(false);

    const fetchData = async () => {
        setLoading(true);
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
            setData({ error: error.message });
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="p-4">
            <h2 className="text-2xl font-bold mb-4">Financial Statement Analysis</h2>
            <div className="flex mb-4">
                <input 
                    type="text" 
                    value={ticker} 
                    onChange={(e) => setTicker(e.target.value)} 
                    className="border p-2 mr-2"
                    placeholder="Enter Ticker Symbol"
                />
                <button onClick={fetchData} className="bg-blue-500 text-white p-2 rounded">Fetch Data</button>
            </div>
            {loading && <p>Loading...</p>}
            {data && data.error && <p className="text-red-500">{data.error}</p>}
            {data && !data.error && (
                <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 border rounded">
                        <h3 className="font-bold">Company</h3>
                        <p>{data.longName} ({data.ticker})</p>
                    </div>
                    <div className="p-4 border rounded">
                        <div className="flex items-center">
                            <h3 className="font-bold">Price to Earnings (P/E)</h3>
                            <div className="group relative ml-2">
                                <span className="cursor-pointer">ⓘ</span>
                                <div className="absolute bottom-full mb-2 w-64 bg-gray-700 text-white text-xs rounded py-1 px-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                                    Measures the company's current share price relative to its per-share earnings. A high P/E could mean the stock is overvalued, or that investors are expecting high growth rates.
                                </div>
                            </div>
                        </div>
                        <p>{data.per}</p>
                    </div>
                    <div className="p-4 border rounded">
                        <div className="flex items-center">
                            <h3 className="font-bold">Price to Book (P/B)</h3>
                            <div className="group relative ml-2">
                                <span className="cursor-pointer">ⓘ</span>
                                <div className="absolute bottom-full mb-2 w-64 bg-gray-700 text-white text-xs rounded py-1 px-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                                    Compares a company's market capitalization to its book value. A lower P/B ratio could mean the stock is undervalued.
                                </div>
                            </div>
                        </div>
                        <p>{data.pbr}</p>
                    </div>
                    <div className="p-4 border rounded">
                        <div className="flex items-center">
                            <h3 className="font-bold">Price to Sales (P/S)</h3>
                            <div className="group relative ml-2">
                                <span className="cursor-pointer">ⓘ</span>
                                <div className="absolute bottom-full mb-2 w-64 bg-gray-700 text-white text-xs rounded py-1 px-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                                    Compares a company's stock price to its revenues. It is often used for growth stocks that have yet to achieve profitability.
                                </div>
                            </div>
                        </div>
                        <p>{data.psr}</p>
                    </div>
                    <div className="p-4 border rounded">
                        <div className="flex items-center">
                            <h3 className="font-bold">Debt Ratio</h3>
                            <div className="group relative ml-2">
                                <span className="cursor-pointer">ⓘ</span>
                                <div className="absolute bottom-full mb-2 w-64 bg-gray-700 text-white text-xs rounded py-1 px-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                                    Measures the extent of a company’s leverage. A debt ratio greater than 1.0 means the company has more debt than assets.
                                </div>
                            </div>
                        </div>
                        <p>{data.debt_ratio}</p>
                    </div>
                    <div className="p-4 border rounded">
                        <div className="flex items-center">
                            <h3 className="font-bold">Liquidity Ratio (Current Ratio)</h3>
                            <div className="group relative ml-2">
                                <span className="cursor-pointer">ⓘ</span>
                                <div className="absolute bottom-full mb-2 w-64 bg-gray-700 text-white text-xs rounded py-1 px-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                                    Measures a company's ability to pay short-term obligations. A ratio under 1 indicates that the company has more liabilities than assets.
                                </div>
                            </div>
                        </div>
                        <p>{data.liquidity_ratio}</p>
                    </div>
                </div>
            )}
        </div>
    );
};

export default FinancialStatement;
