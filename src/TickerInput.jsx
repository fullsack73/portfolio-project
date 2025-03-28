import React, { useState } from 'react';

function TickerInput({ onTickerChange, onSubmit, initialTicker }) {
  const [ticker, setTicker] = useState(initialTicker);

  const handleInputChange = (e) => {
    const newTicker = e.target.value.trim().toUpperCase();
    setTicker(newTicker);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (ticker.trim()) {
      onTickerChange(ticker.trim().toUpperCase());
      onSubmit(e);
    }
  };

  return (
    <div className="ticker-input-container">
      <form onSubmit={handleSubmit}>
        <div className="ticker-input-group">
          <label htmlFor="ticker">Enter Stock Symbol:</label>
          <input
            type="text"
            id="ticker"
            value={ticker}
            onChange={handleInputChange}
            placeholder="e.g., AAPL, MSFT, GOOGL"
            maxLength="5"
            required
          />
        </div>
        <button type="submit">Show Stock Data</button>
      </form>
    </div>
  );
}

export default TickerInput; 