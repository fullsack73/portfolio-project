import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import './App.css';

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

  const { t } = useTranslation();

  return (
    <div className="ticker-input-container">
      <form onSubmit={handleSubmit}>
        <div className="ticker-input-group">
          <label htmlFor="ticker">{t('ticker.label')}</label>
          <input
            type="text"
            id="ticker"
            value={ticker}
            onChange={handleInputChange}
            placeholder={t('ticker.placeholder')}
            required
          />
        </div>
        <button type="submit">{t('ticker.submit')}</button>
      </form>
    </div>
  );
}

export default TickerInput; 