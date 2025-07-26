import React, { useState, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import './App.css';

function TickerInput({ onTickerChange, onSubmit, initialTicker }) {
  const [ticker, setTicker] = useState(initialTicker);
  const { t } = useTranslation();

  // Debounced update function to prevent excessive API calls
  const debouncedUpdate = useCallback(
    (() => {
      let timeoutId;
      return (newTicker, source = 'unknown') => {
        console.log(`🎯 TickerInput debounced update from: ${source}`, { newTicker });
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
          if (newTicker && newTicker.length >= 1 && onTickerChange) {
            console.log('🎯 TickerInput calling parent onTickerChange:', { newTicker });
            onTickerChange(newTicker);
          }
        }, 300); // 300ms delay for ticker input (reduced since we have API cancellation)
      };
    })(),
    [onTickerChange]
  );

  const handleInputChange = (e) => {
    const newTicker = e.target.value.trim().toUpperCase();
    console.log('🎯 TickerInput: User changed ticker:', { newTicker });
    setTicker(newTicker);
    
    // Auto-update when ticker changes (with validation)
    if (newTicker && newTicker.length >= 1) {
      debouncedUpdate(newTicker, 'user-input');
    }
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
      <div className="ticker-input-group">
        <label htmlFor="ticker">{t('ticker.label')}</label>
        <input
          type="text"
          id="ticker"
          value={ticker}
          onChange={handleInputChange}
          placeholder={t('ticker.placeholder')}
          title="Charts will auto-update as you type"
          maxLength="10"
        />
      </div>
    </div>
  );
}

export default TickerInput; 