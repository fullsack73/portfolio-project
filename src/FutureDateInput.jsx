import React, { useState, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const FutureDateInput = ({ onFutureDaysChange, initialDays = 30 }) => {
  const [days, setDays] = useState(initialDays);
  const { t } = useTranslation();

  // Debounced update function to prevent excessive API calls
  const debouncedUpdate = useCallback(
    (() => {
      let timeoutId;
      return (newDays, source = 'unknown') => {
        console.log(`🔮 FutureDateInput debounced update from: ${source}`, { newDays });
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
          if (newDays && onFutureDaysChange) {
            console.log('🔮 FutureDateInput calling parent onFutureDaysChange:', { newDays: parseInt(newDays) });
            onFutureDaysChange(parseInt(newDays));
          }
        }, 500); // 500ms delay
      };
    })(),
    [onFutureDaysChange]
  );

  const handleChange = (e) => {
    const newDays = e.target.value;
    console.log('🔮 FutureDateInput: User changed days:', { newDays });
    setDays(newDays);
    
    // Auto-update when value changes
    if (newDays && parseInt(newDays) > 0) {
      debouncedUpdate(newDays, 'user-input');
    }
  };

  return (
    <div className="date-input-container">
      <div className="date-input-group">
        <label htmlFor="future-days">{t('future.days_to_predict', 'Days to Predict')}</label>
        <input
          type="number"
          id="future-days"
          value={days}
          onChange={handleChange}
          min="1"
          max="365"
          title="Predictions will auto-update when you change this value"
        />
      </div>

    </div>
  );
};

export default FutureDateInput;
