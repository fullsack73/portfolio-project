import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';

const FutureDateInput = ({ onFutureDaysChange, initialDays = 30 }) => {
  const [days, setDays] = useState(initialDays);
  const { t } = useTranslation();

  const handleChange = (e) => {
    setDays(e.target.value);
  };

  const handleClick = () => {
    onFutureDaysChange(days);
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
        />
      </div>
      <button onClick={handleClick}>{t('future.update_prediction', 'Update Prediction')}</button>
    </div>
  );
};

export default FutureDateInput;
