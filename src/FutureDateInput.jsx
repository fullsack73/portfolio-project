import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';

const FutureDateInput = ({ onFutureDaysChange, initialDays = 30 }) => {
  const [days, setDays] = useState(initialDays);
  const { t } = useTranslation();

  const handleChange = (e) => {
    setDays(e.target.value);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onFutureDaysChange(days);
  };

  return (
    <form onSubmit={handleSubmit} className="date-input-container">
      <label htmlFor="future-days">{t('future.days_to_predict', 'Days to Predict')}</label>
      <input
        type="number"
        id="future-days"
        value={days}
        onChange={handleChange}
        min="1"
      />
      <button type="submit">{t('future.update_prediction', 'Update Prediction')}</button>
    </form>
  );
};

export default FutureDateInput;
