import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';

function DateInput({ onDateRangeChange }) {
  const { t } = useTranslation();
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');

  useEffect(() => {
    // set default dates (3 months from yesterday)
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    const threeMonthsAgo = new Date(yesterday);
    threeMonthsAgo.setMonth(threeMonthsAgo.getMonth() - 3);

    // format dates for input fields (YYYY-MM-DD)
    const formatDate = (date) => {
      return date.toISOString().split('T')[0];
    };

    setEndDate(formatDate(yesterday));
    setStartDate(formatDate(threeMonthsAgo));
  }, []);

  const handleDateChange = () => {
    if (startDate && endDate) {
      onDateRangeChange(startDate, endDate);
    }
  };

  return (
    <div className="date-input-container">
      <div className="date-input-group">
        <label htmlFor="startDate">{t('date.start')}</label>
        <input
          type="date"
          id="startDate"
          value={startDate}
          onChange={(e) => setStartDate(e.target.value)}
        />
      </div>
      <div className="date-input-group">
        <label htmlFor="endDate">{t('date.end')}</label>
        <input
          type="date"
          id="endDate"
          value={endDate}
          onChange={(e) => setEndDate(e.target.value)}
        />
      </div>
      <button onClick={handleDateChange}>{t('regression.updateChart')}</button>
    </div>
  );
}

export default DateInput;
