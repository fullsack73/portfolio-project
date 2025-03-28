import React, { useState, useEffect } from 'react';

function DateInput({ onDateRangeChange }) {
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');

  useEffect(() => {
    // Set default dates (3 months from yesterday)
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    const threeMonthsAgo = new Date(yesterday);
    threeMonthsAgo.setMonth(threeMonthsAgo.getMonth() - 3);

    // Format dates for input fields (YYYY-MM-DD)
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
        <label htmlFor="startDate">Start Date:</label>
        <input
          type="date"
          id="startDate"
          value={startDate}
          onChange={(e) => setStartDate(e.target.value)}
        />
      </div>
      <div className="date-input-group">
        <label htmlFor="endDate">End Date:</label>
        <input
          type="date"
          id="endDate"
          value={endDate}
          onChange={(e) => setEndDate(e.target.value)}
        />
      </div>
      <button onClick={handleDateChange}>Update Chart</button>
    </div>
  );
}

export default DateInput;
