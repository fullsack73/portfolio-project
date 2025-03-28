import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import DateInput from './DateInput.jsx';

function StockChart() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = (startDate = null, endDate = null) => {
    setLoading(true);
    let url = 'http://127.0.0.1:5000/get-data';
    if (startDate && endDate) {
      url += `?start_date=${startDate}&end_date=${endDate}`;
    }

    console.log('Attempting to fetch data...');
    fetch(url, {
      method: 'GET',
      mode: 'cors',
      credentials: 'include',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
      }
    })
      .then((response) => {
        console.log('Response status:', response.status);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        console.log('Raw data received:', data);
        setData(data);
        setLoading(false);
      })
      .catch((error) => {
        console.error('Fetch error:', error);
        setError(error.message);
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleDateRangeChange = (startDate, endDate) => {
    fetchData(startDate, endDate);
  };

  if (loading) return <p>Loading...</p>;
  if (error) return <p>Error: {error}</p>;

  // Prepare data for the graph
  const dates = Object.keys(data);
  const prices = Object.values(data);
  
  console.log('Dates array:', dates);
  console.log('Prices array:', prices);
  console.log('Number of data points:', dates.length);

  return (
    <div>
      <h1>Apple Stock Data</h1>
      <DateInput onDateRangeChange={handleDateRangeChange} />
      <Plot
        data={[
          {
            x: dates,
            y: prices,
            type: 'scatter',
            mode: 'lines',
            line: { color: 'blue' },
          },
        ]}
        layout={{
          title: 'Apple Stock Data',
          xaxis: { 
            title: 'Date',
            tickangle: 45,
            tickformat: '%Y-%m-%d'
          },
          yaxis: { title: 'Price ($)' },
          height: 600,
          margin: { t: 50, b: 100, l: 50, r: 50 }
        }}
      />
    </div>
  );
}

export default StockChart;