import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';

function MyComponent() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Fetch data from the Python backend
    fetch('http://127.0.0.1:5000/get-data')
      .then((response) => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then((data) => {
        setData(data);
        setLoading(false);
      })
      .catch((error) => {
        setError(error.message);
        setLoading(false);
      });
  }, []);

  if (loading) return <p>Loading...</p>;
  if (error) return <p>Error: {error}</p>;

  // Prepare data for the graph
  const countries = ['ROK', 'JAP', 'PRC', 'ROC', 'RUS'];
  const values = countries.map((country) => data[country]);

  return (
    <div>
      <h1>GDP Data in East Asian Countries</h1>
      <Plot
        data={[
          {
            x: countries,
            y: values,
            type: 'scatter', // Change to scatter for line graph
            mode: 'lines', // Specify mode as lines
            line: { color: 'blue' }, // Optional: Customize line color
          },
        ]}
        layout={{
          title: 'GDP Data',
          xaxis: { title: 'Country' },
          yaxis: { title: 'Value' },
        }}
      />
    </div>
  );
}

export default MyComponent;