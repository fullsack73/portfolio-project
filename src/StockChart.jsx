import React from 'react';
import Plot from 'react-plotly.js';

function StockChart({ data, ticker }) {
  return (
    <Plot
      data={[
        {
          x: Object.keys(data),
          y: Object.values(data),
          type: 'scatter',
          mode: 'lines',
          line: { color: 'blue' },
        },
      ]}
      layout={{
        title: `${ticker} Stock Data`,
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
  );
}

export default StockChart;