import React from 'react';
import Plot from 'react-plotly.js';

// linear regression for non-linear data. retarded, i know.
function RegressionChart({ data, regression, ticker, formula }) {
  return (
    <Plot
      data={[
        {
          x: Object.keys(data),
          y: Object.values(data),
          type: 'scatter',
          mode: 'markers',
          name: 'Actual Prices',
          marker: { 
            color: 'blue',
            size: 6,
            opacity: 0.6
          }
        },
        {
          x: Object.keys(regression),
          y: Object.values(regression),
          type: 'scatter',
          mode: 'lines',
          name: 'Regression Line',
          line: { 
            color: 'red',
            width: 2
          }
        }
      ]}
      layout={{
        title: `${ticker} Price Regression`,
        xaxis: { 
          title: 'Date',
          tickangle: 45,
          tickformat: '%Y-%m-%d'
        },
        yaxis: { title: 'Price ($)' },
        height: 600,
        margin: { t: 50, b: 100, l: 50, r: 50 },
        showlegend: true,
        legend: {
          x: 1,
          y: 1,
          xanchor: 'right',
          yanchor: 'top'
        },
        annotations: [
          {
            text: formula,
            x: 0.02,
            y: 0.98,
            xref: 'paper',
            yref: 'paper',
            showarrow: false,
            font: {
              size: 12,
              color: 'red'
            }
          }
        ]
      }}
    />
  );
}

export default RegressionChart;

