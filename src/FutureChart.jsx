import React from 'react';
import Plot from 'react-plotly.js';
import { useTranslation } from 'react-i18next';

const FutureChart = ({ data, ticker }) => {
  const { t } = useTranslation();

  return (
    <Plot
      data={[
        {
          x: Object.keys(data),
          y: Object.values(data),
          type: 'scatter',
          mode: 'lines',
          name: t('future.predicted_price', 'Predicted Price'),
          line: { color: '#2ca02c' }, // Green for predicted price
        },
      ]}
      layout={{
        title: t('future.chart_title', `${ticker} Future Price Prediction`),
        xaxis: { 
          title: 'Date',
          tickangle: 45,
          tickformat: '%Y-%m-%d'
        },
        yaxis: { title: 'Price ($)' },
        height: 400,
        margin: { t: 50, b: 100, l: 50, r: 50 }
      }}
      style={{ width: '100%', height: '100%' }}
      useResizeHandler={true}
    />
  );
};

export default FutureChart;
