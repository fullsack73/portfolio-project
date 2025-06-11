import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useTranslation } from 'react-i18next';

const FutureChart = ({ data, ticker }) => {
  const { t } = useTranslation();

  const chartData = Object.keys(data).map(date => ({
    date,
    price: data[date],
  }));

  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart
        data={chartData}
        margin={{
          top: 5,
          right: 30,
          left: 20,
          bottom: 5,
        }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="price" stroke="#82ca9d" name={t('future.predicted_price', 'Predicted Price')} />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default FutureChart;
