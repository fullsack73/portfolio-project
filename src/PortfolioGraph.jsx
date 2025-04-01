import React from 'react';
import Plot from 'react-plotly.js';

const PortfolioGraph = ({ data }) => {
    return (
        <Plot
            data={[
                {
                    x: data.pvols,
                    y: data.prets,
                    mode: 'markers',
                    type: 'scatter',
                    marker: { color: 'rgba(75, 192, 192, 0.6)' },
                    name: 'Portfolio'
                },
                {
                    x: [data.opt_vol],
                    y: [data.opt_ret],
                    mode: 'markers',
                    type: 'scatter',
                    marker: { color: 'yellow', size: 12 },
                    name: 'Optimal Portfolio'
                },
                {
                    x: [data.optv_vol],
                    y: [data.optv_ret],
                    mode: 'markers',
                    type: 'scatter',
                    marker: { color: 'red', size: 12 },
                    name: 'Minimum Volatility'
                }
            ]}
            layout={{
                width: 700, height: 500,
                title: 'Portfolio Returns vs Volatility',
                xaxis: { title: 'Volatility' },
                yaxis: { title: 'Return' },
                showlegend: true
            }}
        />
    );
};

export default PortfolioGraph;
