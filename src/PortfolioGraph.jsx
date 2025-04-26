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
                    x: [data.riskless_point?.vol],
                    y: [data.riskless_point?.ret],
                    mode: 'markers',
                    type: 'scatter',
                    marker: { color: 'green', size: 12, symbol: 'star' },
                    name: 'Riskless Asset'
                },
                {
                    x: data.cml_vols,
                    y: data.cml_rets,
                    mode: 'lines',
                    type: 'scatter',
                    line: { color: 'blue', width: 2, dash: 'dash' },
                    name: 'Capital Market Line'
                },
                {
                    x: [data.opt_vol],
                    y: [data.opt_ret],
                    mode: 'markers',
                    type: 'scatter',
                    marker: { color: 'yellow', size: 12 },
                    name: 'Tangency Portfolio'
                },
                {
                    x: [data.optv_vol],
                    y: [data.optv_ret],
                    mode: 'markers',
                    type: 'scatter',
                    marker: { color: 'red', size: 12 },
                    name: 'Minimum Volatility'
                },
                {
                    x: data.tvols,
                    y: data.trets,
                    mode: 'lines',
                    type: 'scatter',
                    line: { color: 'red', width: 2 },
                    name: 'Efficient Frontier'
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
