import Plot from "react-plotly.js"

function StockChart({ data, ticker }) {
  return (
    <Plot
      data={[
        {
          x: Object.keys(data),
          y: Object.values(data),
          type: "scatter",
          mode: "lines",
          line: {
            color: "#06b6d4",
            width: 2.5,
          },
          fill: "tozeroy",
          fillcolor: "rgba(6, 182, 212, 0.1)",
        },
      ]}
      layout={{
        title: {
          text: `${ticker} Stock Data`,
          font: { color: "#e5e7eb", size: 18, family: "Inter, system-ui, sans-serif" },
        },
        paper_bgcolor: "rgba(30, 41, 59, 0.5)",
        plot_bgcolor: "rgba(15, 23, 42, 0.3)",
        xaxis: {
          title: { text: "Date", font: { color: "#94a3b8" } },
          tickangle: 45,
          tickformat: "%Y-%m-%d",
          color: "#94a3b8",
          gridcolor: "rgba(148, 163, 184, 0.1)",
        },
        yaxis: {
          title: { text: "Price ($)", font: { color: "#94a3b8" } },
          color: "#94a3b8",
          gridcolor: "rgba(148, 163, 184, 0.1)",
        },
        height: 600,
        margin: { t: 50, b: 100, l: 50, r: 50 },
      }}
      config={{
        displayModeBar: true,
        displaylogo: false,
      }}
    />
  )
}

export default StockChart
