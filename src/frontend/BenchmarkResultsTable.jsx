import { useTranslation } from "react-i18next"

function BenchmarkResultsTable({ summary }) {
  const { t } = useTranslation()

  // Helper function to format currency
  const formatCurrency = (value) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value)
  }

  // Helper function to format percentage
  const formatPercent = (value) => {
    return `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`
  }

  // Helper function to get color class based on value
  const getColorClass = (value) => {
    return value >= 0 ? "positive" : "negative"
  }

  const rows = [
    {
      name: t("benchmark.portfolio"),
      data: summary.portfolio,
    },
    {
      name: t("benchmark.sp500"),
      data: summary.sp500_benchmark,
    },
    {
      name: t("benchmark.riskFree"),
      data: summary.risk_free_asset,
    },
  ]

  return (
    <div className="benchmark-table-container">
      <table className="benchmark-table">
        <thead>
          <tr>
            <th>{t("benchmark.investmentType")}</th>
            <th>{t("benchmark.initialValue")}</th>
            <th>{t("benchmark.finalValue")}</th>
            <th>{t("benchmark.profitLoss")}</th>
            <th>{t("benchmark.returnPercent")}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>
              <td className="benchmark-name">{row.name}</td>
              <td>{formatCurrency(row.data.initial_value)}</td>
              <td>{formatCurrency(row.data.final_value)}</td>
              <td className={getColorClass(row.data.profit_loss)}>
                {formatCurrency(row.data.profit_loss)}
              </td>
              <td className={getColorClass(row.data.return_pct)}>
                {formatPercent(row.data.return_pct)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Comparison Section */}
      <div className="benchmark-comparison">
        <h3>{t("benchmark.comparison")}</h3>
        <div className="comparison-grid">
          <div className="comparison-item">
            <span className="comparison-label">{t("benchmark.vssp500")}:</span>
            <span className={getColorClass(summary.portfolio.return_pct - summary.sp500_benchmark.return_pct)}>
              {formatPercent(summary.portfolio.return_pct - summary.sp500_benchmark.return_pct)}
            </span>
          </div>
          <div className="comparison-item">
            <span className="comparison-label">{t("benchmark.vsRiskFree")}:</span>
            <span className={getColorClass(summary.portfolio.return_pct - summary.risk_free_asset.return_pct)}>
              {formatPercent(summary.portfolio.return_pct - summary.risk_free_asset.return_pct)}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default BenchmarkResultsTable
