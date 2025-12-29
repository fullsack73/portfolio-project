"use client"

import { useState, useEffect, useCallback } from "react"
import { useTranslation } from "react-i18next"

function DateInput({ onDateRangeChange }) {
  const { t } = useTranslation()
  const [startDate, setStartDate] = useState("")
  const [endDate, setEndDate] = useState("")

  useEffect(() => {
    // set default dates (3 months from yesterday)
    const today = new Date()
    const yesterday = new Date(today)
    yesterday.setDate(yesterday.getDate() - 1)
    const threeMonthsAgo = new Date(yesterday)
    threeMonthsAgo.setMonth(threeMonthsAgo.getMonth() - 3)

    // format dates for input fields (YYYY-MM-DD)
    const formatDate = (date) => {
      return date.toISOString().split("T")[0]
    }

    setEndDate(formatDate(yesterday))
    setStartDate(formatDate(threeMonthsAgo))
  }, [])

  // Debounced update function to prevent excessive API calls
  const debouncedUpdate = useCallback(
    (() => {
      let timeoutId
      return (start, end, source = "unknown") => {
        console.log(`📅 DateInput debounced update from: ${source}`, { start, end })
        clearTimeout(timeoutId)
        timeoutId = setTimeout(() => {
          if (start && end && onDateRangeChange) {
            console.log("📅 DateInput calling parent onDateRangeChange:", { start, end })
            onDateRangeChange(start, end)
          }
        }, 500) // 500ms delay
      }
    })(),
    [onDateRangeChange],
  )

  // Track if this is initial setup to avoid triggering parent callback on mount
  const [isInitialSetup, setIsInitialSetup] = useState(true)

  // Auto-update when dates change (but not on initial setup)
  useEffect(() => {
    if (startDate && endDate) {
      if (isInitialSetup) {
        console.log("📅 DateInput: Skipping initial setup callback")
        setIsInitialSetup(false)
      } else {
        console.log("📅 DateInput: User changed dates, calling debounced update")
        debouncedUpdate(startDate, endDate, "user-input")
      }
    }
  }, [startDate, endDate, debouncedUpdate, isInitialSetup])

  return (
    <div className="date-input-container">
      <div className="date-input-group">
        <label htmlFor="startDate">{t("date.start")}</label>
        <input
          type="date"
          id="startDate"
          value={startDate}
          onChange={(e) => setStartDate(e.target.value)}
          title="Chart will auto-update when both dates are selected"
        />
      </div>
      <div className="date-input-group">
        <label htmlFor="endDate">{t("date.end")}</label>
        <input
          type="date"
          id="endDate"
          value={endDate}
          onChange={(e) => setEndDate(e.target.value)}
          title="Chart will auto-update when both dates are selected"
        />
      </div>
    </div>
  )
}

export default DateInput
