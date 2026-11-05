import React, { useEffect, useRef, useState } from 'react'
import { useConfig } from '../context/ConfigContext'
import { getForecast, getSummary } from '../services/datect'
import FilterBar from '../components/forecast/FilterBar'
import ForecastCard from '../components/forecast/ForecastCard'
import SpikeRiskCard from '../components/forecast/SpikeRiskCard'
import LatestMeasurementCard from '../components/forecast/LatestMeasurementCard'
import GroundDataChart from '../components/forecast/GroundDataChart'
import MetricsPanel from '../components/forecast/MetricsPanel'
import AllSitesTable from '../components/forecast/AllSitesTable'
import Caution from '../components/ui/Caution'

const SLOW_MS = 3000

function safeStorage() {
  try {
    return window.localStorage
  } catch {
    return null
  }
}

function storedSite(sites) {
  try {
    const s = localStorage.getItem('datect.site')
    if (s && sites.includes(s)) return s
  } catch {
    // storage unavailable
  }
  return sites[0]
}

export default function Forecast() {
  const { config, error: configError } = useConfig()
  if (configError) return <Caution>{configError}</Caution>
  if (!config) return <p className="text-sm text-gray-500">Loading</p>
  return <ForecastPage config={config} />
}

function ForecastPage({ config }) {
  const [site, setSite] = useState(() => storedSite(config.sites))
  const [week, setWeek] = useState(config.latestWeek)
  const [forecast, setForecast] = useState(null)
  const [summary, setSummary] = useState(null)
  const [summaryError, setSummaryError] = useState(null)
  const [loading, setLoading] = useState(true)
  const [slow, setSlow] = useState(false)
  const [error, setError] = useState(null)
  const [settled, setSettled] = useState(false)
  const latest = useRef(0)
  const [storage] = useState(safeStorage)

  useEffect(() => {
    let cancelled = false
    getSummary()
      .then((s) => !cancelled && setSummary(s))
      .catch((e) => !cancelled && setSummaryError(e?.message || String(e)))
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    const id = ++latest.current
    setLoading(true)
    setSlow(false)
    setError(null)
    const timer = setTimeout(() => latest.current === id && setSlow(true), SLOW_MS)
    getForecast(site, week)
      .then((f) => latest.current === id && (setForecast(f), setLoading(false), setSettled(true)))
      .catch(
        (e) =>
          latest.current === id &&
          (setForecast(null), setError(e?.message || String(e)), setLoading(false), setSettled(true)),
      )
      .finally(() => clearTimeout(timer))
    return () => clearTimeout(timer)
  }, [site, week])

  const chooseSite = (s) => {
    setSite(s)
    try {
      localStorage.setItem('datect.site', s)
    } catch {
      // storage unavailable
    }
  }

  return (
    <div className="space-y-5">
      <FilterBar
        config={config}
        summary={summary}
        site={site}
        week={week}
        anchorDate={loading ? null : forecast?.anchorDate ?? null}
        onSite={chooseSite}
        onWeek={setWeek}
      />
      {loading && slow && <p className="text-xs text-gray-500">Computing forecast; this can take a minute.</p>}
      {error && <Caution>{error}</Caution>}
      {summaryError && <Caution>The track record could not be loaded: {summaryError}</Caution>}
      {forecast && (
        <div className={`grid grid-cols-1 md:grid-cols-3 gap-5 ${loading ? 'opacity-60' : ''}`}>
          <ForecastCard forecast={forecast} summary={summary} config={config} />
          <SpikeRiskCard forecast={forecast} summary={summary} config={config} />
          <LatestMeasurementCard forecast={forecast} config={config} />
        </div>
      )}
      <GroundDataChart site={site} week={week} forecast={forecast} config={config} loading={loading} />
      <MetricsPanel site={site} forecast={forecast} summary={summary} summaryError={summaryError} config={config} storage={storage} loading={loading} />
      <AllSitesTable week={week} site={site} ready={settled} config={config} summary={summary} onSelectSite={chooseSite} />
    </div>
  )
}
