import React, { useEffect, useState } from 'react'
import { useConfig } from '../context/ConfigContext'
import { getSummary } from '../services/datect'
import { formatDate } from '../lib/format'
import { trackRecordText } from '../lib/copy'
import { siteReliability } from '../lib/risk'
import Panel from '../components/ui/Panel'
import Caution from '../components/ui/Caution'

export default function About() {
  const { config, error } = useConfig()
  if (error) return <Caution>{error}</Caution>
  if (!config) return <p className="text-sm text-gray-500">Loading</p>
  return <AboutPage config={config} />
}

/** Sites whose range held under the weak threshold, as "Site k of n". */
function weakSiteText(config, holdout) {
  return config.sites
    .filter((s) => siteReliability(holdout.perSite[s]) === 'weak')
    .map((s) => `${s} ${holdout.perSite[s].band.inside} of ${holdout.perSite[s].band.nBanded}`)
    .join(', ')
}

function AboutPage({ config }) {
  const [summary, setSummary] = useState(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    let live = true
    getSummary()
      .then((s) => live && setSummary(s))
      .catch(() => live && setFailed(true))
    return () => {
      live = false
    }
  }, [])

  const holdout = summary?.reliability?.holdout
  const holdoutLabel = config.windows.holdout.label
  const weak = holdout ? weakSiteText(config, holdout) : ''

  return (
    <div className="max-w-4xl space-y-5">
      <h1 className="text-xl font-semibold">About DATect</h1>

      <Panel className="p-5 space-y-4">
        <h2 className="text-base font-semibold">Domoic acid forecasting system</h2>
        <p className="text-sm text-gray-600">
          DATect is a scientific machine learning system for forecasting harmful algal bloom concentrations
          (domoic acid) using satellite oceanographic data and environmental measurements.
        </p>

        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-sm font-semibold">Key features</h3>
            <ul className="list-disc list-inside text-sm text-gray-600 space-y-1 mt-1">
              <li>21 years of temporal coverage (2003-2023)</li>
              <li>10 Pacific Coast monitoring sites</li>
              <li>XGBoost + Random Forest ensemble with per-site configuration</li>
              <li>One-week forecast with an {Math.round(config.bandLevel * 100)}% range, graded spike risk and a track record</li>
              <li>Scientific data visualization</li>
            </ul>
          </div>
          <div>
            <h3 className="text-sm font-semibold">Data sources</h3>
            <ul className="list-disc list-inside text-sm text-gray-600 space-y-1 mt-1">
              <li>MODIS satellite and anomaly</li>
              <li>Climate indices (PDO, ONI, BEUTI)</li>
              <li>USGS streamflow measurements</li>
              <li>Shellfish toxin monitoring data</li>
              <li>Pseudo-nitzschia cell counts</li>
            </ul>
          </div>
        </div>

        <div>
          <h3 className="text-sm font-semibold">Scientific validation</h3>
          <p className="text-sm text-gray-600 mt-1">
            The system implements strict temporal safeguards to prevent data leakage and maintains
            peer-review quality standards with comprehensive validation testing.
          </p>
        </div>
      </Panel>

      <Panel className="p-5">
        <h2 className="text-base font-semibold">What the dashboard cannot do</h2>
        <ul className="list-disc list-inside text-sm text-gray-700 space-y-1.5 mt-2">
          {holdout && (
            <li>
              The range is calibrated over all sites, not per site.
              {weak && ` Weakest sites in ${holdoutLabel}, measurements inside the range: ${weak}.`}
            </li>
          )}
          {holdout && <li>{trackRecordText(holdout.overall, holdoutLabel)}</li>}
          <li>Data ends {formatDate(config.newestMeasurement)}; forecasts are not live.</li>
        </ul>
        {failed && <p className="text-xs text-gray-500 mt-2">Track record figures are unavailable right now.</p>}
      </Panel>
    </div>
  )
}
