import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { useConfig } from '../context/ConfigContext'
import { getSummary } from '../services/datect'
import { formatDate, formatPct, formatValue } from '../lib/format'
import { siteReliability } from '../lib/risk'
import Panel from '../components/ui/Panel'
import Caution from '../components/ui/Caution'

export default function About() {
  const { config, error } = useConfig()
  if (error) return <Caution>{error}</Caution>
  if (!config) return <p className="text-sm text-gray-500">Loading</p>
  return <AboutPage config={config} />
}

function Section({ title, children }) {
  return (
    <Panel className="p-5">
      <h2 className="text-base font-semibold">{title}</h2>
      <div className="mt-2 text-sm text-gray-700 space-y-2 leading-relaxed">{children}</div>
    </Panel>
  )
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

  const threshold = config.spikeThreshold
  const holdout = summary?.reliability?.holdout
  const holdoutLabel = config.windows.holdout.label
  const mae = summary?.headline?.overall?.mae
  const weakSites = holdout ? config.sites.filter((s) => siteReliability(holdout.perSite[s]) === 'weak') : []

  return (
    <div className="max-w-3xl space-y-5">
      <h1 className="text-xl font-semibold">About DATect</h1>

      <Section title="What this is">
        <p>
          Domoic acid is a natural toxin made by some algae (<i>Pseudo-nitzschia</i>). Shellfish such as razor clams and
          mussels take it up, and eating them can cause amnesic shellfish poisoning. Washington and Oregon close shellfish
          harvests when the toxin reaches {threshold} µg/g.
        </p>
        <p>
          DATect is a research project that forecasts next week&apos;s toxin level at ten beaches on the Washington and
          Oregon coast, from 2003 to 2023.
        </p>
      </Section>

      <Section title="How to read the forecast page">
        <ul className="list-disc pl-5 space-y-1.5">
          <li>
            <span className="font-medium">How much toxin next week?</span> The forecast and its likely range: across all
            sites, about {formatPct(config.bandLevel)} of measurements land inside that range.
          </li>
          <li>
            <span className="font-medium">Any sign of a sudden rise?</span> A separate alert for jumps from below{' '}
            {threshold} µg/g to {threshold} or more, which the forecast alone tends to miss. It reads Quiet, Watch or
            Warning.
          </li>
          <li>
            <span className="font-medium">What was measured last?</span> The latest lab results and how old they are. A
            forecast without a recent sample deserves less trust.
          </li>
        </ul>
      </Section>

      <Section title="How it works">
        <p>
          Each site&apos;s forecast blends two machine-learning models (XGBoost and a random forest) with the last
          measurement. The models learn from past toxin measurements, algae counts, satellite readings of the ocean,
          river flow and climate indices. Every forecast uses
          only data that was available a week before the measurement it predicts.
        </p>
        <p className="text-gray-600">
          Data: shellfish toxin monitoring, <i>Pseudo-nitzschia</i> counts, MODIS satellite ocean colour and temperature,
          the PDO, ONI and BEUTI climate and upwelling indices, and USGS river flow.
        </p>
      </Section>

      <Section title="How well it works">
        {holdout ? (
          <p>
            In {holdoutLabel}, years never used to tune the models, forecasts were off by{' '}
            {mae ? `${formatValue(mae.value)} µg/g` : 'a few µg/g'} on average. The alert caught {holdout.overall.onsets.caught}{' '}
            of {holdout.overall.onsets.n} sudden rises to {threshold} µg/g, and {holdout.overall.alarms.followed} of its{' '}
            {holdout.overall.alarms.n} alarms were followed by one; {holdout.overall.alarms.falseNearMiss} of the other{' '}
            {holdout.overall.alarms.n - holdout.overall.alarms.followed} came close (10–20 µg/g).
          </p>
        ) : (
          <p>{failed ? 'Track record figures are unavailable right now.' : 'Loading the track record.'}</p>
        )}
        <p>
          The <Link to="/track-record" className="underline">Track record</Link> page has the details for every site.
        </p>
      </Section>

      <Section title="Limits">
        <ul className="list-disc pl-5 space-y-1.5">
          <li>
            The data ends on {formatDate(config.newestMeasurement)}, so this is a demonstration on past weeks, not a live
            service.
          </li>
          {weakSites.length > 0 && (
            <li>
              Forecasts are less reliable at {weakSites.join(' and ')}, where measurements often landed outside the likely
              range.
            </li>
          )}
          <li>
            The alert&apos;s score is not a probability. The forecast page shows how often weeks at each alert level were
            actually followed by a rise.
          </li>
          <li>Not for making real harvest decisions: always follow official closure notices.</li>
        </ul>
      </Section>
    </div>
  )
}
