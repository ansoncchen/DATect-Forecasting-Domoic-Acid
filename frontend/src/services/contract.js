// The one place API response shapes are known. Everything else in the dashboard
// consumes the view models typed below and never touches an API field name.
import { windowLabel } from '../lib/format'

/**
 * @typedef {{ name: string, start: string, end: string, label: string }} Window
 *
 * @typedef {{
 *   spikeThreshold: number, regressionAlert: number, classifierCutoff: number,
 *   levels: { name: string, cutoff: number | null, placeholder: boolean }[],
 *   categories: { lo: number, hi: number, label: string }[],
 *   windows: { validation: Window, holdout: Window },
 *   bandLevel: number, newestMeasurement: string, latestWeek: string,
 *   sites: string[], dateRange: { min: string, max: string }
 * }} Config
 *
 * @typedef {{
 *   probability: number | null, byClassifier: boolean, byRegression: boolean, fired: boolean,
 *   level: 'Quiet' | 'Watch' | 'Warning' | null, reason: string | null
 * }} Alert
 *
 * @typedef {{
 *   site: string, date: string, anchorDate: string, error: string | null,
 *   value: number | null,
 *   band: { lo: number, hi: number, nCalibration: number } | null,
 *   bandReason: string | null,
 *   categoryChances: number[] | null, bandPExceed: number | null,
 *   alert: Alert | null,
 *   lastMeasurement: { date: string, value: number } | null,
 *   recentMeasurements: { date: string, value: number }[],
 *   parts: { xgb: number, rf: number, naive: number, weights: number[] } | null,
 *   drivers: { feature: string, importance: number }[],
 *   provenance: { source: string | null, modelVersion: string | null, trainingSamples: number | null }
 * }} Forecast
 *
 * @typedef {{
 *   date: string, anchorDate: string, actual: number | null, predicted: number | null,
 *   lo: number | null, hi: number | null, probability: number | null,
 *   fired: boolean, level: string | null
 * }} HistoryRow
 *
 * @typedef {{ value: number, ci: [number, number] | null }} Stat
 *
 * @typedef {{
 *   n: number, r2: Stat | null, logR2: Stat | null, mae: Stat | null,
 *   models: { ensemble: object, naive: object, linear: object } | null,
 *   band: { coverage: number | null, onsetCoverage: number | null, medianWidth: number | null,
 *           inside: number, nBanded: number, onsetsInside: number },
 *   onsets: { n: number, caught: number, quiet: number, quietAlarmed: number },
 *   alarms: { n: number, followed: number, falseBelow5: number, false5To10: number, falseNearMiss: number },
 *   levels: Object<string, { weeks: number, followed: number }>
 * }} Reliability
 *
 * @typedef {{
 *   n: number, nSpikes: number, r2: Stat | null, logR2: Stat | null, mae: Stat | null,
 *   spike: { recall: number | null, precision: number | null, f2: number | null }
 * }} Headline
 *
 * @typedef {{
 *   source: object,
 *   headline: { overall: Headline, perSite: Object<string, Headline> },
 *   reliability: Object<string, { overall: Reliability, perSite: Object<string, Reliability> }>
 * }} Summary
 *
 * @typedef {{
 *   window: Window, overall: Reliability, perSite: Object<string, Reliability>,
 *   rows: { date: string, site: string, actual: number | null, ensemble: number | null,
 *           naive: number | null, linear: number | null, lo: number | null, hi: number | null,
 *           pExceed: number | null, probability: number | null, fired: boolean }[]
 * }} TrackRecord
 */

const orNull = (v) => (v === undefined ? null : v)

/**
 * The API sends r2 / mae as a number or as [value, lo, hi] (a one-element array has no interval).
 * @returns {Stat | null}
 */
export function toStat(raw) {
  if (raw === null || raw === undefined) return null
  if (Array.isArray(raw)) {
    const [value, lo, hi] = raw
    if (value === null || value === undefined) return null
    return { value, ci: lo === null || lo === undefined || hi === null || hi === undefined ? null : [lo, hi] }
  }
  return { value: raw, ci: null }
}

function toWindow(name, start, end) {
  return { name, start, end, label: windowLabel(start, end) }
}

/** @returns {Config} */
export function toConfig(raw, rawSites) {
  const w = raw.eval_windows
  const bins = raw.category_bins
  return {
    spikeThreshold: raw.spike_threshold,
    regressionAlert: raw.regression_alert_threshold,
    classifierCutoff: raw.classifier_alert_cutoff,
    levels: raw.alert_levels.map((l) => ({ name: l.name, cutoff: l.cutoff, placeholder: l.placeholder })),
    categories: raw.category_labels.map((label, i) => ({
      lo: bins[i],
      hi: bins[i + 1] === null ? Infinity : bins[i + 1],
      label: label.split(' (')[0],
    })),
    windows: {
      validation: toWindow('validation', w.validation_start, w.validation_end),
      holdout: toWindow('holdout', w.holdout_start, w.holdout_end),
    },
    bandLevel: raw.band_level,
    newestMeasurement: raw.newest_measurement_date,
    latestWeek: raw.latest_forecast_week,
    sites: raw.sites_north_to_south,
    dateRange: { min: rawSites.date_range.min, max: rawSites.date_range.max },
  }
}

/** @returns {Alert | null} */
function toAlert(raw) {
  if (!raw) return null
  return {
    probability: orNull(raw.classifier_probability),
    byClassifier: raw.by_classifier,
    byRegression: raw.by_regression,
    fired: raw.alert,
    level: orNull(raw.level),
    reason: orNull(raw.level_reason),
  }
}

function toBand(raw) {
  return raw ? { lo: raw.lo, hi: raw.hi, nCalibration: raw.n_calibration } : null
}

function toParts(ensemble) {
  if (!ensemble) return null
  return {
    xgb: ensemble.xgb_prediction,
    rf: ensemble.rf_prediction,
    naive: ensemble.naive_prediction,
    weights: ensemble.ensemble_weights,
  }
}

function toDrivers(regression) {
  const importance = regression?.feature_importance
  if (!importance) return []
  return Object.entries(importance)
    .map(([feature, value]) => ({ feature, importance: value }))
    .sort((a, b) => b.importance - a.importance)
}

/** @returns {Forecast} */
export function toForecast(raw) {
  const error = raw.error ?? (raw.success === false ? 'The forecast failed' : null)
  const regression = raw.regression
  return {
    site: raw.site,
    date: raw.forecast_date,
    anchorDate: raw.anchor_date,
    error,
    value: error ? null : (regression?.predicted_da ?? null),
    band: toBand(raw.band),
    bandReason: orNull(raw.band_reason),
    categoryChances: raw.category_probabilities ?? raw.band?.category_probabilities ?? null,
    bandPExceed: raw.band_p_exceed ?? raw.band?.p_exceed ?? null,
    alert: toAlert(raw.alert),
    lastMeasurement: raw.last_measurement ?? null,
    recentMeasurements: raw.recent_measurements ?? [],
    parts: toParts(raw.ensemble),
    drivers: toDrivers(regression),
    provenance: {
      source: orNull(raw.source),
      modelVersion: orNull(raw.model_version),
      trainingSamples: regression?.training_samples ?? null,
    },
  }
}

/** One Forecast per site; fields the endpoint does not carry are null or []. @returns {Forecast[]} */
export function toAllSites(raw) {
  return raw.sites.map((s) => ({
    site: s.site,
    date: raw.date,
    anchorDate: s.anchor_date,
    error: s.error ?? null,
    value: s.error ? null : orNull(s.predicted_da),
    band: toBand(s.band),
    bandReason: null,
    categoryChances: s.band?.category_probabilities ?? null,
    bandPExceed: s.band?.p_exceed ?? null,
    alert: toAlert(s.alert),
    lastMeasurement: s.last_measurement ?? null,
    recentMeasurements: [],
    parts: null,
    drivers: [],
    provenance: { source: orNull(s.source), modelVersion: null, trainingSamples: null },
  }))
}

/** @returns {HistoryRow[]} */
export function toHistory(raw) {
  return raw.rows.map((r) => ({
    date: r.date,
    anchorDate: r.anchor_date,
    actual: orNull(r.actual_da),
    predicted: orNull(r.predicted_da),
    lo: orNull(r.band_lo),
    hi: orNull(r.band_hi),
    probability: orNull(r.spike_probability),
    fired: r.alert ?? r.spike_alert ?? false,
    level: orNull(r.level),
  }))
}

const toModelScores = (m) => (m ? { r2: m.r2, mae: m.mae } : null)

/** @returns {Reliability} */
function toReliability(raw) {
  const m = raw.models
  return {
    n: raw.n,
    r2: toStat(raw.r2),
    logR2: toStat(raw.log_r2),
    mae: toStat(raw.mae),
    models: m ? { ensemble: toModelScores(m.ensemble), naive: toModelScores(m.naive), linear: toModelScores(m.linear) } : null,
    band: {
      coverage: orNull(raw.band.coverage),
      onsetCoverage: orNull(raw.band.coverage_at_onsets),
      medianWidth: orNull(raw.band.median_width),
      inside: raw.band.inside,
      nBanded: raw.band.n_banded,
      onsetsInside: raw.band.onsets_inside,
    },
    onsets: {
      n: raw.onsets.n,
      caught: raw.onsets.caught,
      quiet: raw.onsets.quiet,
      quietAlarmed: raw.onsets.quiet_alarmed,
    },
    alarms: {
      n: raw.alarms.n,
      followed: raw.alarms.followed_by_jump,
      falseBelow5: raw.alarms.false_below_5,
      false5To10: raw.alarms.false_5_to_10,
      falseNearMiss: raw.alarms.false_near_miss,
    },
    levels: Object.fromEntries(
      Object.entries(raw.levels).map(([name, l]) => [name, { weeks: l.weeks, followed: l.followed_by_jump }]),
    ),
  }
}

const mapValues = (obj, fn) => Object.fromEntries(Object.entries(obj).map(([k, v]) => [k, fn(v)]))

/** @returns {Headline} */
function toHeadline(raw) {
  return {
    n: raw.n,
    nSpikes: raw.n_spikes,
    r2: toStat(raw.r2),
    logR2: toStat(raw.r2_log1p),
    mae: toStat(raw.mae),
    spike: { recall: orNull(raw.spike_recall), precision: orNull(raw.spike_precision), f2: orNull(raw.spike_f2) },
  }
}

/** @returns {Summary} */
export function toSummary(raw) {
  return {
    source: raw.source,
    headline: { overall: toHeadline(raw.overall), perSite: mapValues(raw.per_site, toHeadline) },
    reliability: mapValues(raw.reliability, (r) => ({
      overall: toReliability(r.summary),
      perSite: mapValues(r.per_site, toReliability),
    })),
  }
}

/** @returns {TrackRecord} */
export function toTrackRecord(raw) {
  const w = raw.window
  return {
    window: toWindow(w.name, w.start, w.end),
    overall: toReliability(raw.summary),
    perSite: mapValues(raw.per_site, toReliability),
    rows: raw.rows.map((r) => ({
      date: r.date,
      site: r.site,
      actual: orNull(r.actual_da),
      ensemble: orNull(r.predicted_da),
      naive: orNull(r.naive_prediction),
      linear: orNull(r.predicted_da_linear),
      lo: orNull(r.band_lo),
      hi: orNull(r.band_hi),
      pExceed: orNull(r.p_exceed),
      probability: orNull(r.spike_probability),
      fired: r.alert ?? r.spike_alert ?? false,
    })),
  }
}
