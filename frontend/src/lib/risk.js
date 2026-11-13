import { daysBetween, formatDate } from './format'

export const STALE_DAYS = 21
export const WEAK_COVERAGE = 0.7
export const MIN_ROWS = 10

/**
 * Left-closed category containing v: [lo, hi).
 * @param {number} v
 * @param {import('../services/contract').Config['categories']} categories
 */
export function categoryOf(v, categories) {
  return categories.find((c) => v >= c.lo && v < c.hi) ?? categories[categories.length - 1]
}

/**
 * Whether the top of the range reaches the threshold (a value at it counts).
 * @param {{ lo: number, hi: number } | null} band
 * @param {number} threshold
 * @returns {boolean}
 */
export function reachesThreshold(band, threshold) {
  return band != null && band.hi >= threshold
}

/**
 * Maps the API's alert onto a display key; the API decides the level.
 * @param {{ level: string | null, reason: string | null } | null} alert
 * @returns {'already' | 'quiet' | 'watch' | 'warning' | 'unavailable'}
 */
export function levelKey(alert) {
  if (!alert) return 'unavailable'
  if (alert.reason === 'already_at_or_above_threshold') return 'already'
  if (alert.level) return alert.level.toLowerCase()
  return 'unavailable'
}

/**
 * No measurement, or the newest is more than STALE_DAYS before the anchor.
 * @param {string | null} lastDate ISO date
 * @param {string} anchorDate ISO date
 * @returns {boolean}
 */
export function isStale(lastDate, anchorDate) {
  return lastDate == null || daysBetween(lastDate, anchorDate) > STALE_DAYS
}

/**
 * What to say with no latest measurement. A failed forecast may not have got as far as
 * reading the measurements, so it does not claim there are none.
 * @param {{ error: string | null }} forecast
 * @returns {string}
 */
export function noMeasurementText({ error }) {
  return error ? 'Measurements unavailable for this forecast' : 'No measurement before this forecast'
}

/**
 * @param {import('../services/contract').Reliability | undefined} reliability a site's holdout reliability
 * @returns {'weak' | 'ok' | 'too-few'}
 */
export function siteReliability(reliability) {
  const band = reliability?.band
  if (!band || band.coverage == null || band.nBanded < MIN_ROWS) return 'too-few'
  return band.coverage < WEAK_COVERAGE ? 'weak' : 'ok'
}

/**
 * A notice when the newest measurement is more than 30 days before today.
 * @param {string} newestIso
 * @param {string} todayIso
 * @returns {string | null}
 */
export function dataCurrencyNotice(newestIso, todayIso) {
  if (daysBetween(newestIso, todayIso) <= 30) return null
  return `A demo on past data: the newest measurement is from ${formatDate(newestIso)}, so every forecast here is for a past week.`
}
