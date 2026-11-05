import { differenceInCalendarDays, parseISO, subDays } from 'date-fns'

/**
 * Label for an evaluation window whose end is exclusive, with an en dash and the
 * inclusive end year: ('2022-01-01', '2024-01-01') -> '2022–2023'.
 * @param {string} start ISO date
 * @param {string} end ISO date, exclusive
 * @returns {string}
 */
export function windowLabel(start, end) {
  return `${parseISO(start).getFullYear()}–${subDays(parseISO(end), 1).getFullYear()}`
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

/**
 * One decimal; a positive value under 0.1 prints '<0.1'.
 * @param {number} v
 * @returns {string}
 */
export function formatValue(v) {
  if (v > 0 && v < 0.1) return '<0.1'
  return v.toFixed(1)
}

/**
 * Whole-number percentage of a 0..1 proportion.
 * @param {number} p
 * @returns {string}
 */
export function formatPct(p) {
  return `${Math.round(p * 100)}%`
}

/**
 * A classifier score to two decimals. It is not a calibrated chance, so never a percentage.
 * @param {number | null | undefined} score
 * @returns {string} '—' for a missing score
 */
export function formatScore(score) {
  return score == null ? '—' : score.toFixed(2)
}

/**
 * 'D Mon YYYY' from an ISO date, independent of locale and time zone.
 * @param {string} iso
 * @returns {string}
 */
export function formatDate(iso) {
  const d = parseISO(iso)
  return `${d.getDate()} ${MONTHS[d.getMonth()]} ${d.getFullYear()}`
}

/**
 * Whole calendar days from isoA to isoB (negative when B is earlier).
 * @param {string} isoA
 * @param {string} isoB
 * @returns {number}
 */
export function daysBetween(isoA, isoB) {
  return differenceInCalendarDays(parseISO(isoB), parseISO(isoA))
}

/**
 * A span of days in the coarsest unit that reads naturally: days under two
 * weeks, weeks under about two months, otherwise months.
 * @param {number} days
 * @returns {string}
 */
export function formatAge(days) {
  const plural = (n, unit) => `${n} ${unit}${n === 1 ? '' : 's'}`
  if (days < 14) return plural(days, 'day')
  if (days < 60) return plural(Math.round(days / 7), 'week')
  return plural(Math.round(days / 30.44), 'month')
}

/**
 * k of n in words: 'never', 'about N in 10' (rate of 30% or more), else 'about 1 in M'.
 * @param {number} k
 * @param {number} n
 * @returns {string}
 */
export function ratioWords(k, n) {
  if (k === 0) return 'never'
  if (k / n >= 0.3) return `about ${Math.round((10 * k) / n)} in 10`
  return `about 1 in ${Math.round(n / k)}`
}

/**
 * A summary statistic with its interval where the API gives one: '0.52 [0.31, 0.65]'.
 * @param {{ value: number, ci: [number, number] | null } | null} stat
 * @param {number} digits
 * @returns {string} '—' for a missing stat
 */
export function formatStat(stat, digits) {
  if (!stat) return '—'
  const v = stat.value.toFixed(digits)
  return stat.ci ? `${v} [${stat.ci[0].toFixed(digits)}, ${stat.ci[1].toFixed(digits)}]` : v
}
