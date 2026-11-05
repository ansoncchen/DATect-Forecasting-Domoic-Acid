import { formatPct, ratioWords } from './format'
import { siteReliability } from './risk'

/**
 * Sentence on how often the range held at a site.
 * @param {string} site
 * @param {import('../services/contract').Reliability | undefined} reliability
 * @param {import('../services/contract').Reliability} overall
 * @param {string} holdoutLabel e.g. '2022–2023'
 * @returns {string}
 */
export function reliabilityText(site, reliability, overall, holdoutLabel) {
  const all = formatPct(overall.band.coverage)
  const kind = siteReliability(reliability)
  if (kind === 'too-few') {
    const n = reliability?.band?.nBanded ?? 0
    return `Only ${n} measurements here in ${holdoutLabel}, too few to check the range at this site (all sites: ${all}).`
  }
  const { inside, nBanded, coverage } = reliability.band
  const held = `In ${holdoutLabel} the range held ${inside} of ${nBanded} measurements at ${site} (${formatPct(coverage)}`
  if (kind === 'weak') {
    return `Less reliable here. ${held}), against ${all} across all sites. Expect values outside it.`
  }
  return `${held}; all sites ${all}).`
}

/**
 * @param {string} levelName
 * @param {{ weeks: number, followed: number }} record
 * @param {string} holdoutLabel
 * @returns {string}
 */
export function levelRecordText(levelName, { weeks, followed }, holdoutLabel) {
  return `${levelName} weeks in ${holdoutLabel} were followed by a jump ${followed} of ${weeks} times (${ratioWords(followed, weeks)}).`
}

/**
 * All-sites record of the alert on jumps to 20 or more.
 * @param {import('../services/contract').Reliability} overall
 * @param {string} holdoutLabel
 * @returns {string}
 */
export function trackRecordText(overall, holdoutLabel) {
  const { onsets, alarms } = overall
  const falseAlarms = alarms.n - alarms.followed
  return `${holdoutLabel}, all sites: caught ${onsets.caught} of ${onsets.n} jumps (${ratioWords(onsets.caught, onsets.n)}); ${alarms.followed} of ${alarms.n} alarms were followed by one (${ratioWords(alarms.followed, alarms.n)}). ${alarms.falseNearMiss} of the ${falseAlarms} false alarms were near misses at 10–20 µg/g.`
}

/**
 * One site's jumps caught and alarms followed.
 * @param {string} site
 * @param {{ n: number, caught: number, quietAlarmed: number }} onsets
 * @returns {string}
 */
export function siteRecordText(site, onsets) {
  const alarms = onsets.caught + onsets.quietAlarmed
  const jumps = onsets.n ? `${onsets.caught} of ${onsets.n} jumps caught` : 'no jumps to ≥ 20'
  const followed = alarms ? `${onsets.caught} of ${alarms} alarms followed by one` : 'no alarms'
  return `${site}: ${jumps}; ${followed}.`
}

/**
 * Why the range's chance of 20 or more differs from the alert's probability.
 * @param {{ bandPExceed: number, alert: { probability: number } }} forecast
 * @returns {string}
 */
export function bandVsAlertText(forecast) {
  return `Not the alert. The range puts ≥ 20 at ${formatPct(forecast.bandPExceed)}; the alert's classifier says ${formatPct(forecast.alert.probability)}. The classifier is trained to catch jumps and ranked them better in 2022–2023 (AUC 0.77 vs 0.75), so the alert uses it.`
}
