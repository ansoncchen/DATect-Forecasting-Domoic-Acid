import { formatPct, formatScore, formatValue, ratioWords } from './format'
import { levelKey, siteReliability } from './risk'

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

const COUNT_WORDS = ['one', 'two', 'three', 'four', 'five']

/**
 * The last few measurements, in words a reader cannot take for newest first.
 * @param {number[]} values oldest first, at most five
 * @returns {string}
 */
export function recentValuesText(values) {
  const list = `${values.map((v) => formatValue(v)).join(', ')} µg/g`
  return values.length === 1 ? `Last one: ${list}` : `Last ${COUNT_WORDS[values.length - 1]}, oldest to newest: ${list}`
}

/**
 * @param {number} threshold
 * @returns {string}
 */
export function jumpDefinition(threshold) {
  return `A jump: the last value known when the forecast was made was below ${threshold} and the measurement reached ${threshold}.`
}

/**
 * What the score beside the big number is.
 * @param {number} threshold
 * @param {boolean} hasRecord whether the card shows the line on what weeks at this level did
 * @returns {string}
 */
export function scoreCaption(threshold, hasRecord) {
  const pointer = hasRecord ? '; the line below says what weeks at this level did' : ''
  return `classifier score for a jump to ≥ ${threshold} µg/g next week. Not a calibrated chance${pointer}.`
}

/**
 * The classifier score each level starts from, for the levels that have a cutoff.
 * @param {{ name: string, cutoff: number | null }[]} levels
 * @returns {string}
 */
export function levelCutoffsText(levels) {
  return levels
    .filter((l) => l.cutoff != null)
    .map((l) => `${l.name} from ${formatScore(l.cutoff)}`)
    .join(', ')
}

/**
 * The classifier's alert condition. Whether it holds is the API's call; when two decimals
 * read equal the wording keeps that call instead of printing "0.10 < 0.10".
 * @param {number | null} score
 * @param {number} cutoff
 * @param {boolean} holds
 * @returns {string}
 */
export function classifierConditionText(score, cutoff, holds) {
  if (score == null) return 'Classifier score unavailable'
  const [s, c] = [formatScore(score), formatScore(cutoff)]
  if (holds) return `Classifier score ${s} ≥ ${c}`
  return s === c ? `Classifier score just under ${c}` : `Classifier score ${s} < ${c}`
}

/**
 * Why the range's chance of the threshold or more differs from the alert's classifier score.
 * @param {{ bandPExceed: number, alert: { probability: number | null, reason: string | null } }} forecast
 * @param {number} threshold
 * @returns {string}
 */
export function bandVsAlertText(forecast, threshold) {
  const { bandPExceed, alert } = forecast
  const range = `Not the alert. The range puts ≥ ${threshold} at ${formatPct(bandPExceed)}`
  if (levelKey(alert) === 'already') {
    return `${range}; the alert does not apply, because the last measurement is already at or above ${threshold}.`
  }
  if (alert.probability == null) return `${range}; the alert's classifier score is unavailable for this week.`
  return `${range}; the alert's classifier score is ${formatScore(alert.probability)}, which is not a calibrated chance. The alert keeps the classifier because the range was not shown to catch jumps as well.`
}
