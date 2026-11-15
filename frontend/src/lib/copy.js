import { formatDate, formatPct, formatScore, formatValue, ratioWords } from './format'
import { levelKey } from './risk'

/**
 * A short warning for a site whose range missed often, in plain words.
 * @param {string} site
 * @param {import('../services/contract').Reliability} reliability
 * @param {string} holdoutLabel
 * @returns {string}
 */
export function weakSiteText(site, reliability, holdoutLabel) {
  const { inside, nBanded } = reliability.band
  return `Less reliable at ${site}: in ${holdoutLabel} only ${inside} of ${nBanded} measurements landed inside the likely range.`
}

/**
 * What the alert level means, in plain words. The level is the API's call.
 * @param {'already' | 'quiet' | 'watch' | 'warning' | 'unavailable'} level
 * @param {number} threshold
 * @returns {string}
 */
export function alertPlain(level, threshold) {
  const rise = `A sudden rise to ${threshold} µg/g or more`
  switch (level) {
    case 'quiet':
      return `${rise} looks unlikely.`
    case 'watch':
      return `${rise} is possible.`
    case 'warning':
      return `${rise} is more likely than usual.`
    case 'already':
      return `The last measurement was already at or above ${threshold} µg/g, so this alert does not apply; follow the forecast and the measurements.`
    default:
      return 'No rise alert is available for this week.'
  }
}

/**
 * How often weeks at an alert level were followed by a rise, in plain words.
 * @param {string} levelName
 * @param {{ weeks: number, followed: number }} record
 * @param {string} holdoutLabel
 * @param {number} threshold
 * @returns {string}
 */
export function levelRecordPlain(levelName, { weeks, followed }, holdoutLabel, threshold) {
  const rise = `a rise to ${threshold} µg/g or more`
  if (followed === 0) return `In ${holdoutLabel}, none of the ${weeks} ${levelName} weeks was followed by ${rise}.`
  return `In ${holdoutLabel}, ${followed} of ${weeks} ${levelName} weeks (${ratioWords(followed, weeks)}) were followed by ${rise}.`
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
  return `A sudden rise (or jump): the last value known when the forecast was made was below ${threshold} µg/g and the measurement reached ${threshold} or more.`
}

/**
 * What the classifier score is.
 * @param {number} threshold
 * @returns {string}
 */
export function scoreCaption(threshold) {
  return `classifier score for a jump to ≥ ${threshold} µg/g next week. Not a calibrated chance.`
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

/**
 * Why a bloom Watch is on, in plain words.
 * @param {string} site
 * @param {{ count: number, days: number, lastBelow: number }} rule
 * @returns {string}
 */
export function bloomReasonText(site, rule) {
  return `An algae bloom at ${site}: the algae that make domoic acid reached ${rule.count.toLocaleString('en-US')} cells per litre in the ${rule.days} days before the forecast, while the toxin was still below ${rule.lastBelow} µg/g.`
}

/**
 * The bloom Watch's record before the holdout, from the API.
 * @param {{ start: string, end: string, on: number, followed: number }} record
 * @param {number} threshold
 * @param {{ sites: string[] }} rule
 * @returns {string}
 */
export function bloomRecordText({ start, end, on, followed }, threshold, rule) {
  const span = `${start.slice(0, 4)}–${Number(end.slice(0, 4)) - 1}`
  const after = followed === 0 ? 'none was followed by' : `${followed} of them (${ratioWords(followed, on)}) were followed by`
  return `In ${span} the bloom Watch would have been on in ${on} forecast weeks at ${rule.sites.join(' and ')}; ${after} a sudden rise to ${threshold} µg/g or more. Most blooms are not toxic.`
}

/**
 * Which sudden rises the bloom Watch was on before. The beaches it covers were chosen
 * after seeing this record, so the text says so.
 * @param {{ suddenRises: number, caught: number }} record
 * @param {{ lastBelow: number }} rule
 * @returns {string}
 */
export function bloomCaughtText({ suddenRises, caught }, rule) {
  const share = caught === suddenRises ? `all ${suddenRises}` : `${caught} of the ${suddenRises}`
  return `It was on before ${share} sudden rises from below ${rule.lastBelow} µg/g there. These beaches were picked after seeing that record, and the events are few, so treat it as rough.`
}

/**
 * The bloom condition in the alert's checklist.
 * @param {string} site
 * @param {{ count: number, lastBelow: number }} rule
 * @param {{ counted: boolean, lastCountDate: string | null }} status
 * @param {boolean} holds
 * @returns {string}
 */
export function bloomConditionText(site, rule, status, holds) {
  if (!status.counted) {
    const newest = status.lastCountDate ? ` (newest: ${formatDate(status.lastCountDate)})` : ''
    return `No algae count at ${site} in the week before the forecast${newest}`
  }
  const count = `${rule.count.toLocaleString('en-US')} cells/L`
  return holds ? `Algae count ≥ ${count} with toxin below ${rule.lastBelow}` : `No algae count ≥ ${count} with toxin below ${rule.lastBelow}`
}
