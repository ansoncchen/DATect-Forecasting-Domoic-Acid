import { describe, it, expect } from 'vitest'
import {
  trackRecordText, siteRecordText, bandVsAlertText,
  scoreCaption, levelCutoffsText, classifierConditionText, recentValuesText, jumpDefinition,
  alertPlain, levelRecordPlain, weakSiteText,
} from './copy'

const overall = {
  band: { coverage: 0.802, nBanded: 400, inside: 321 },
  onsets: { n: 27, caught: 19 },
  alarms: { n: 106, followed: 19, falseNearMiss: 48 },
}
const newport = { band: { coverage: 0.479, nBanded: 48, inside: 23 } }
const L = '2022–2023'

describe('record sentences', () => {
  it('describes the all-sites track record', () => {
    expect(trackRecordText(overall, L)).toBe(
      '2022–2023, all sites: caught 19 of 27 jumps (about 7 in 10); 19 of 106 alarms were followed by one (about 1 in 6). 48 of the 87 false alarms were near misses at 10–20 µg/g.')
  })
  it('describes a site record', () => {
    expect(siteRecordText('Newport', { n: 6, caught: 5, quiet: 24, quietAlarmed: 8 })).toBe(
      'Newport: 5 of 6 jumps caught; 5 of 13 alarms followed by one.')
    expect(siteRecordText('Kalaloch', { n: 0, caught: 0, quiet: 30, quietAlarmed: 0 })).toBe(
      'Kalaloch: no jumps to ≥ 20; no alarms.')
  })
})

describe('bandVsAlertText', () => {
  it('contrasts the range with the classifier score, with no claim the data does not back', () => {
    const t = bandVsAlertText({ bandPExceed: 0.064, alert: { probability: 0.223, reason: null } }, 20)
    expect(t).toBe(
      "Not the alert. The range puts ≥ 20 at 6%; the alert's classifier score is 0.22, which is not a calibrated chance. The alert keeps the classifier because the range was not shown to catch jumps as well.")
    expect(t).not.toContain('AUC')
  })
  it('takes the threshold from its caller', () => {
    expect(bandVsAlertText({ bandPExceed: 0.5, alert: { probability: 0.2, reason: null } }, 30)).toContain('≥ 30 at 50%')
  })
  it('never prints a missing score as zero', () => {
    expect(bandVsAlertText({ bandPExceed: 0.064, alert: { probability: null, reason: 'no_classifier' } }, 20)).toBe(
      "Not the alert. The range puts ≥ 20 at 6%; the alert's classifier score is unavailable for this week.")
  })
  it('prints no score when the last measurement is already at or above the threshold', () => {
    const t = bandVsAlertText({ bandPExceed: 0.7, alert: { probability: 0.9, reason: 'already_at_or_above_threshold' } }, 20)
    expect(t).toBe('Not the alert. The range puts ≥ 20 at 70%; the alert does not apply, because the last measurement is already at or above 20.')
    expect(t).not.toContain('0.90')
  })
})

describe('score sentences', () => {
  it('captions the score as a score, not a chance', () => {
    expect(scoreCaption(20)).toBe('classifier score for a jump to ≥ 20 µg/g next week. Not a calibrated chance.')
  })
  it('lists the cutoffs from config, leaving out levels with none', () => {
    const levels = [
      { name: 'Quiet', cutoff: null }, { name: 'Watch', cutoff: 0.1 }, { name: 'Warning', cutoff: 0.35 },
    ]
    expect(levelCutoffsText(levels)).toBe('Watch from 0.10, Warning from 0.35')
  })
  it('states the classifier condition from the API, with the score to two decimals', () => {
    expect(classifierConditionText(0.61, 0.1, true)).toBe('Classifier score 0.61 ≥ 0.10')
    expect(classifierConditionText(0.04, 0.1, false)).toBe('Classifier score 0.04 < 0.10')
    expect(classifierConditionText(0.1, 0.1, true)).toBe('Classifier score 0.10 ≥ 0.10')
  })
  it('keeps the API state when two decimals read equal, without printing "0.10 < 0.10"', () => {
    expect(classifierConditionText(0.0996, 0.1, false)).toBe('Classifier score just under 0.10')
    expect(classifierConditionText(0.1004, 0.1, true)).toBe('Classifier score 0.10 ≥ 0.10')
  })
  it('says when there is no score', () => {
    expect(classifierConditionText(null, 0.1, false)).toBe('Classifier score unavailable')
  })
})

describe('recentValuesText', () => {
  it('says how many and in what order', () => {
    expect(recentValuesText([0, 0, 15, 0, 6.3])).toBe('Last five, oldest to newest: 0.0, 0.0, 15.0, 0.0, 6.3 µg/g')
    expect(recentValuesText([1, 2])).toBe('Last two, oldest to newest: 1.0, 2.0 µg/g')
  })
  it('has no order to state for one value', () => {
    expect(recentValuesText([6.3])).toBe('Last one: 6.3 µg/g')
  })
})

describe('jumpDefinition', () => {
  it('defines a jump from the threshold', () => {
    expect(jumpDefinition(20)).toBe(
      'A sudden rise (or jump): the last value known when the forecast was made was below 20 µg/g and the measurement reached 20 or more.')
  })
})

describe('plain-language helpers', () => {
  it('says what each alert level means without calling the score a chance', () => {
    expect(alertPlain('quiet', 20)).toBe('A sudden rise to 20 µg/g or more looks unlikely.')
    expect(alertPlain('watch', 20)).toBe('A sudden rise to 20 µg/g or more is possible.')
    expect(alertPlain('warning', 20)).toBe('A sudden rise to 20 µg/g or more is more likely than usual.')
    expect(alertPlain('already', 20)).toMatch(/^The last measurement was already at or above 20 µg\/g/)
    expect(alertPlain('unavailable', 20)).toBe('No rise alert is available for this week.')
    for (const level of ['quiet', 'watch', 'warning', 'already', 'unavailable']) {
      expect(alertPlain(level, 20)).not.toMatch(/chance|probab|%/)
    }
  })
  it('states a level record as counts with a rate in words', () => {
    expect(levelRecordPlain('Quiet', { weeks: 213, followed: 8 }, L, 20)).toBe(
      'In 2022–2023, 8 of 213 Quiet weeks (about 1 in 27) were followed by a rise to 20 µg/g or more.')
    expect(levelRecordPlain('Warning', { weeks: 5, followed: 0 }, L, 20)).toBe(
      'In 2022–2023, none of the 5 Warning weeks was followed by a rise to 20 µg/g or more.')
  })
  it('flags a weak site with its counts', () => {
    expect(weakSiteText('Newport', newport, L)).toBe(
      'Less reliable at Newport: in 2022–2023 only 23 of 48 measurements landed inside the likely range.')
  })
})
