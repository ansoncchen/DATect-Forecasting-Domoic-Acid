import { describe, it, expect } from 'vitest'
import {
  reliabilityText, levelRecordText, trackRecordText, siteRecordText, bandVsAlertText,
  scoreCaption, levelCutoffsText, classifierConditionText,
} from './copy'

const overall = {
  band: { coverage: 0.802, nBanded: 400, inside: 321 },
  onsets: { n: 27, caught: 19 },
  alarms: { n: 106, followed: 19, falseNearMiss: 48 },
}
const newport = { band: { coverage: 0.479, nBanded: 48, inside: 23 } }
const L = '2022–2023'

describe('reliabilityText', () => {
  it('warns when the range is weak at a site', () => {
    expect(reliabilityText('Newport', newport, overall, L)).toBe(
      'Less reliable here. In 2022–2023 the range held 23 of 48 measurements at Newport (48%), against 80% across all sites. Expect values outside it.')
  })
  it('reports a site where the range held', () => {
    const ok = { band: { coverage: 0.75, nBanded: 40, inside: 30 } }
    expect(reliabilityText('Copalis', ok, overall, L)).toBe(
      'In 2022–2023 the range held 30 of 40 measurements at Copalis (75%; all sites 80%).')
  })
  it('says there are too few measurements', () => {
    const few = { band: { coverage: 1, nBanded: 2, inside: 2 } }
    const msg = 'Only 2 measurements here in 2022–2023, too few to check the range at this site (all sites: 80%).'
    expect(reliabilityText('Kalaloch', few, overall, L)).toBe(msg)
    expect(reliabilityText('Kalaloch', undefined, overall, L)).toMatch(/^Only 0 measurements/)
  })
})

describe('record sentences', () => {
  it('describes a level record', () => {
    expect(levelRecordText('Watch', { weeks: 68, followed: 9 }, L)).toBe(
      'Watch weeks in 2022–2023 were followed by a jump 9 of 68 times (about 1 in 8).')
  })
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
    expect(scoreCaption(20, true)).toBe(
      'classifier score for a jump to ≥ 20 µg/g next week. Not a calibrated chance; the line below says what weeks at this level did.')
  })
  it('points at no line that is not there', () => {
    expect(scoreCaption(20, false)).toBe('classifier score for a jump to ≥ 20 µg/g next week. Not a calibrated chance.')
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
