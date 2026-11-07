import { describe, it, expect } from 'vitest'
import { windowLabel, formatValue, formatPct, formatScore, formatDate, daysBetween, formatAge, ratioWords } from './format'

describe('windowLabel', () => {
  it('uses an en dash and the inclusive end year', () => {
    expect(windowLabel('2022-01-01', '2024-01-01')).toBe('2022–2023')
    expect(windowLabel('2019-01-01', '2022-01-01')).toBe('2019–2021')
  })
})

describe('formatters', () => {
  it('prints values to one decimal and tiny positives as <0.1', () => {
    expect(formatValue(0)).toBe('0.0')
    expect(formatValue(0.05)).toBe('<0.1')
    expect(formatValue(17.46)).toBe('17.5')
  })
  it('prints whole-number percentages', () => {
    expect(formatPct(0.802)).toBe('80%')
  })
  it('prints a classifier score to two decimals, never as a percentage', () => {
    expect(formatScore(0.6123)).toBe('0.61')
    expect(formatScore(0.1)).toBe('0.10')
    expect(formatScore(0)).toBe('0.00')
    expect(formatScore(1)).toBe('1.00')
  })
  it('prints a missing classifier score as a dash', () => {
    expect(formatScore(null)).toBe('—')
    expect(formatScore(undefined)).toBe('—')
  })
  it('does not shift dates in US time zones', () => {
    expect(formatDate('2023-11-28')).toBe('28 Nov 2023')
  })
  it('counts calendar days', () => {
    expect(daysBetween('2023-10-30', '2023-11-20')).toBe(21)
  })
  it('picks a natural unit for ages', () => {
    expect(formatAge(1)).toBe('1 day')
    expect(formatAge(5)).toBe('5 days')
    expect(formatAge(20)).toBe('20 days')
    expect(formatAge(22)).toBe('22 days')
    expect(formatAge(29)).toBe('29 days')
    expect(formatAge(30)).toBe('4 weeks')
    expect(formatAge(59)).toBe('8 weeks')
    expect(formatAge(60)).toBe('2 months')
    expect(formatAge(475)).toBe('16 months')
  })
  it('puts ratios in words', () => {
    expect(ratioWords(19, 27)).toBe('about 7 in 10')
    expect(ratioWords(19, 106)).toBe('about 1 in 6')
    expect(ratioWords(9, 68)).toBe('about 1 in 8')
    expect(ratioWords(0, 213)).toBe('never')
  })
})
