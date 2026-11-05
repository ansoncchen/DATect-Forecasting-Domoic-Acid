import { describe, expect, it } from 'vitest'
import { TRACK_RECORD_COLUMNS, TRACK_RECORD_HEADERS, toCsv } from './csv'

describe('toCsv', () => {
  it('quotes values with commas or quotes', () => {
    expect(toCsv([{ site: 'Coos Bay', note: 'a "b", c' }], ['site', 'note'])).toBe('site,note\nCoos Bay,"a ""b"", c"')
  })
  it('prints null and undefined as empty', () => {
    expect(toCsv([{ site: 'X', v: null }], ['site', 'v'])).toBe('site,v\nX,')
    expect(toCsv([{ site: 'X' }], ['site', 'v'])).toBe('site,v\nX,')
  })
  it('writes only the header for no rows, and booleans as text', () => {
    expect(toCsv([], ['a', 'b'])).toBe('a,b')
    expect(toCsv([{ a: true, b: 0 }], ['a', 'b'])).toBe('a,b\ntrue,0')
  })
  it('calls the classifier number a score in the track record export', () => {
    const csv = toCsv([{ date: '2023-01-02', probability: 0.61 }], TRACK_RECORD_COLUMNS, TRACK_RECORD_HEADERS)
    const [header, line] = csv.split('\n')
    expect(header.split(',')).toContain('classifier_score')
    expect(header).not.toContain('probability')
    expect(line).toContain('0.61')
  })
})
