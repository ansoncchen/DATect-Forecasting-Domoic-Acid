import { afterEach, describe, it, expect, vi } from 'vitest'
import api from './api'
import { getSiteHistory } from './datect'

afterEach(() => vi.restoreAllMocks())

describe('getSiteHistory', () => {
  it('asks for history from the date it is given, so weeks before 2019 have a chart', async () => {
    const get = vi.spyOn(api, 'get').mockResolvedValue({ data: { site: 'Coos Bay', rows: [] } })
    await getSiteHistory('Coos Bay', '2003-01-06')
    expect(get).toHaveBeenCalledWith('/api/sites/Coos%20Bay/history', { params: { since: '2003-01-06' } })
  })
})
