// Fetchers: each returns a Promise of a view model and rejects with Error(detail).
import api from './api'
import {
  toConfig, toForecast, toAllSites, toHistory, toSummary, toTrackRecord,
} from './contract'

async function request(call) {
  try {
    const response = await call()
    return response.data
  } catch (error) {
    const detail = error.response?.data?.detail
    throw new Error(typeof detail === 'string' ? detail : (error.message || 'Request failed'))
  }
}

export async function getConfig() {
  const [rawConfig, rawSites] = await Promise.all([
    request(() => api.get('/api/config')),
    request(() => api.get('/api/sites')),
  ])
  return toConfig(rawConfig, rawSites)
}

/** A failed forecast is HTTP 200 with success: false; it resolves with `error` set. */
export async function getForecast(site, date) {
  const raw = await request(() => api.post('/api/forecast/enhanced', { date, site, task: 'regression', model: 'ensemble' }))
  return toForecast(raw)
}

export async function getAllSites(date) {
  const raw = await request(() => api.get('/api/forecast/all', { params: { date, model: 'ensemble' } }))
  return toAllSites(raw)
}

/** Every real measurement at `site` from `since` (an ISO date) on, with the forecast made for it. */
export async function getSiteHistory(site, since) {
  const raw = await request(() => api.get(`/api/sites/${encodeURIComponent(site)}/history`, { params: { since } }))
  return toHistory(raw)
}

export async function getSummary() {
  return toSummary(await request(() => api.get('/api/metrics/summary')))
}

export async function getTrackRecord(windowName, site) {
  const params = site ? { window: windowName, site } : { window: windowName }
  return toTrackRecord(await request(() => api.get('/api/track-record', { params })))
}
