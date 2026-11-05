import React, { Suspense, lazy } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Navigation from './components/Navigation'
import ErrorBoundary from './components/ErrorBoundary'
import DataCurrencyBanner from './components/ui/DataCurrencyBanner'
import { ConfigProvider, useConfig } from './context/ConfigContext'

// Pages load on demand so the About page does not wait for the charting code
const Forecast = lazy(() => import('./pages/Forecast'))
const TrackRecord = lazy(() => import('./pages/TrackRecord'))
const DataExplorer = lazy(() => import('./pages/DataExplorer'))
const About = lazy(() => import('./pages/About'))

function Pages() {
  const { config, error } = useConfig()
  if (error) {
    return <div className="text-sm text-gray-800">Could not load settings from the API: {error}</div>
  }
  if (!config) return <div className="text-gray-500">Loading…</div>
  return (
    <Suspense fallback={<div className="text-gray-500">Loading…</div>}>
      <Routes>
        <Route path="/" element={<Forecast />} />
        <Route path="/track-record" element={<TrackRecord />} />
        <Route path="/data" element={<DataExplorer />} />
        <Route path="/about" element={<About />} />
        <Route path="/historical" element={<Navigate to="/data" replace />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Suspense>
  )
}

function App() {
  return (
    <ConfigProvider>
      <div className="min-h-screen bg-gray-50 text-gray-900">
        <Navigation />
        <main className="max-w-6xl mx-auto px-4 py-5 space-y-5">
          <DataCurrencyBanner />
          <ErrorBoundary>
            <Pages />
          </ErrorBoundary>
        </main>
      </div>
    </ConfigProvider>
  )
}

export default App
