import React, { Suspense, lazy } from 'react'
import { Routes, Route } from 'react-router-dom'
import Navigation from './components/Navigation'
import ErrorBoundary from './components/ErrorBoundary'
import './App.css'

// Pages load on demand so the About page does not wait for the charting code
const Dashboard = lazy(() => import('./pages/Dashboard'))
const Historical = lazy(() => import('./pages/Historical'))
const About = lazy(() => import('./pages/About'))

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation />
      <main className="container mx-auto px-4 py-8">
        <ErrorBoundary>
          <Suspense fallback={<div className="text-gray-500">Loading…</div>}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/historical" element={<Historical />} />
              <Route path="/about" element={<About />} />
            </Routes>
          </Suspense>
        </ErrorBoundary>
      </main>
    </div>
  )
}

export default App