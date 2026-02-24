import React from 'react'
import { Routes, Route } from 'react-router-dom'
import Navigation from './components/Navigation'
import ErrorBoundary from './components/ErrorBoundary'
import Dashboard from './pages/Dashboard'
import Historical from './pages/Historical'
import About from './pages/About'
import './App.css'

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation />
      <main className="container mx-auto px-4 py-8">
        <ErrorBoundary>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/historical" element={<Historical />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </ErrorBoundary>
      </main>
    </div>
  )
}

export default App