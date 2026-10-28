import React from 'react'

const About = () => {
  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <h1 className="text-3xl font-bold text-gray-900">About DATect</h1>
      
      <div className="bg-white rounded-lg shadow p-6 space-y-4">
        <h2 className="text-xl font-semibold text-gray-800">Domoic Acid Forecasting System</h2>
        <p className="text-gray-600">
          DATect is a scientific machine learning system for forecasting harmful algal bloom concentrations
          (domoic acid) using satellite oceanographic data and environmental measurements.
        </p>
        
        <div className="grid md:grid-cols-2 gap-6 mt-6">
          <div>
            <h3 className="font-semibold text-gray-800">Key Features</h3>
            <ul className="list-disc list-inside text-gray-600 space-y-1">
              <li>21 years of temporal coverage (2003-2023)</li>
              <li>10 Pacific Coast monitoring sites</li>
              <li>XGBoost + Random Forest ensemble with per-site configuration</li>
              <li>Real-time and retrospective analysis</li>
              <li>Scientific data visualization</li>
            </ul>
          </div>
          
          <div>
            <h3 className="font-semibold text-gray-800">Data Sources</h3>
            <ul className="list-disc list-inside text-gray-600 space-y-1">
              <li>MODIS satellite and anomaly</li>
              <li>Climate indices (PDO, ONI, BEUTI)</li>
              <li>USGS streamflow measurements</li>
              <li>Shellfish toxin monitoring data</li>
              <li>Pseudo-nitzschia cell counts</li>
            </ul>
          </div>
        </div>
        
        <div className="mt-6">
          <h3 className="font-semibold text-gray-800">Scientific Validation</h3>
          <p className="text-gray-600">
            The system implements strict temporal safeguards to prevent data leakage and maintains
            peer-review quality standards with comprehensive validation testing.
          </p>
        </div>
      </div>
    </div>
  )
}

export default About