import React from 'react'

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  componentDidCatch(error, errorInfo) {
    console.error('UI error boundary caught an error:', error, errorInfo)
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null })
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="max-w-3xl mx-auto bg-red-50 border border-red-200 rounded-lg p-6 text-red-900">
          <h2 className="text-lg font-semibold mb-2">Something went wrong</h2>
          <p className="text-sm mb-4">
            The forecasting UI hit an unexpected error. You can try again or reload the page.
          </p>
          <div className="flex gap-3">
            <button
              onClick={this.handleReset}
              className="bg-red-600 text-white px-4 py-2 rounded-md hover:bg-red-700"
            >
              Try Again
            </button>
            <button
              onClick={() => window.location.reload()}
              className="bg-white text-red-700 px-4 py-2 rounded-md border border-red-300 hover:bg-red-100"
            >
              Reload
            </button>
          </div>
          {this.state.error && (
            <pre className="mt-4 text-xs text-red-800 whitespace-pre-wrap">
              {String(this.state.error)}
            </pre>
          )}
        </div>
      )
    }

    return this.props.children
  }
}

export default ErrorBoundary
