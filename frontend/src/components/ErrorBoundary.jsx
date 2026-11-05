import React from 'react'
import Panel from './ui/Panel'

const BUTTON = 'px-3 py-1.5 border border-gray-300 rounded text-sm bg-white hover:bg-gray-50'

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
        <Panel className="p-5">
          <h2 className="text-base font-semibold">Something went wrong</h2>
          <p className="text-sm text-gray-700 mt-1">
            The dashboard hit an unexpected error. You can try again or reload the page.
          </p>
          <div className="flex gap-2 mt-3">
            <button type="button" onClick={this.handleReset} className={BUTTON}>
              Try again
            </button>
            <button type="button" onClick={() => window.location.reload()} className={BUTTON}>
              Reload
            </button>
          </div>
          {this.state.error && (
            <pre className="mt-3 text-xs text-gray-500 whitespace-pre-wrap break-words">{String(this.state.error)}</pre>
          )}
        </Panel>
      )
    }

    return this.props.children
  }
}

export default ErrorBoundary
