import axios from 'axios'

// In production (built site served by the backend), default to same-origin
const isProd = import.meta.env.PROD
const API_BASE_URL = import.meta.env.VITE_API_URL || (isProd ? '' : 'http://localhost:8000')

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 600000, // Increased to 10 minutes for large retrospective analysis (500+ anchors)
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Log API requests in development only
    if (import.meta.env.DEV) {
      console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`)
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => {
    // Log API responses in development only
    if (import.meta.env.DEV) {
      console.log(`API Response: ${response.status} ${response.config.url}`)
    }
    return response
  },
  (error) => {
    // Log API errors in development only
    if (import.meta.env.DEV) {
      console.error('API Error:', error.response?.data || error.message)
    }
    return Promise.reject(error)
  }
)

export default api