import React from 'react'
import ReactDOM from 'react-dom/client'
import 'leaflet/dist/leaflet.css'
import App from './App.tsx'
import './index.css'

const storedTheme = window.localStorage.getItem('flight-ui-theme')

const prefersDark = window.matchMedia(
  '(prefers-color-scheme: dark)'
).matches

const initialDarkMode = storedTheme
  ? storedTheme === 'dark'
  : prefersDark

document.documentElement.classList.toggle(
  'dark',
  initialDarkMode
)

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)