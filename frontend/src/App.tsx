import { useEffect, useMemo, useState } from 'react'
import {
  Activity,
  CloudSun,
  Database,
  Moon,
  Radar,
  SunMedium,
} from 'lucide-react'
import { ALGORITHM_NAMES, getJSON, pct, type StatsResponse } from './lib/api'
import PredictionForm from './components/PredictionForm'
import HistoricalAnalysis from './components/HistoricalAnalysis'
import AirportNetwork from './components/AirportNetwork'
import Dashboard from './components/Dashboard'

type Tab = 'dashboard' | 'predict' | 'history' | 'network'
type Theme = 'light' | 'dark'

function getPreferredTheme(): Theme {
  if (typeof window === 'undefined') return 'light'

  const storedTheme = window.localStorage.getItem('flight-ui-theme')
  if (storedTheme === 'light' || storedTheme === 'dark') return storedTheme

  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

const tabs: Array<{ id: Tab; label: string }> = [
  { id: 'dashboard', label: 'Command Center' },
  { id: 'predict', label: 'Prediction Studio' },
  { id: 'history', label: 'Delay Patterns' },
  { id: 'network', label: 'Route Network' },
]

function buildHighlights(stats: StatsResponse | null) {
  return [
    {
      label: 'Real flights analysed',
      value: stats ? stats.training_rows.toLocaleString() : '...',
      detail: 'US Bureau of Transportation Statistics records, every month of 2024.',
      icon: Database,
    },
    {
      label: 'Flights delayed 15+ min',
      value: stats ? pct(stats.delay_rate) : '...',
      detail: 'Base rate the model has to beat, so accuracy alone is not enough.',
      icon: Activity,
    },
    {
      label: 'Model ROC-AUC',
      value: stats ? stats.metrics.roc_auc.toFixed(2) : '...',
      detail: stats
        ? `${ALGORITHM_NAMES[stats.algorithm] ?? stats.algorithm}, best of 13 runs tracked in MLflow.`
        : 'Loading model metrics...',
      icon: Radar,
    },
  ]
}

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('dashboard')
  const [theme, setTheme] = useState<Theme>(() => getPreferredTheme())
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [apiState, setApiState] = useState<'loading' | 'online' | 'offline'>('loading')

  useEffect(() => {
    getJSON<StatsResponse>('/stats')
      .then((s) => {
        setStats(s)
        setApiState('online')
      })
      .catch(() => {
        setStats(null)
        setApiState('offline')
      })
  }, [])

  const highlights = buildHighlights(stats)

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
    window.localStorage.setItem('flight-ui-theme', theme)
  }, [theme])

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')

    const handleSystemThemeChange = (event: MediaQueryListEvent) => {
      const savedTheme = window.localStorage.getItem('flight-ui-theme')
      if (savedTheme === 'light' || savedTheme === 'dark') return
      setTheme(event.matches ? 'dark' : 'light')
    }

    mediaQuery.addEventListener('change', handleSystemThemeChange)

    return () => mediaQuery.removeEventListener('change', handleSystemThemeChange)
  }, [])

  // real status: whether the prediction API answered and which trained model it serves
  const statusText = useMemo(() => {
    if (apiState === 'loading') return 'Connecting to the prediction API...'
    if (apiState === 'offline') return 'Prediction API not reachable'
    return stats ? `${ALGORITHM_NAMES[stats.algorithm] ?? stats.algorithm} model online` : 'Model online'
  }, [apiState, stats])
  const statusDot = apiState === 'online' ? 'bg-emerald-400 shadow-[0_0_16px_rgba(74,222,128,0.8)]' : apiState === 'offline' ? 'bg-red-500' : 'bg-amber-400'

  return (
   <div className="min-h-screen bg-gray-100 dark:bg-gray-950 transition-all duration-300">
      <div className="mx-auto flex min-h-screen max-w-7xl flex-col px-4 py-5 sm:px-6 lg:px-8">
        <header className="surface-card sticky top-4 z-20 mb-6 overflow-hidden px-4 py-4 sm:px-6">
          <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan-400/60 to-transparent" />
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex items-start gap-3">
              <div className="icon-chip p-3">
                <CloudSun className="h-6 w-6" />
              </div>
              <div>
                <p className="eyebrow">CSL236 MLOps Project</p>
                <h1 className="mt-2 text-2xl font-semibold tracking-tight sm:text-3xl">
                  Flight Delay Predictor
                </h1>
                <p className="subtle-text mt-2 max-w-2xl text-sm sm:text-base">
                  Predicts the chance that a US flight arrives 15+ minutes late, with a model trained
                  on 120,000 real 2024 flights.
                </p>
              </div>
            </div>

            <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
             <button
  type="button"
  onClick={() => setTheme((current) => (current === 'dark' ? 'light' : 'dark'))}
  className="
    flex items-center gap-2
    rounded-xl
    border
    border-gray-300
    dark:border-gray-700
    bg-white/90
    dark:bg-gray-800/90
    px-4
    py-2
    text-sm
    font-medium
    text-gray-900
    dark:text-white
    shadow-md
    backdrop-blur
    transition-all
    duration-300
    hover:scale-105
    hover:shadow-lg
  "
>
  {theme === 'dark' ? (
    <>
      <SunMedium className="h-4 w-4 text-yellow-400" />
      Light Mode
    </>
  ) : (
    <>
      <Moon className="h-4 w-4 text-cyan-500" />
      Dark Mode
    </>
  )}
</button>
              <div className="surface-panel flex items-center gap-3 px-4 py-3">
                <span className={`h-2.5 w-2.5 rounded-full ${statusDot}`} />
                <div>
                  <p className="text-sm font-semibold">API status</p>
                  <p className="subtle-text text-xs">{statusText}</p>
                </div>
              </div>
            </div>
          </div>
        </header>

        <section className="mb-8 grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
          <div className="surface-card relative overflow-hidden p-6 sm:p-8 lg:p-10">
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(56,189,248,0.16),transparent_34%),radial-gradient(circle_at_bottom_right,rgba(245,158,11,0.16),transparent_30%)]" />
            <div className="relative">
              <p className="eyebrow">Overview</p>
              <h2 className="mt-4 max-w-3xl text-4xl font-semibold leading-tight tracking-tight sm:text-5xl">
                Know your flight's delay risk before you fly.
              </h2>
              <p className="subtle-text mt-5 max-w-2xl text-base leading-7 sm:text-lg">
                A machine learning model trained on real US flight records predicts the chance of a
                15+ minute delay, explains which factors drive it, and a local AI model turns that
                into plain-English advice.
              </p>

              <div className="mt-8 flex flex-wrap gap-3">
                <button
                  type="button"
                  onClick={() => setActiveTab('predict')}
                  className="primary-button"
                >
                  Start a prediction
                </button>
                <button
                  type="button"
                  onClick={() => setActiveTab('network')}
                  className="secondary-button"
                >
                  Explore route network
                </button>
              </div>

              <div className="mt-10 grid gap-4 md:grid-cols-3">
                {highlights.map(({ label, value, detail, icon: Icon }) => (
                  <div key={label} className="surface-panel p-4">
                    <div className="flex items-center justify-between">
                      <p className="subtle-text text-sm">{label}</p>
                      <Icon className="accent-icon h-4 w-4" />
                    </div>
                    <p className="mt-3 text-3xl font-semibold tracking-tight">{value}</p>
                    <p className="subtle-text mt-2 text-sm leading-6">{detail}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="surface-card overflow-hidden p-3">
            <div className="hero-frame h-full min-h-[360px]">
              <div className="hero-overlay absolute inset-0 z-10" />
              <div className="absolute -left-10 bottom-0 z-10 h-44 w-44 rounded-full bg-amber-300/20 blur-3xl" />
              <img
                src="/airplane-hero.png"
                alt="Commercial airplane flying above clouds"
                className="h-full w-full object-cover object-center scale-105"
              />
              <div className="absolute inset-x-0 bottom-0 z-20 p-6 sm:p-7">
                <p className="text-xs font-semibold uppercase tracking-[0.24em] text-white/70">
                  How it works
                </p>
                <p className="mt-2 max-w-sm text-2xl font-semibold leading-tight text-white">
                  Trained on real flight data. Tracked with MLflow. Explained by local generative AI.
                </p>
              </div>
            </div>
          </div>
        </section>

        <nav className="mb-8 flex flex-wrap gap-3">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id)}
              className={
                activeTab === tab.id
                  ? 'primary-button'
                  : 'secondary-button'
              }
            >
              {tab.label}
            </button>
          ))}
        </nav>

        <main className="pb-10">
          {activeTab === 'dashboard' && <Dashboard />}
          {activeTab === 'predict' && <PredictionForm theme={theme} />}
          {activeTab === 'history' && <HistoricalAnalysis />}
          {activeTab === 'network' && <AirportNetwork />}
        </main>

        <footer className="ui-border mt-auto border-t px-1 py-6 text-sm">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <p className="subtle-text">
              Predictions come from a Gradient Boosting model trained on US Bureau of Transportation
              Statistics flight records (2024).
            </p>
            <p className="subtle-text">CSL236 end-term project · FastAPI + scikit-learn + MLflow + Ollama</p>
          </div>
        </footer>
      </div>
    </div>
  )
}
