import { useEffect, useState } from 'react'
import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import { AlertCircle, Database, Gauge, Radar, Sparkles } from 'lucide-react'
import { ALGORITHM_NAMES, getJSON, pct, type StatsResponse } from '../lib/api'

const COLORS = ['#0f766e', '#38bdf8', '#0891b2', '#f59e0b', '#f97316']

export default function Dashboard() {
  const [data, setData] = useState<StatsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    getJSON<StatsResponse>('/stats')
      .then(setData)
      .catch(() => setError('Could not reach the prediction API. Is the backend running?'))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="surface-card p-8 text-center">
        <p className="subtle-text">Loading dashboard...</p>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="error-panel p-4">
        <div className="flex gap-3">
          <AlertCircle className="mt-0.5 h-5 w-5 shrink-0" />
          <p>{error}</p>
        </div>
      </div>
    )
  }

  const bucketData = Object.entries(data.delay_buckets).map(([name, share]) => ({
    name,
    value: Number((share * 100).toFixed(1)),
  }))

  const hourData = Object.entries(data.by_hour)
    .filter(([, s]) => s.flights >= 100) // skip night hours with too few flights to be reliable
    .map(([hour, s]) => ({ hour: `${hour.padStart(2, '0')}:00`, rate: Number((s.rate * 100).toFixed(1)) }))

  const cards = [
    {
      label: 'Flights delayed',
      value: pct(data.delay_rate, 1),
      detail: `of ${data.training_rows.toLocaleString()} real flights arrived 15+ min late`,
      icon: Gauge,
    },
    {
      label: 'Model ROC-AUC',
      value: data.metrics.roc_auc.toFixed(3),
      detail: `${ALGORITHM_NAMES[data.algorithm] ?? data.algorithm}, measured on unseen test flights`,
      icon: Radar,
    },
    {
      label: 'Delays caught (recall)',
      value: pct(data.metrics.recall),
      detail: `F1 score ${data.metrics.f1.toFixed(3)} on the 20% test split`,
      icon: Sparkles,
    },
    {
      label: 'Predictions served',
      value: data.predictions_served.toLocaleString(),
      detail: 'Logged requests (used for drift checks)',
      icon: Database,
    },
  ]

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {cards.map(({ label, value, detail, icon: Icon }) => (
          <div key={label} className="metric-card">
            <div className="flex items-center justify-between">
              <p className="subtle-text text-sm">{label}</p>
              <div className="icon-chip p-2">
                <Icon className="h-4 w-4" />
              </div>
            </div>
            <p className="mt-5 text-4xl font-semibold tracking-tight">{value}</p>
            <p className="subtle-text mt-2 text-sm">{detail}</p>
          </div>
        ))}
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <div className="surface-card p-6">
          <div className="mb-5">
            <p className="eyebrow">Distribution</p>
            <h3 className="mt-3 section-title">How late flights arrive</h3>
            <p className="subtle-text mt-2 text-sm">Share of all flights in the training data.</p>
          </div>
          <ResponsiveContainer width="100%" height={320}>
            <PieChart>
              <Pie
                data={bucketData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={(entry) => `${entry.value}%`}
                outerRadius={108}
                dataKey="value"
              >
                {bucketData.map((entry, index) => (
                  <Cell key={entry.name} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip formatter={(value, name) => [`${value}%`, name]} />
            </PieChart>
          </ResponsiveContainer>
          <div className="mt-2 flex flex-wrap justify-center gap-3 text-xs">
            {bucketData.map((entry, index) => (
              <span key={entry.name} className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full" style={{ background: COLORS[index % COLORS.length] }} />
                {entry.name}
              </span>
            ))}
          </div>
        </div>

        <div className="surface-card p-6">
          <div className="mb-5">
            <p className="eyebrow">Trendline</p>
            <h3 className="mt-3 section-title">Delay rate by departure time</h3>
            <p className="subtle-text mt-2 text-sm">% of flights arriving 15+ minutes late.</p>
          </div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={hourData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.22)" />
              <XAxis dataKey="hour" stroke="#94a3b8" tickLine={false} axisLine={false} interval={2} />
              <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} unit="%" />
              <Tooltip formatter={(value) => [`${value}%`, 'Delayed']} />
              <Bar dataKey="rate" fill="#0891b2" radius={[10, 10, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
        <div className="surface-card p-6">
          <p className="eyebrow">Model Card</p>
          <h3 className="mt-3 section-title">How the prediction works</h3>
          <p className="subtle-text mt-4 max-w-2xl leading-7">
            A {ALGORITHM_NAMES[data.algorithm] ?? data.algorithm} classifier was trained on{' '}
            {data.training_rows.toLocaleString()} real flights from the US Bureau of Transportation
            Statistics (all 12 months of 2024). It uses the airline, route, month, day, departure
            hour and distance to estimate the probability that a flight arrives 15+ minutes late.
            Thirteen runs with six algorithms, two feature sets, hyperparameter tuning and 5-fold cross-validation were compared in MLflow; this model had the best ROC-AUC and is registered as the production model.
          </p>
          <div className="mt-5 grid grid-cols-2 gap-3 text-sm sm:grid-cols-5">
            {Object.entries(data.metrics).map(([name, value]) => (
              <div key={name} className="surface-panel px-3 py-2 text-center">
                <p className="subtle-text text-xs uppercase">{name.replace('_', '-')}</p>
                <p className="mt-1 font-semibold">{value.toFixed(3)}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="surface-card p-6">
          <p className="eyebrow">Health</p>
          <h3 className="mt-3 section-title">Platform readiness</h3>
          <div className="mt-5 space-y-4">
            <div className="surface-panel flex items-center justify-between px-4 py-3">
              <span className="subtle-text text-sm">API status</span>
              <span className="font-semibold text-emerald-500">Online</span>
            </div>
            <div className="surface-panel flex items-center justify-between px-4 py-3">
              <span className="subtle-text text-sm">Model loaded</span>
              <span className="font-semibold text-emerald-500">
                {ALGORITHM_NAMES[data.algorithm] ?? data.algorithm}
              </span>
            </div>
            <div className="surface-panel flex items-center justify-between px-4 py-3">
              <span className="subtle-text text-sm">Local AI ({data.llm.model})</span>
              <span className={`font-semibold ${data.llm.available ? 'text-emerald-500' : 'text-amber-500'}`}>
                {data.llm.available ? 'Available' : 'Offline (template fallback)'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
