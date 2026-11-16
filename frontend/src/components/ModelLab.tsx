import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { AlertCircle, ArrowDown, ArrowRight, CheckCircle2 } from 'lucide-react'
import { getJSON, type ExperimentRun, type ExperimentsResponse } from '../lib/api'

// Everything on this page comes from model/experiments.json, exported from MLflow by compare_models.py.

const FEATURE_LABELS: Record<string, string> = {
  dep_hour: 'Departure hour',
  month: 'Month',
  day_of_month: 'Day of month',
  airline: 'Airline',
  dest: 'Destination airport',
  day_of_week: 'Day of week',
  days_to_holiday: 'Days to holiday',
  origin: 'Origin airport',
  distance: 'Distance',
}
const ENGINEERED = new Set(['day_of_month', 'days_to_holiday'])

const TEAL = '#0d9488'
const GREY = '#94a3b8'
const axisColor = '#94a3b8'

const PIPELINE = [
  { step: 'Data', detail: 'BTS 2024, 6.97 M flights, 120k sample' },
  { step: 'Features', detail: 'date + days to US holiday' },
  { step: 'Training', detail: '6 algorithms, 13 MLflow runs' },
  { step: 'Registry', detail: 'best model marked production' },
  { step: 'API', detail: 'FastAPI + data contract' },
  { step: 'Website', detail: 'forecast, reasons, safer options' },
]
const QUALITY = ['23 automated tests', 'Docker image', 'GitHub Actions CI', 'Drift monitoring']

const fmt = (x: number) => x.toFixed(3)
const AUC_TICKS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]

export default function ModelLab() {
  const [data, setData] = useState<ExperimentsResponse | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    getJSON<ExperimentsResponse>('/experiments')
      .then(setData)
      .catch(() => setError('Could not load the experiments. Run `python compare_models.py` and restart the API.'))
  }, [])

  if (error) {
    return (
      <div className="error-panel p-4">
        <div className="flex gap-3">
          <AlertCircle className="mt-0.5 h-5 w-5 shrink-0" />
          <p>{error}</p>
        </div>
      </div>
    )
  }
  if (!data) {
    return (
      <div className="surface-card p-8 text-center">
        <p className="subtle-text">Loading experiments...</p>
      </div>
    )
  }

  const runs = data.runs
  const selected = runs.find((r) => r.selected) ?? runs.reduce((a, b) => (b.roc_auc > a.roc_auc ? b : a))
  const cvRun = runs.find((r) => r.cv_mean !== null)
  const families = new Set(runs.map((r) => r.code)).size

  // best run of each algorithm (runs with the date features where available)
  const bestByAlgorithm = Object.values(
    runs.reduce<Record<string, ExperimentRun>>((acc, r) => {
      if (!acc[r.code] || r.roc_auc > acc[r.code].roc_auc) acc[r.code] = r
      return acc
    }, {}),
  )
    .sort((a, b) => b.roc_auc - a.roc_auc)
    .map((r) => ({ name: r.algorithm.replace(' (MLP)', ''), roc_auc: r.roc_auc, f1: r.f1, isBest: r.code === selected.code }))

  // effect of the date features: best run per algorithm with and without them
  const ablation = ['logreg', 'rf', 'gb']
    .map((code) => {
      const best = (fs: string) => Math.max(0, ...runs.filter((r) => r.code === code && r.features === fs).map((r) => r.roc_auc))
      const name = runs.find((r) => r.code === code)?.algorithm ?? code
      return { name, base: best('base'), date: best('date') }
    })
    .filter((r) => r.base > 0 && r.date > 0)

  const importance = [...data.feature_importance]
    .sort((a, b) => b.importance - a.importance)
    .map((f) => ({ name: FEATURE_LABELS[f.feature] ?? f.feature, importance: f.importance, engineered: ENGINEERED.has(f.feature) }))

  return (
    <div className="space-y-6">
      <div className="surface-card p-6">
        <p className="eyebrow">Model Lab</p>
        <h2 className="mt-3 text-3xl font-semibold tracking-tight">How the model was chosen</h2>
        <p className="subtle-text mt-4 max-w-3xl text-sm leading-7">
          {runs.length} training runs of {families} algorithms were tracked in MLflow on the same 80/20 split of 120,000
          real 2024 flights. {selected.algorithm} had the highest ROC-AUC and is the model this website uses.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="metric-card">
          <p className="subtle-text text-sm">Selected model</p>
          <p className="mt-2 text-2xl font-semibold tracking-tight">{selected.algorithm}</p>
          <p className="subtle-text mt-1 text-xs">{selected.settings}</p>
        </div>
        <div className="metric-card">
          <p className="subtle-text text-sm">ROC-AUC on 24,000 test flights</p>
          <p className="mt-2 text-3xl font-semibold tracking-tight">{fmt(selected.roc_auc)}</p>
          <p className="subtle-text mt-1 text-xs">F1 {fmt(selected.f1)} · recall {fmt(selected.recall)}</p>
        </div>
        <div className="metric-card">
          <p className="subtle-text text-sm">5-fold cross-validation</p>
          <p className="mt-2 text-3xl font-semibold tracking-tight">
            {cvRun ? `${fmt(cvRun.cv_mean ?? 0)} ± ${fmt(cvRun.cv_std ?? 0)}` : 'n/a'}
          </p>
          <p className="subtle-text mt-1 text-xs">stable across different splits</p>
        </div>
        <div className="metric-card">
          <p className="subtle-text text-sm">Experiments</p>
          <p className="mt-2 text-3xl font-semibold tracking-tight">{runs.length} runs</p>
          <p className="subtle-text mt-1 text-xs">{families} algorithms · 2 feature sets</p>
        </div>
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <div className="surface-card p-6">
          <h3 className="section-title">Best run of each algorithm</h3>
          <p className="subtle-text mt-1 text-sm">
            ROC-AUC on the test set. The axis starts at 0.5, the score of random guessing.
          </p>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={bestByAlgorithm} layout="vertical" margin={{ left: 40, right: 40, top: 12 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.22)" horizontal={false} />
              <XAxis type="number" domain={[0.5, 0.75]} ticks={AUC_TICKS} stroke={axisColor} tickLine={false} axisLine={false} />
              <YAxis dataKey="name" type="category" width={150} stroke={axisColor} tickLine={false} axisLine={false} fontSize={12} />
              <Tooltip formatter={(v: unknown, k: unknown) => [Number(v).toFixed(3), k === 'roc_auc' ? 'ROC-AUC' : 'F1']} />
              <Bar dataKey="roc_auc" radius={[0, 8, 8, 0]} label={{ position: 'right', fill: axisColor, fontSize: 12, formatter: (v: unknown) => Number(v).toFixed(3) }}>
                {bestByAlgorithm.map((r) => (
                  <Cell key={r.name} fill={r.isBest ? TEAL : GREY} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="surface-card p-6">
          <h3 className="section-title">What matters most to the model</h3>
          <p className="subtle-text mt-1 text-sm">
            Drop in ROC-AUC when a feature is shuffled. Teal = the two date features we engineered.
          </p>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={importance} layout="vertical" margin={{ left: 40, right: 40, top: 12 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.22)" horizontal={false} />
              <XAxis type="number" stroke={axisColor} tickLine={false} axisLine={false} />
              <YAxis dataKey="name" type="category" width={150} stroke={axisColor} tickLine={false} axisLine={false} fontSize={12} />
              <Tooltip formatter={(v: unknown) => [Number(v).toFixed(3), 'Importance']} />
              <Bar dataKey="importance" radius={[0, 8, 8, 0]} label={{ position: 'right', fill: axisColor, fontSize: 12, formatter: (v: unknown) => Number(v).toFixed(3) }}>
                {importance.map((f) => (
                  <Cell key={f.name} fill={f.engineered ? TEAL : GREY} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {ablation.length > 0 && (
        <div className="surface-card p-6">
          <h3 className="section-title">Effect of the date features</h3>
          <p className="subtle-text mt-1 text-sm">
            Same algorithm, with and without day of month and days to the nearest holiday. Tree models gain; the
            linear model cannot use the up-and-down holiday pattern.
          </p>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={ablation} margin={{ top: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.22)" vertical={false} />
              <XAxis dataKey="name" stroke={axisColor} tickLine={false} axisLine={false} fontSize={12} />
              <YAxis domain={[0.5, 0.75]} ticks={AUC_TICKS} stroke={axisColor} tickLine={false} axisLine={false} />
              <ReferenceLine y={0.5} stroke={axisColor} />
              <Tooltip formatter={(v: unknown, k: unknown) => [Number(v).toFixed(3), k === 'base' ? 'Schedule only' : 'With date features']} />
              <Legend formatter={(v: string) => (v === 'base' ? 'Schedule features only' : 'With date features')} />
              <Bar dataKey="base" fill={GREY} radius={[6, 6, 0, 0]} label={{ position: 'top', fill: axisColor, fontSize: 11, formatter: (v: unknown) => Number(v).toFixed(3) }} />
              <Bar dataKey="date" fill={TEAL} radius={[6, 6, 0, 0]} label={{ position: 'top', fill: axisColor, fontSize: 11, formatter: (v: unknown) => Number(v).toFixed(3) }} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="surface-card p-6">
        <h3 className="section-title">The MLOps pipeline</h3>
        <p className="subtle-text mt-1 text-sm">From raw government data to this website, every step is scripted and repeatable.</p>
        <div className="mt-5 flex flex-col items-stretch gap-2 lg:flex-row lg:items-center">
          {PIPELINE.map((p, i) => (
            <div key={p.step} className="flex flex-col items-center gap-2 lg:flex-1 lg:flex-row">
              <div className="w-full rounded-2xl border border-cyan-500/30 bg-cyan-500/10 p-3 text-center">
                <p className="text-xs font-semibold uppercase tracking-wider text-cyan-700 dark:text-cyan-300">{i + 1}. {p.step}</p>
                <p className="mt-1 text-xs leading-5">{p.detail}</p>
              </div>
              {i < PIPELINE.length - 1 && (
                <>
                  <ArrowRight className="hidden h-4 w-4 shrink-0 text-gray-400 lg:block" />
                  <ArrowDown className="h-4 w-4 shrink-0 text-gray-400 lg:hidden" />
                </>
              )}
            </div>
          ))}
        </div>
        <div className="mt-4 flex flex-wrap gap-2">
          <span className="subtle-text text-xs font-semibold uppercase tracking-wider">Quality and operations:</span>
          {QUALITY.map((q) => (
            <span key={q} className="inline-flex items-center gap-1 rounded-full border border-gray-200 px-3 py-1 text-xs dark:border-white/10">
              <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" /> {q}
            </span>
          ))}
        </div>
      </div>

      <div className="surface-card p-6">
        <h3 className="section-title">All training runs</h3>
        <p className="subtle-text mt-1 text-sm">In the order they were trained. Recall and F1 are for the delayed class.</p>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[720px] text-left text-sm">
            <thead>
              <tr className="subtle-text border-b border-gray-200 text-xs uppercase tracking-wider dark:border-white/10">
                <th className="py-2 pr-3">Run</th>
                <th className="py-2 pr-3">Algorithm</th>
                <th className="py-2 pr-3">Features</th>
                <th className="py-2 pr-3">Settings</th>
                <th className="py-2 pr-3 text-right">Recall</th>
                <th className="py-2 pr-3 text-right">F1</th>
                <th className="py-2 pr-3 text-right">ROC-AUC</th>
                <th className="py-2 text-right">5-fold CV</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr
                  key={r.run}
                  className={`border-b border-gray-100 dark:border-white/5 ${r.selected ? 'bg-teal-500/10 font-semibold' : ''}`}
                >
                  <td className="py-2 pr-3">{r.run}</td>
                  <td className="py-2 pr-3">
                    {r.algorithm}
                    {r.selected && (
                      <span className="ml-2 rounded-full px-2 py-0.5 text-xs text-white" style={{ background: TEAL }}>
                        in use
                      </span>
                    )}
                  </td>
                  <td className="py-2 pr-3">
                    <span
                      className="rounded-full px-2 py-0.5 text-xs"
                      style={{ background: r.features === 'date' ? 'rgba(13,148,136,0.15)' : 'rgba(148,163,184,0.2)' }}
                    >
                      {r.features === 'date' ? 'with dates' : 'schedule'}
                    </span>
                  </td>
                  <td className="subtle-text py-2 pr-3">{r.settings}</td>
                  <td className="py-2 pr-3 text-right">{fmt(r.recall)}</td>
                  <td className="py-2 pr-3 text-right">{fmt(r.f1)}</td>
                  <td className="py-2 pr-3 text-right" style={{ color: r.selected ? TEAL : undefined }}>
                    {fmt(r.roc_auc)}
                  </td>
                  <td className="py-2 text-right">{r.cv_mean !== null ? `${fmt(r.cv_mean)} ± ${fmt(r.cv_std ?? 0)}` : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="subtle-text mt-3 text-xs">
          Source: MLflow tracking database, exported with <code>compare_models.py</code>.
        </p>
      </div>
    </div>
  )
}

