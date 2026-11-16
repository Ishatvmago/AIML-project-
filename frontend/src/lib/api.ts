// In development Vite proxies /api/* to FastAPI on port 8000 (see vite.config.ts).
// In production FastAPI serves this UI itself, so the API is on the same origin with no prefix.
export const API_BASE: string = import.meta.env.VITE_API_BASE ?? (import.meta.env.DEV ? '/api' : '')

async function handle<T>(response: Response): Promise<T> {
  if (response.ok) return response.json() as Promise<T>
  let message = `Request failed (${response.status})`
  try {
    const body = await response.json()
    if (Array.isArray(body.detail)) {
      // FastAPI 422: list of validation errors from the Pydantic data contract
      message = body.detail.map((d: { msg: string }) => d.msg).join('; ')
    } else if (body.detail) {
      message = String(body.detail)
    }
  } catch {
    // response had no JSON body
  }
  throw new Error(message)
}

export async function getJSON<T>(path: string): Promise<T> {
  return handle<T>(await fetch(`${API_BASE}${path}`))
}

export async function postJSON<T>(path: string, body: unknown): Promise<T> {
  return handle<T>(
    await fetch(`${API_BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  )
}

// ---------- shared types (match app.py) ----------

export interface FlightInput {
  flight_date: string // YYYY-MM-DD; the API derives month, weekday, day of month and holiday distance
  dep_hour: number
  distance: number
  airline: string
  origin: string
  dest: string
}

export interface Reason {
  feature: string
  label: string
  value: string | number
  impact: number
}

export interface PredictionResult {
  delay_probability: number
  prediction: 'Delayed' | 'On time'
  risk_level: 'Low' | 'Medium' | 'High'
  reasons: Reason[]
  tips: string[]
  delay_causes: {
    scope: 'route' | 'origin' | 'overall'
    label: string
    delayed_flights: number
    shares: Record<string, number>
  } | null
  date_info: {
    weekday: string
    month: string
    holiday: string | null
    days_to_holiday: number
  }
}

export interface AlternativeOption {
  kind: 'time' | 'airline' | 'date'
  label: string
  delay_probability: number
  risk_level: 'Low' | 'Medium' | 'High'
  reduction: number
  flight: FlightInput
}

export interface AlternativesResponse {
  current_probability: number
  options: AlternativeOption[]
}

export interface ExplainResult {
  explanation: string
  source: 'llm' | 'template'
}

export interface Airline {
  code: string
  name: string
  delay_rate: number
}

export interface OptionsResponse {
  airlines: Airline[]
  airports: string[]
  route_distance: Record<string, number>
}

export interface GroupStat {
  rate: number
  avg_delay: number
  flights: number
}

export interface StatsResponse {
  algorithm: string
  params: Record<string, string | number | null>
  metrics: { accuracy: number; precision: number; recall: number; f1: number; roc_auc: number }
  training_rows: number
  delay_rate: number
  avg_delay: number
  delay_buckets: Record<string, number>
  by_hour: Record<string, GroupStat>
  predictions_served: number
  llm: { available: boolean; model: string }
}

export interface FlightLeg {
  airline: string
  origin: string
  dest: string
  dep_hour: number
  distance: number
  flights_2024: number
}

export interface FlightLookupResponse {
  code: string
  airline: string
  airline_name: string
  legs: FlightLeg[]
  source: string
}

export interface ExperimentRun {
  run: number
  code: string
  algorithm: string
  features: 'base' | 'date'
  settings: string
  accuracy: number
  precision: number
  recall: number
  f1: number
  roc_auc: number
  cv_mean: number | null
  cv_std: number | null
  selected: boolean
}

export interface ExperimentsResponse {
  generated: string
  runs: ExperimentRun[]
  feature_importance: { feature: string; importance: number; std: number }[]
}

export const ALGORITHM_NAMES: Record<string, string> = {
  gb: 'Gradient Boosting',
  rf: 'Random Forest',
  logreg: 'Logistic Regression',
  dt: 'Decision Tree',
  et: 'Extra Trees',
  mlp: 'Neural Network (MLP)',
}

export const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
export const MONTHS = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December',
]

export const pct = (x: number, digits = 0) => `${(x * 100).toFixed(digits)}%`

/**
 * Links that open a public flight search for the same route and date in a new tab.
 * Only a link: nothing is fetched from these sites and no API or key is used.
 */
export function flightSearchLinks(f: { origin: string; dest: string; flight_date: string }, airlineName?: string) {
  const query = `${airlineName ? `${airlineName} ` : ''}flights from ${f.origin} to ${f.dest} on ${f.flight_date} one way`
  return {
    google: `https://www.google.com/travel/flights?q=${encodeURIComponent(query)}`,
    kayak: `https://www.kayak.com/flights/${f.origin}-${f.dest}/${f.flight_date}`,
  }
}
