import { CircularProgressbar, buildStyles } from 'react-circular-progressbar'
import 'react-circular-progressbar/dist/styles.css'
import { useEffect, useMemo, useState } from 'react'
import {
  AlertCircle,
  ArrowDownRight,
  ArrowUpRight,
  CalendarDays,
  Clock,
  Lightbulb,
  MapPin,
  PlaneTakeoff,
  Ruler,
  ShieldCheck,
  Sparkles,
} from 'lucide-react'
import PredictionMap from './PredictionMap'
import { AIRPORTS, AIRPORTS_BY_CODE, haversineMiles } from '../data/airports'
import {
  DAYS,
  MONTHS,
  getJSON,
  pct,
  postJSON,
  type ExplainResult,
  type FlightInput,
  type OptionsResponse,
  type PredictionResult,
  type AlternativeOption,
  type AlternativesResponse,
  type Reason,
} from '../lib/api'

type Theme = 'light' | 'dark'

interface PredictionFormProps {
  theme: Theme
}

const AIRLINE_LOGOS: Record<string, string> = {
  DL: '/airlines/delta.png',
  UA: '/airlines/united.png',
  AA: '/airlines/american.png',
  WN: '/airlines/southwest.png',
}

const RISK_COLORS: Record<string, string> = {
  Low: '#22c55e',
  Medium: '#f59e0b',
  High: '#f97316',
}

const selectClass = 'w-full bg-slate-900 text-white rounded-xl p-3 border border-white/10'

function formatHour(hour: number) {
  return `${String(hour).padStart(2, '0')}:00`
}

// YYYY-MM-DD in the user's local time zone (toISOString would shift to UTC)
function isoDate(d: Date) {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}

function parseDate(value: string) {
  const [y, m, d] = value.split('-').map(Number)
  return new Date(y, m - 1, d)
}

function formatDate(value: string) {
  return parseDate(value).toLocaleDateString('en-GB', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' })
}

const DEFAULT_DATE = isoDate(new Date(Date.now() + 7 * 24 * 3600 * 1000)) // one week from today

export default function PredictionForm({ theme }: PredictionFormProps) {
  const [options, setOptions] = useState<OptionsResponse | null>(null)
  const [form, setForm] = useState({
    airline: 'DL',
    origin: 'ATL',
    dest: 'LAX',
    flight_date: DEFAULT_DATE,
    dep_hour: 18,
  })
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [explanation, setExplanation] = useState<ExplainResult | null>(null)
  const [alternatives, setAlternatives] = useState<AlternativesResponse | null>(null)
  const [findingAlternatives, setFindingAlternatives] = useState(false)
  const [loading, setLoading] = useState(false)
  const [explaining, setExplaining] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    getJSON<OptionsResponse>('/options')
      .then(setOptions)
      .catch(() => setError('Could not reach the prediction API. Is the backend running?'))
  }, [])

  const airlineNames = useMemo(
    () => Object.fromEntries((options?.airlines ?? []).map((a) => [a.code, a.name])),
    [options],
  )
  const airports = useMemo(
    () => AIRPORTS.filter((a) => !options || options.airports.includes(a.code)),
    [options],
  )

  const departure = AIRPORTS_BY_CODE[form.origin]
  const arrival = AIRPORTS_BY_CODE[form.dest]

  // Real BTS route distance when the route exists in the data, otherwise great-circle distance.
  const distance = useMemo(() => {
    const known = options?.route_distance[`${form.origin}-${form.dest}`]
    return known ?? haversineMiles(departure, arrival)
  }, [options, form.origin, form.dest, departure, arrival])

  const payload: FlightInput = { ...form, distance }

  const update = (name: keyof typeof form, value: string) => {
    setResult(null)
    setExplanation(null)
    setAlternatives(null)
    setError('')
    setForm((prev) => {
      const next = { ...prev, [name]: name === 'dep_hour' ? Number(value) : value }
      // keep departure and arrival different
      if (next.origin === next.dest) {
        const other = AIRPORTS.find((a) => a.code !== next.origin)
        if (other) {
          if (name === 'origin') next.dest = other.code
          else next.origin = other.code
        }
      }
      return next
    })
  }

  const runPrediction = async (input: FlightInput) => {
    setLoading(true)
    setError('')
    setResult(null)
    setExplanation(null)
    setAlternatives(null)
    try {
      setResult(await postJSON<PredictionResult>('/predict', input))
    } catch (err) {
      setError(`Prediction failed: ${(err as Error).message}`)
      return
    } finally {
      setLoading(false)
    }
    // safer options for the same trip, scored by the same model (after the main result is shown)
    setFindingAlternatives(true)
    try {
      setAlternatives(await postJSON<AlternativesResponse>('/alternatives', input))
    } catch {
      setAlternatives(null)
    } finally {
      setFindingAlternatives(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    await runPrediction(payload)
  }

  // "Use this": put the safer option into the form and forecast it
  const applyOption = (option: AlternativeOption) => {
    const { airline, origin, dest, flight_date, dep_hour } = option.flight
    setForm({ airline, origin, dest, flight_date, dep_hour })
    void runPrediction(option.flight)
  }

  const handleExplain = async () => {
    setExplaining(true)
    try {
      setExplanation(await postJSON<ExplainResult>('/explain', payload))
    } catch (err) {
      setError(`Explanation failed: ${(err as Error).message}`)
    } finally {
      setExplaining(false)
    }
  }

  const formatReasonValue = (r: Reason) => {
    switch (r.feature) {
      case 'dep_hour':
        return formatHour(Number(r.value))
      case 'day_of_week':
        return DAYS[Number(r.value) - 1]
      case 'month':
        return MONTHS[Number(r.value) - 1]
      case 'day_of_month':
        return `day ${r.value}`
      case 'days_to_holiday':
        return Number(r.value) >= 14 ? 'no holiday nearby' : Number(r.value) === 0 ? 'on the holiday' : `${r.value} days away`
      case 'distance':
        return `${r.value} miles`
      case 'airline':
        return airlineNames[String(r.value)] ?? String(r.value)
      default:
        return String(r.value)
    }
  }

  const routeBrief = result
    ? `${airlineNames[form.airline] ?? form.airline} from ${departure.code} to ${arrival.code} has a ${pct(result.delay_probability)} chance of arriving 15+ minutes late.`
    : `${departure.city} to ${arrival.city} is the active route. Choose the airline, date and time, then generate a forecast.`

  const maxImpact = Math.max(0.01, ...(result?.reasons ?? []).map((r) => Math.abs(r.impact)))

  return (
    <div className="grid gap-6 xl:grid-cols-[0.92fr_1.08fr]">
      {/* ---------------- inputs ---------------- */}
      <div className="surface-card p-6 sm:p-8">
        <div className="mb-8 flex items-start justify-between gap-4">
          <div>
            <p className="eyebrow">Prediction Studio</p>
            <h2 className="mt-3 text-3xl font-semibold tracking-tight">Flight details</h2>
            <p className="subtle-text mt-3 max-w-xl text-sm leading-6">
              Enter the flight as it appears on the ticket. The model was trained on real US flights
              from every month of 2024 and estimates the chance of arriving 15 or more minutes late.
            </p>
          </div>
          <div className="icon-chip p-3">
            <PlaneTakeoff className="h-6 w-6" />
          </div>
        </div>

        <form onSubmit={handleSubmit} className="grid gap-5 lg:grid-cols-2">
          <div className="surface-panel p-4 lg:col-span-2">
            <label className="mb-3 block text-sm font-medium">Airline</label>
            <select name="airline" value={form.airline} onChange={(e) => update('airline', e.target.value)} className={selectClass}>
              {(options?.airlines ?? [{ code: form.airline, name: form.airline, delay_rate: 0 }]).map((a) => (
                <option key={a.code} value={a.code}>
                  {a.name} ({a.code})
                </option>
              ))}
            </select>
            <div className="mt-4 flex items-center gap-4">
              {AIRLINE_LOGOS[form.airline] ? (
                <img src={AIRLINE_LOGOS[form.airline]} alt={form.airline} className="h-12 rounded-xl bg-white p-2 shadow-lg" />
              ) : (
                <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-white text-sm font-bold text-slate-900 shadow-lg">
                  {form.airline}
                </div>
              )}
              <div>
                <p className="font-semibold">{airlineNames[form.airline] ?? form.airline}</p>
                <p className="subtle-text text-sm">
                  {options
                    ? `Historically delayed on ${pct(options.airlines.find((a) => a.code === form.airline)?.delay_rate ?? 0)} of flights`
                    : 'Loading airline history...'}
                </p>
              </div>
            </div>
          </div>

          <div className="surface-panel p-4">
            <label className="mb-3 block text-sm font-medium">Departure Airport</label>
            <select name="origin" value={form.origin} onChange={(e) => update('origin', e.target.value)} className={selectClass}>
              {airports.map((a) => (
                <option key={a.code} value={a.code}>
                  {a.code} - {a.city}
                </option>
              ))}
            </select>
            <p className="subtle-text mt-3 text-sm">{departure.name}</p>
          </div>

          <div className="surface-panel p-4">
            <label className="mb-3 block text-sm font-medium">Arrival Airport</label>
            <select name="dest" value={form.dest} onChange={(e) => update('dest', e.target.value)} className={selectClass}>
              {airports
                .filter((a) => a.code !== form.origin)
                .map((a) => (
                  <option key={a.code} value={a.code}>
                    {a.code} - {a.city}
                  </option>
                ))}
            </select>
            <p className="subtle-text mt-3 text-sm">{arrival.name}</p>
          </div>

          <div className="surface-panel p-4 lg:col-span-2">
            <label className="mb-3 flex items-center gap-2 text-sm font-medium">
              <CalendarDays className="accent-icon h-4 w-4" /> Flight Date
            </label>
            <input
              type="date"
              name="flight_date"
              required
              min="2024-01-01"
              max="2030-12-31"
              value={form.flight_date}
              onChange={(e) => e.target.value && update('flight_date', e.target.value)}
              className={selectClass}
              style={{ colorScheme: 'dark' }}
            />
            <p className="subtle-text mt-2 text-xs">
              {formatDate(form.flight_date)}. The model uses the month, weekday, day of the month and
              how close the date is to a US public holiday.
            </p>
          </div>

          <div className="surface-panel p-4 lg:col-span-2">
            <label className="mb-3 flex items-center gap-2 text-sm font-medium">
              <Clock className="accent-icon h-4 w-4" /> Scheduled Departure Time
            </label>
            <input
              type="range"
              name="dep_hour"
              min="0"
              max="23"
              value={form.dep_hour}
              onChange={(e) => update('dep_hour', e.target.value)}
              className="w-full"
            />
            <div className="mt-3 flex items-center justify-between text-sm">
              <span className="subtle-text">00:00</span>
              <span className="font-semibold">{formatHour(form.dep_hour)}</span>
              <span className="subtle-text">23:00</span>
            </div>
          </div>

          <div className="surface-panel flex items-center justify-between p-4 lg:col-span-2">
            <span className="flex items-center gap-2 text-sm font-medium">
              <Ruler className="accent-icon h-4 w-4" /> Route distance
            </span>
            <span className="font-semibold">{distance.toLocaleString()} miles</span>
          </div>

          <div className="lg:col-span-2">
            <button type="submit" disabled={loading || !options} className="primary-button w-full">
              {loading ? 'Running prediction...' : 'Generate delay forecast'}
            </button>
          </div>
        </form>
      </div>

      {/* ---------------- map + result ---------------- */}
      <div className="space-y-6">
        <div className="surface-card p-5 sm:p-6 lg:p-7">
          <div className="mb-6 flex items-start justify-between gap-4">
            <div>
              <p className="eyebrow">Live Route Map</p>
              <h2 className="mt-3 text-3xl font-semibold tracking-tight">US airport movement</h2>
              <p className="subtle-text mt-3 text-sm leading-6">
                The route colour changes with the predicted risk: green is low, amber is medium,
                orange is high.
              </p>
            </div>
            <div className="icon-chip p-3">
              <MapPin className="h-6 w-6" />
            </div>
          </div>

          <PredictionMap
            departureAirport={form.origin}
            arrivalAirport={form.dest}
            riskColor={result ? RISK_COLORS[result.risk_level] : null}
            airports={airports}
            theme={theme}
          />
        </div>

        <div className="surface-card p-6 sm:p-8">
          <div className="mb-8">
            <p className="eyebrow">Result</p>
            <h2 className="mt-3 text-3xl font-semibold tracking-tight">Prediction briefing</h2>
            <p className="subtle-text mt-3 text-sm leading-6">{routeBrief}</p>
          </div>

          {error && (
            <div className="error-panel mb-5 p-4">
              <div className="flex gap-3">
                <AlertCircle className="mt-0.5 h-5 w-5 shrink-0" />
                <p>{error}</p>
              </div>
            </div>
          )}

          {result && (
            <div className="space-y-5">
              <div className="rounded-[1.75rem] bg-[linear-gradient(135deg,rgba(8,47,73,0.98),rgba(14,116,144,0.95))] p-7 text-white shadow-2xl shadow-cyan-950/25">
                <div className="grid items-center gap-6 sm:grid-cols-[1fr_auto]">
                  <div>
                    <p className="text-sm uppercase tracking-[0.24em] text-cyan-100/80">Delay probability</p>
                    <p className="mt-4 text-6xl font-semibold tracking-tight">{pct(result.delay_probability)}</p>
                    <p className="mt-2 text-cyan-100/80">
                      chance of arriving 15+ minutes late on {departure.code} to {arrival.code}
                    </p>
                    {result.date_info.holiday && (
                      <p className="mt-3 inline-flex items-center gap-2 rounded-full bg-white/10 px-3 py-1 text-sm">
                        <CalendarDays className="h-4 w-4" />
                        {result.date_info.days_to_holiday === 0
                          ? `On ${result.date_info.holiday}`
                          : `${result.date_info.days_to_holiday} day${result.date_info.days_to_holiday > 1 ? 's' : ''} from ${result.date_info.holiday}`}
                      </p>
                    )}
                    <span
                      className="mt-4 inline-block rounded-full px-3 py-1 text-sm font-semibold text-slate-950"
                      style={{ background: RISK_COLORS[result.risk_level] }}
                    >
                      {result.risk_level} risk · model says “{result.prediction}”
                    </span>
                  </div>
                  <div className="mx-auto h-32 w-32">
                    <CircularProgressbar
                      value={result.delay_probability * 100}
                      text={pct(result.delay_probability)}
                      styles={buildStyles({
                        textColor: '#ffffff',
                        pathColor: RISK_COLORS[result.risk_level],
                        trailColor: 'rgba(255,255,255,0.15)',
                      })}
                    />
                  </div>
                </div>
              </div>

              <div className="surface-panel p-5">
                <h3 className="text-base font-semibold">Why the model says this</h3>
                <p className="subtle-text mt-1 text-sm">
                  How much each input moves the probability compared with a typical flight.
                </p>
                <ul className="mt-4 space-y-3">
                  {result.reasons.map((r) => (
                    <li key={r.feature}>
                      <div className="flex items-center justify-between text-sm">
                        <span className="flex items-center gap-2">
                          {r.impact > 0 ? (
                            <ArrowUpRight className="h-4 w-4 text-orange-500" />
                          ) : (
                            <ArrowDownRight className="h-4 w-4 text-emerald-500" />
                          )}
                          <span className="capitalize">{r.label}</span>
                          <span className="subtle-text">({formatReasonValue(r)})</span>
                        </span>
                        <span className={`font-semibold ${r.impact > 0 ? 'text-orange-500' : 'text-emerald-500'}`}>
                          {r.impact > 0 ? '+' : '−'}
                          {pct(Math.abs(r.impact), 1)}
                        </span>
                      </div>
                      <div className="mt-1.5 h-2 rounded-full bg-gray-200 dark:bg-white/10">
                        <div
                          className={`h-2 rounded-full ${r.impact > 0 ? 'bg-orange-500' : 'bg-emerald-500'}`}
                          style={{ width: `${(Math.abs(r.impact) / maxImpact) * 100}%` }}
                        />
                      </div>
                    </li>
                  ))}
                  {result.reasons.length === 0 && (
                    <li className="subtle-text text-sm">This flight looks like a typical flight; no single factor stands out.</li>
                  )}
                </ul>
              </div>

              <div className="surface-panel p-5">
                <h3 className="flex items-center gap-2 text-base font-semibold">
                  <Lightbulb className="h-4 w-4 text-amber-500" /> Travel tips
                </h3>
                <ul className="mt-3 space-y-2 text-sm leading-6">
                  {result.tips.map((tip) => (
                    <li key={tip} className="subtle-text">
                      • {tip}
                    </li>
                  ))}
                </ul>
              </div>

              <div className="surface-panel p-5">
                <h3 className="flex items-center gap-2 text-base font-semibold">
                  <ShieldCheck className="h-4 w-4 text-emerald-500" /> Safer options for this trip
                </h3>
                {findingAlternatives && <p className="subtle-text mt-3 text-sm">Checking other times, airlines and dates...</p>}
                {!findingAlternatives && alternatives && alternatives.options.length === 0 && (
                  <p className="subtle-text mt-3 text-sm leading-6">
                    No clearly safer time, airline or nearby date for this route: this flight is already one of
                    the better choices.
                  </p>
                )}
                {!findingAlternatives && alternatives && alternatives.options.length > 0 && (
                  <ul className="mt-3 space-y-2">
                    {alternatives.options.map((o) => (
                      <li key={o.label} className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-gray-200 p-3 dark:border-white/10">
                        <div className="flex items-start gap-3">
                          <span className="mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: RISK_COLORS[o.risk_level] }} />
                          <div>
                            <p className="text-sm font-medium">{o.label}</p>
                            <p className="subtle-text text-xs">
                              {pct(o.delay_probability)} delay risk, {Math.round(o.reduction * 100)} points lower than your flight
                            </p>
                          </div>
                        </div>
                        <button type="button" onClick={() => applyOption(o)} className="secondary-button text-sm">
                          Use this
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
                <p className="subtle-text mt-3 text-xs leading-5">
                  Scored by the same trained model. Airlines shown flew this route in 2024; check that the
                  flight exists when you book.
                </p>
              </div>

              <div className="surface-panel p-5">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <h3 className="flex items-center gap-2 text-base font-semibold">
                    <Sparkles className="accent-icon h-4 w-4" /> AI explanation
                  </h3>
                  {!explanation && (
                    <button type="button" onClick={handleExplain} disabled={explaining} className="secondary-button text-sm">
                      {explaining ? 'Llama is writing (runs on this computer)...' : 'Explain in plain English'}
                    </button>
                  )}
                </div>
                {explanation ? (
                  <>
                    <p className="mt-3 whitespace-pre-line text-sm leading-7">{explanation.explanation}</p>
                    <p className="subtle-text mt-3 text-xs">
                      {explanation.source === 'llm'
                        ? 'Written by Llama 3.2 running locally through Ollama (no API key, no internet). The numbers come from our trained model.'
                        : 'Local LLM not available, so this summary was generated from a template using the model’s reasons.'}
                    </p>
                  </>
                ) : (
                  <p className="subtle-text mt-3 text-sm">
                    A local generative AI model turns the prediction and reasons above into a short
                    explanation for the passenger.
                  </p>
                )}
              </div>

              <div className="surface-panel p-5">
                <h3 className="text-base font-semibold">Briefing summary</h3>
                <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                  <span className="subtle-text">Airline</span>
                  <span className="font-medium">{airlineNames[form.airline] ?? form.airline}</span>
                  <span className="subtle-text">Route</span>
                  <span className="font-medium">
                    {departure.code} to {arrival.code}
                  </span>
                  <span className="subtle-text">Date</span>
                  <span className="font-medium">{formatDate(form.flight_date)}</span>
                  <span className="subtle-text">Nearest holiday</span>
                  <span className="font-medium">
                    {result.date_info.holiday
                      ? `${result.date_info.holiday} (${result.date_info.days_to_holiday} days)`
                      : 'None within two weeks'}
                  </span>
                  <span className="subtle-text">Departure</span>
                  <span className="font-medium">{formatHour(form.dep_hour)}</span>
                  <span className="subtle-text">Distance</span>
                  <span className="font-medium">{distance.toLocaleString()} miles</span>
                </div>
              </div>
            </div>
          )}

          {!result && !error && (
            <div className="surface-panel flex min-h-[240px] items-center justify-center p-6 text-center">
              <div>
                <p className="text-lg font-semibold">Preview the active route</p>
                <p className="subtle-text mt-3 max-w-md text-sm leading-6">
                  Airport selections already update the map above. Generate a forecast to see the
                  delay probability, the reasons behind it and travel tips.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
