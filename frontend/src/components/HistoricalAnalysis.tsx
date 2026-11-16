import { useEffect, useState } from 'react'
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import { AlertCircle } from 'lucide-react'
import { DAYS, MONTHS, getJSON, pct, type GroupStat } from '../lib/api'

interface HistoricalData {
  by_hour: Record<string, GroupStat>
  by_day: Record<string, GroupStat>
  by_month: Record<string, GroupStat>
  rows: number
  by_days_to_holiday: Record<string, GroupStat>
  by_airline: Record<string, GroupStat>
  by_origin: Record<string, GroupStat>
  airline_names: Record<string, string>
  delay_rate: number
}

const toPct = (x: number) => Number((x * 100).toFixed(1))
const tooltipPct = (value: unknown) => [`${value}%`, 'Delayed 15+ min']

export default function HistoricalAnalysis() {
  const [data, setData] = useState<HistoricalData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    getJSON<HistoricalData>('/historical-data')
      .then(setData)
      .catch(() => setError('Failed to load historical data'))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="surface-card p-8 text-center">
        <p className="subtle-text">Loading data...</p>
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

  const overall = toPct(data.delay_rate)

  const hourlyData = Object.entries(data.by_hour)
    .filter(([, s]) => s.flights >= 100) // night hours have too few flights to be reliable
    .map(([hour, s]) => ({ hour: `${hour.padStart(2, '0')}:00`, rate: toPct(s.rate), avg: s.avg_delay }))

  const airportData = Object.entries(data.by_origin)
    .map(([airport, s]) => ({ airport, rate: toPct(s.rate) }))
    .sort((a, b) => b.rate - a.rate)

  const airlineData = Object.entries(data.by_airline)
    .map(([code, s]) => ({ airline: data.airline_names[code] ?? code, rate: toPct(s.rate) }))
    .sort((a, b) => a.rate - b.rate)

  const dayData = Object.entries(data.by_day).map(([d, s]) => ({ day: DAYS[Number(d) - 1].slice(0, 3), rate: toPct(s.rate) }))
  const monthData = Object.entries(data.by_month).map(([m, s]) => ({ month: MONTHS[Number(m) - 1].slice(0, 3), rate: toPct(s.rate), avg: s.avg_delay }))

  const worstHour = hourlyData.reduce((a, b) => (b.rate > a.rate ? b : a))
  const bestHour = hourlyData.reduce((a, b) => (b.rate < a.rate ? b : a))
  // group "days to the nearest US public holiday" into readable buckets (flight-weighted)
  const holidayBuckets: [string, number, number][] = [
    ['0-1 days', 0, 1],
    ['2-3 days', 2, 3],
    ['4-7 days', 4, 7],
    ['8-13 days', 8, 13],
    ['No holiday nearby', 14, 99],
  ]
  const holidayData = holidayBuckets
    .map(([label, lo, hi]) => {
      const groups = Object.entries(data.by_days_to_holiday ?? {})
        .filter(([d]) => Number(d) >= lo && Number(d) <= hi)
        .map(([, g]) => g)
      const flights = groups.reduce((n, g) => n + g.flights, 0)
      const delayed = groups.reduce((n, g) => n + g.rate * g.flights, 0)
      return { bucket: label, rate: flights ? toPct(delayed / flights) : 0, flights }
    })
    .filter((b) => b.flights > 0)

  const worstMonth = monthData.reduce((a, b) => (b.rate > a.rate ? b : a))
  const bestMonth = monthData.reduce((a, b) => (b.rate < a.rate ? b : a))

  return (
    <div className="space-y-6">
      <div className="surface-card p-6">
        <p className="eyebrow">Historical Patterns</p>
        <h2 className="mt-3 text-3xl font-semibold tracking-tight">Delay performance trends</h2>
        <p className="subtle-text mt-4 max-w-3xl text-sm leading-7">
          Calculated from {data.rows.toLocaleString()} real US flights (Bureau of Transportation
          Statistics, every month of 2024). A flight counts as delayed when it arrives 15 or more minutes late, the
          official US DOT definition. The dashed line is the overall average ({overall}%).
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <div className="metric-card text-center">
          <p className="subtle-text text-sm">Worst departure hour</p>
          <p className="mt-3 text-3xl font-semibold tracking-tight">{worstHour.hour}</p>
          <p className="subtle-text mt-1 text-sm">{worstHour.rate}% delayed</p>
        </div>
        <div className="metric-card text-center">
          <p className="subtle-text text-sm">All flights</p>
          <p className="mt-3 text-3xl font-semibold tracking-tight">{pct(data.delay_rate, 1)}</p>
          <p className="subtle-text mt-1 text-sm">arrived 15+ minutes late</p>
        </div>
        <div className="metric-card text-center">
          <p className="subtle-text text-sm">Best departure hour</p>
          <p className="mt-3 text-3xl font-semibold tracking-tight">{bestHour.hour}</p>
          <p className="subtle-text mt-1 text-sm">{bestHour.rate}% delayed</p>
        </div>
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <div className="surface-card p-6 xl:col-span-2">
          <h3 className="section-title">Delay rate by departure hour</h3>
          <p className="subtle-text mt-1 text-sm">Delays build up during the day as late aircraft knock on to later flights.</p>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={hourlyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.22)" />
              <XAxis dataKey="hour" angle={-45} textAnchor="end" height={60} stroke="#94a3b8" tickLine={false} axisLine={false} />
              <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} unit="%" />
              <Tooltip formatter={tooltipPct} />
              <ReferenceLine y={overall} stroke="#94a3b8" strokeDasharray="6 6" />
              <Line type="monotone" dataKey="rate" stroke="#0891b2" strokeWidth={3} dot={{ r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="surface-card p-6">
          <h3 className="section-title">Delay rate by departure airport</h3>
          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={airportData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.22)" />
              <XAxis dataKey="airport" stroke="#94a3b8" tickLine={false} axisLine={false} interval={0} angle={-45} textAnchor="end" height={50} />
              <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} unit="%" />
              <Tooltip formatter={tooltipPct} />
              <ReferenceLine y={overall} stroke="#94a3b8" strokeDasharray="6 6" />
              <Bar dataKey="rate" fill="#f59e0b" radius={[10, 10, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="surface-card p-6">
          <h3 className="section-title">Airline delay rate</h3>
          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={airlineData} layout="vertical" margin={{ left: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.22)" />
              <XAxis type="number" stroke="#94a3b8" tickLine={false} axisLine={false} unit="%" />
              <YAxis dataKey="airline" type="category" stroke="#94a3b8" tickLine={false} axisLine={false} width={110} fontSize={12} />
              <Tooltip formatter={tooltipPct} />
              <ReferenceLine x={overall} stroke="#94a3b8" strokeDasharray="6 6" />
              <Bar dataKey="rate" fill="#0f766e" radius={[0, 10, 10, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="surface-card p-6">
          <h3 className="section-title">Delay rate by day of week</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={dayData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.22)" />
              <XAxis dataKey="day" stroke="#94a3b8" tickLine={false} axisLine={false} />
              <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} unit="%" />
              <Tooltip formatter={tooltipPct} />
              <Bar dataKey="rate" fill="#38bdf8" radius={[10, 10, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="surface-card p-6">
          <h3 className="section-title">Delay rate by month (2024)</h3>
          <p className="subtle-text mt-1 text-sm">
            Worst: {worstMonth.month} ({worstMonth.rate}%) · Best: {bestMonth.month} ({bestMonth.rate}%)
          </p>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={monthData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.22)" />
              <XAxis dataKey="month" stroke="#94a3b8" tickLine={false} axisLine={false} fontSize={12} />
              <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} unit="%" />
              <Tooltip formatter={tooltipPct} />
              <ReferenceLine y={overall} stroke="#94a3b8" strokeDasharray="6 6" />
              <Bar dataKey="rate" fill="#6366f1" radius={[10, 10, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="surface-card p-6">
          <h3 className="section-title">Holiday effect</h3>
          <p className="subtle-text mt-1 text-sm">
            Delay rate by how many days the flight is from the nearest US public holiday
            (Thanksgiving, Christmas, July 4 and others).
          </p>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={holidayData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.22)" />
              <XAxis dataKey="bucket" stroke="#94a3b8" tickLine={false} axisLine={false} fontSize={12} />
              <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} unit="%" />
              <Tooltip formatter={tooltipPct} />
              <ReferenceLine y={overall} stroke="#94a3b8" strokeDasharray="6 6" />
              <Bar dataKey="rate" fill="#ec4899" radius={[10, 10, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}
