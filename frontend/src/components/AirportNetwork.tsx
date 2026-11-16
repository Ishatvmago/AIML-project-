import { useEffect, useState } from 'react'
import { AlertCircle } from 'lucide-react'
import { AIRPORTS_BY_CODE } from '../data/airports'
import { getJSON, type GroupStat } from '../lib/api'

interface Airport {
  lat: number
  lon: number
  name: string
  rate: number
  avg_delay: number
  flights: number
}

interface Connection {
  source: string
  target: string
  volume: number
  rate: number
  distance: number
}

interface NetworkResponse {
  airports: Record<string, GroupStat>
  connections: Connection[]
}

interface NetworkData {
  airports: { [key: string]: Airport }
  connections: Connection[]
}

export default function AirportNetwork() {
  const [data, setData] = useState<NetworkData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    const fetchData = async () => {
      try {
        const result = await getJSON<NetworkResponse>('/airport-network')
        // delay statistics come from the API; names and coordinates from the reference list
        const airports = Object.fromEntries(
          Object.entries(result.airports)
            .filter(([code]) => AIRPORTS_BY_CODE[code])
            .map(([code, s]) => {
              const ref = AIRPORTS_BY_CODE[code]
              return [code, { lat: ref.lat, lon: ref.lon, name: ref.name, ...s }]
            }),
        )
        setData({ airports, connections: result.connections })
      } catch (err) {
        setError('Failed to load airport network')
        console.error(err)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  if (loading) {
    return (
      <div className="surface-card p-8 text-center">
        <p className="subtle-text">Loading network data...</p>
      </div>
    )
  }

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

  const width = 800
  const height = 500
  const padding = 40
  const airports = data?.airports || {}
  const connections = data?.connections || []

  const lats = Object.values(airports).map((airport) => airport.lat)
  const lons = Object.values(airports).map((airport) => airport.lon)
  const minLat = Math.min(...lats)
  const maxLat = Math.max(...lats)
  const minLon = Math.min(...lons)
  const maxLon = Math.max(...lons)

  const getX = (lon: number) =>
    padding + ((lon - minLon) / (maxLon - minLon)) * (width - 2 * padding)
  const getY = (lat: number) =>
    padding + ((maxLat - lat) / (maxLat - minLat)) * (height - 2 * padding)

  return (
    <div className="space-y-6">
      <div className="surface-card p-6">
        <p className="eyebrow">Route Map</p>
        <h2 className="mt-3 text-3xl font-semibold tracking-tight">Airport network view</h2>
        <p className="subtle-text mt-4 max-w-2xl text-sm leading-7">
          The 20 busiest US airports in the dataset and the 40 busiest routes between them. Every
          number is calculated from real 2024 flights.
        </p>
      </div>

      <div className="surface-card p-6">
        <div className="mb-5">
          <h3 className="section-title">Airport network map</h3>
          <p className="subtle-text mt-2 text-sm">
            Circle size represents the airport's delay rate. Route thickness represents the number of flights.
          </p>
        </div>
        <div className="overflow-x-auto">
          <svg width={width} height={height} className="min-w-[760px] rounded-[1.5rem]">
            <defs>
              <linearGradient id="networkBg" x1="0%" x2="100%" y1="0%" y2="100%">
                <stop offset="0%" stopColor="#082f49" />
                <stop offset="100%" stopColor="#111827" />
              </linearGradient>
            </defs>

            <rect width={width} height={height} rx="24" fill="url(#networkBg)" />

            {connections.map((conn) => {
              const source = airports[conn.source]
              const target = airports[conn.target]
              if (!source || !target) return null

              return (
                <line
                  key={`${conn.source}-${conn.target}`}
                  x1={getX(source.lon)}
                  y1={getY(source.lat)}
                  x2={getX(target.lon)}
                  y2={getY(target.lat)}
                  stroke="#67e8f9"
                  strokeWidth={Math.max(1.25, conn.volume / 70)}
                  opacity="0.45"
                />
              )
            })}

            {Object.entries(airports).map(([code, airport]) => (
              <g key={code}>
                <circle
                  cx={getX(airport.lon)}
                  cy={getY(airport.lat)}
                  r={Math.max(6, airport.rate * 40)}
                  fill="#f59e0b"
                  opacity="0.95"
                />
                <circle
                  cx={getX(airport.lon)}
                  cy={getY(airport.lat)}
                  r={Math.max(12, airport.rate * 60)}
                  fill="#f59e0b"
                  opacity="0.12"
                />
                <text
                  x={getX(airport.lon)}
                  y={getY(airport.lat) - 18}
                  textAnchor="middle"
                  style={{ fill: '#f8fafc', fontSize: '12px', fontWeight: 700 }}
                >
                  {code}
                </text>
              </g>
            ))}
          </svg>
        </div>
      </div>

      <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
        <div className="surface-card p-6">
          <h3 className="section-title">Airport details</h3>
          <div className="mt-5 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="ui-border border-b text-left">
                  <th className="px-4 py-3 font-medium">Code</th>
                  <th className="px-4 py-3 font-medium">Airport</th>
                  <th className="px-4 py-3 text-right font-medium">Delayed</th>
                  <th className="px-4 py-3 text-right font-medium">Avg late</th>
                  <th className="px-4 py-3 text-right font-medium">Flights</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(airports)
                  .sort((a, b) => b[1].rate - a[1].rate)
                  .map(([code, airport]) => (
                    <tr
                      key={code}
                      className="ui-row ui-row-hover border-b transition-colors"
                    >
                      <td className="accent-icon px-4 py-4 font-semibold">
                        {code}
                      </td>
                      <td className="px-4 py-4">{airport.name}</td>
                      <td className="px-4 py-4 text-right font-medium">{(airport.rate * 100).toFixed(1)}%</td>
                      <td className="px-4 py-4 text-right">{airport.avg_delay.toFixed(1)} min</td>
                      <td className="px-4 py-4 text-right subtle-text">{airport.flights.toLocaleString()}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="space-y-6">
          <div className="surface-card p-6">
            <h3 className="section-title">Top routes</h3>
            <ul className="mt-5 space-y-3">
              {[...connections]
                .sort((a, b) => b.volume - a.volume)
                .slice(0, 5)
                .map((conn) => (
                  <li key={`${conn.source}-${conn.target}`} className="surface-panel flex items-center justify-between px-4 py-3 text-sm">
                    <span>
                      {conn.source} to {conn.target}
                    </span>
                    <span className="accent-icon font-semibold">
                      {conn.volume} flights · {(conn.rate * 100).toFixed(0)}% delayed
                    </span>
                  </li>
                ))}
            </ul>
          </div>

          <div className="surface-card p-6">
            <h3 className="section-title">Network summary</h3>
            <div className="mt-5 space-y-3">
              <div className="surface-panel flex items-center justify-between px-4 py-3 text-sm">
                <span className="subtle-text">Total airports</span>
                <span className="font-semibold">{Object.keys(airports).length}</span>
              </div>
              <div className="surface-panel flex items-center justify-between px-4 py-3 text-sm">
                <span className="subtle-text">Busiest routes shown</span>
                <span className="font-semibold">{connections.length}</span>
              </div>
              <div className="surface-panel flex items-center justify-between px-4 py-3 text-sm">
                <span className="subtle-text">Departures in dataset</span>
                <span className="font-semibold">
                  {Object.values(airports).reduce((sum, airport) => sum + airport.flights, 0).toLocaleString()}
                </span>
              </div>
              <div className="surface-panel flex items-center justify-between px-4 py-3 text-sm">
                <span className="subtle-text">Average airport delay rate</span>
                <span className="font-semibold text-amber-500">
                  {(
                    (Object.values(airports).reduce((sum, airport) => sum + airport.rate, 0) /
                      Object.keys(airports).length) *
                    100
                  ).toFixed(1)}
                  %
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
