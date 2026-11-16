import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import L, { type LatLngBoundsExpression, type LatLngTuple } from 'leaflet'
import { CircleMarker, GeoJSON, MapContainer, Marker, Pane, Polyline, TileLayer, Tooltip, useMap } from 'react-leaflet'
import type { GeoJsonObject } from 'geojson'
import { Maximize2, Minus, Plus } from 'lucide-react'
import { AIRPORTS_BY_CODE, type AirportRecord } from '../data/airports'
// US state outlines bundled with the app (public US Census-based GeoJSON). The "Offline" style uses
// them so the map works with no tile server and no internet; it is also the automatic fallback.
import usStates from '../data/us-states.json'

/**
 * Map background styles. The online ones load free public map tiles (no API key needed).
 * Only the map picture comes from these services; every prediction comes from our own model.
 */
type MapStyle = 'dark' | 'streets' | 'satellite' | 'offline'

const ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services'
const ONLINE_STYLES: Record<Exclude<MapStyle, 'offline'>, { base: string; labels?: string; attribution: string }> = {
  dark: {
    base: `${ESRI}/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}`,
    labels: `${ESRI}/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}`,
    attribution: 'Tiles &copy; Esri, HERE, Garmin, &copy; OpenStreetMap contributors',
  },
  streets: {
    base: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  },
  satellite: {
    base: `${ESRI}/World_Imagery/MapServer/tile/{z}/{y}/{x}`,
    labels: `${ESRI}/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}`,
    attribution: 'Tiles &copy; Esri, Maxar, Earthstar Geographics',
  },
}
const STYLE_LABELS: Record<MapStyle, string> = { dark: 'Dark', streets: 'Streets', satellite: 'Satellite', offline: 'Offline' }
const STYLE_KEY = 'flight-map-style'

function initialStyle(): MapStyle {
  try {
    const saved = localStorage.getItem(STYLE_KEY) as MapStyle | null
    if (saved && saved in STYLE_LABELS) return saved
  } catch {
    // storage blocked (private window): fall through to the default
  }
  return typeof navigator !== 'undefined' && navigator.onLine === false ? 'offline' : 'dark'
}

/** Online tiles; if they cannot be loaded (no internet, service down) we fall back to the offline map. */
function OnlineTiles({ style, onFail }: { style: Exclude<MapStyle, 'offline'>; onFail: () => void }) {
  const loaded = useRef(0)
  const failed = useRef(0)
  const config = ONLINE_STYLES[style]

  useEffect(() => {
    loaded.current = 0
    failed.current = 0
    // nothing loaded after 8 seconds also counts as a failure
    const timer = window.setTimeout(() => {
      if (loaded.current === 0) onFail()
    }, 8000)
    return () => window.clearTimeout(timer)
  }, [style, onFail])

  const handlers = {
    tileload: () => {
      loaded.current += 1
    },
    tileerror: () => {
      failed.current += 1
      if (failed.current >= 4 && loaded.current === 0) onFail()
    },
  }

  return (
    <>
      <TileLayer key={style} url={config.base} attribution={config.attribution} maxZoom={16} eventHandlers={handlers} />
      {config.labels && <TileLayer key={`${style}-labels`} url={config.labels} maxZoom={16} />}
    </>
  )
}

type Theme = 'light' | 'dark'

interface PredictionMapProps {
  departureAirport: string
  arrivalAirport: string
  theme: Theme
  airports: AirportRecord[]
  riskColor?: string | null
}

const USA_BOUNDS: LatLngBoundsExpression = [
  [24.396308, -125.0],
  [49.384358, -66.93457],
]

const FLIGHT_SECONDS = 5 // one animated flight from departure to arrival
const DEPARTURE_COLOR = '#22c55e'
const ARRIVAL_COLOR = '#f59e0b'

// Plane silhouette pointing up (north); rotated to the direction of travel while it flies.
const PLANE_PATH =
  'M21 16v-2l-8-5V3.5c0-.83-.67-1.5-1.5-1.5S10 2.67 10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z'

const prefersReducedMotion = () =>
  typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

/**
 * Curved flight path between two airports, like the arcs on flight-tracking sites:
 * a quadratic Bezier curve whose middle is pushed sideways (towards the north).
 */
function arcPoints(a: LatLngTuple, b: LatLngTuple, steps = 90): LatLngTuple[] {
  const [lat1, lon1] = a
  const [lat2, lon2] = b
  const dLat = lat2 - lat1
  const dLon = lon2 - lon1
  const length = Math.hypot(dLat, dLon) || 1
  let nLat = -dLon / length
  let nLon = dLat / length
  if (nLat < 0) {
    nLat = -nLat
    nLon = -nLon
  }
  const bend = length * 0.2
  const cLat = (lat1 + lat2) / 2 + nLat * bend
  const cLon = (lon1 + lon2) / 2 + nLon * bend
  return Array.from({ length: steps + 1 }, (_, i) => {
    const t = i / steps
    const u = 1 - t
    return [u * u * lat1 + 2 * u * t * cLat + t * t * lat2, u * u * lon1 + 2 * u * t * cLon + t * t * lon2]
  })
}

function airportIcon(code: string, color: string, label: string) {
  return L.divIcon({
    className: '',
    iconSize: [0, 0],
    html: `
      <div class="airport-pulse" style="--pulse:${color}">
        <span class="airport-pulse__ring"></span>
        <span class="airport-pulse__ring airport-pulse__ring--late"></span>
        <span class="airport-pulse__dot"></span>
        <span class="airport-pulse__label"><b>${code}</b> ${label}</span>
      </div>`,
  })
}

function FitMapToSelection({ points }: { points: LatLngTuple[] | null }) {
  const map = useMap()
  const firstFit = useRef(true)

  const fit = useCallback(
    (animate: boolean) => {
      map.invalidateSize()
      const size = map.getSize()
      if (size.x < 50 || size.y < 50) return // not laid out yet; the ResizeObserver below retries
      // extra room on the right for the airport name labels and at the top for the route card
      const options = { paddingTopLeft: L.point(40, 90), paddingBottomRight: L.point(Math.min(150, size.x / 4), 60) }
      if (!points) map.fitBounds(USA_BOUNDS, { padding: [20, 20] })
      else if (animate) map.flyToBounds(L.latLngBounds(points), { ...options, duration: 1.1 })
      else map.fitBounds(L.latLngBounds(points), options)
    },
    [map, points],
  )

  // instant fit the first time, a smooth flight to the new route after that
  useEffect(() => {
    fit(!firstFit.current)
    firstFit.current = false
  }, [fit])

  // re-fit when the map box changes size (page layout finishing, window resize, phone rotation)
  useEffect(() => {
    const observer = new ResizeObserver(() => fit(false))
    observer.observe(map.getContainer())
    return () => observer.disconnect()
  }, [map, fit])

  // After an animated flyToBounds, Leaflet can leave HTML markers at a position computed before
  // the final frame. Re-place every marker when a move or zoom has finished.
  useEffect(() => {
    const refresh = () =>
      map.eachLayer((layer) => {
        if (layer instanceof L.Marker) layer.setLatLng(layer.getLatLng())
      })
    map.on('moveend zoomend', refresh)
    return () => {
      map.off('moveend zoomend', refresh)
    }
  }, [map])

  return null
}

/** Plane that flies along the arc, turning to follow the curve, and leaves a solid trail behind it. */
function FlyingPlane({ points, color }: { points: LatLngTuple[]; color: string }) {
  const map = useMap()

  useEffect(() => {
    const icon = L.divIcon({
      className: '',
      iconSize: [34, 34],
      iconAnchor: [17, 17],
      html: `<svg class="flying-plane" width="34" height="34" viewBox="0 0 24 24" style="--plane:${color}">
               <path d="${PLANE_PATH}" />
             </svg>`,
    })
    const plane = L.marker(points[0], { icon, interactive: false, keyboard: false }).addTo(map)
    const trail = L.polyline([points[0]], { pane: 'routes', color, weight: 4, opacity: 0.95, interactive: false }).addTo(map)

    const place = (progress: number) => {
      const exact = progress * (points.length - 1)
      const i = Math.min(Math.floor(exact), points.length - 2)
      const f = exact - i
      const [aLat, aLon] = points[i]
      const [bLat, bLon] = points[i + 1]
      const here: LatLngTuple = [aLat + (bLat - aLat) * f, aLon + (bLon - aLon) * f]
      plane.setLatLng(here)
      trail.setLatLngs([...points.slice(0, i + 1), here])
      // heading in screen space, so the nose follows the curve on the map
      const p1 = map.latLngToLayerPoint(points[i])
      const p2 = map.latLngToLayerPoint(points[i + 1])
      const angle = (Math.atan2(p2.y - p1.y, p2.x - p1.x) * 180) / Math.PI + 90
      const svg = plane.getElement()?.querySelector('svg')
      if (svg) svg.style.transform = `rotate(${angle}deg)`
    }

    if (prefersReducedMotion()) {
      place(0.5)
      return () => {
        plane.remove()
        trail.remove()
      }
    }

    let frame = 0
    let start: number | null = null
    const tick = (now: number) => {
      start ??= now
      const cycle = ((now - start) / 1000) % (FLIGHT_SECONDS + 0.8) // short pause at the arrival airport
      const t = Math.min(cycle / FLIGHT_SECONDS, 1)
      const eased = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2 // take off slowly, cruise, land slowly
      place(eased)
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(frame)
      plane.remove()
      trail.remove()
    }
  }, [map, points, color])

  return null
}

/** Zoom in / zoom out / fit the route back in view. Lives inside the map so it can use the map instance. */
function ZoomControls({ points }: { points: LatLngTuple[] | null }) {
  const map = useMap()
  const box = useRef<HTMLDivElement>(null)

  // clicks on the buttons must not also drag or zoom the map underneath
  useEffect(() => {
    if (box.current) {
      L.DomEvent.disableClickPropagation(box.current)
      L.DomEvent.disableScrollPropagation(box.current)
    }
  }, [])

  const fitRoute = () => {
    const size = map.getSize()
    if (points)
      map.flyToBounds(L.latLngBounds(points), {
        paddingTopLeft: L.point(40, 90),
        paddingBottomRight: L.point(Math.min(150, size.x / 4), 60),
        duration: 0.8,
      })
    else map.flyToBounds(USA_BOUNDS, { padding: [20, 20], duration: 0.8 })
  }

  const button =
    'flex h-9 w-9 items-center justify-center text-slate-200 transition hover:bg-white/10 hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-400'

  return (
    <div
      ref={box}
      className="absolute bottom-12 right-4 z-[500] flex flex-col overflow-hidden rounded-xl border border-white/10 bg-slate-950/80 shadow-xl backdrop-blur"
    >
      <button type="button" className={button} onClick={() => map.zoomIn()} aria-label="Zoom in" title="Zoom in">
        <Plus className="h-4 w-4" />
      </button>
      <button type="button" className={`${button} border-y border-white/10`} onClick={() => map.zoomOut()} aria-label="Zoom out" title="Zoom out">
        <Minus className="h-4 w-4" />
      </button>
      <button type="button" className={button} onClick={fitRoute} aria-label="Fit route in view" title="Fit route in view">
        <Maximize2 className="h-4 w-4" />
      </button>
    </div>
  )
}

export default function PredictionMap({
  departureAirport,
  arrivalAirport,
  theme,
  airports,
  riskColor,
}: PredictionMapProps) {
  const departure = AIRPORTS_BY_CODE[departureAirport]
  const arrival = AIRPORTS_BY_CODE[arrivalAirport]
  const routeColor = riskColor ?? '#38bdf8'
  const dark = theme === 'dark'

  const [mapStyle, setMapStyle] = useState<MapStyle>(initialStyle)
  const [notice, setNotice] = useState('')

  const chooseStyle = (style: MapStyle) => {
    setMapStyle(style)
    setNotice('')
    try {
      localStorage.setItem(STYLE_KEY, style)
    } catch {
      // not saved; the map still works
    }
  }

  const fallBackToOffline = useMemo(
    () => () => {
      setMapStyle('offline')
      setNotice('Online map could not be loaded, so the offline map is shown.')
    },
    [],
  )

  const stateStyle = dark
    ? { color: '#334155', weight: 1, fillColor: '#16213a', fillOpacity: 1 }
    : { color: '#94a3b8', weight: 1, fillColor: '#e2e8f0', fillOpacity: 1 }

  const points = useMemo(
    () =>
      departure && arrival && departure.code !== arrival.code
        ? arcPoints([departure.lat, departure.lon], [arrival.lat, arrival.lon])
        : null,
    [departure, arrival],
  )

  return (
    <div className="relative overflow-hidden rounded-[1.75rem] border border-white/10 shadow-2xl">
      <MapContainer
        bounds={USA_BOUNDS}
        scrollWheelZoom={false}
        zoomControl={false}
        className="prediction-map h-[460px] w-full lg:h-[520px]"
        style={{ background: dark ? 'radial-gradient(circle at 50% 40%, #0f1b33, #070d1a 75%)' : '#f8fafc' }}
      >
        {mapStyle === 'offline' ? (
          <GeoJSON key={theme} data={usStates as GeoJsonObject} style={() => stateStyle} interactive={false} />
        ) : (
          <OnlineTiles style={mapStyle} onFail={fallBackToOffline} />
        )}

        <FitMapToSelection points={points} />
        <ZoomControls points={points} />

        {/* other airports in the data: small dots with a tooltip */}
        {airports
          .filter((a) => a.code !== departureAirport && a.code !== arrivalAirport)
          .map((airport) => (
            <CircleMarker
              key={airport.code}
              center={[airport.lat, airport.lon]}
              radius={5}
              pathOptions={{ color: '#38bdf8', weight: 1, fillColor: '#38bdf8', fillOpacity: 0.45 }}
            >
              <Tooltip direction="top" offset={[0, -6]} opacity={1}>
                <div className="space-y-1">
                  <p className="text-xs font-semibold">{airport.code}</p>
                  <p className="text-[11px]">{airport.city}</p>
                  <p className="text-[11px] opacity-80">{airport.name}</p>
                </div>
              </Tooltip>
            </CircleMarker>
          ))}

        <Pane name="routes" style={{ zIndex: 420 }}>
          {points && (
            <>
              {/* soft glow under the route, coloured by the predicted risk */}
              <Polyline positions={points} pathOptions={{ color: routeColor, weight: 14, opacity: 0.18, lineCap: 'round' }} />
              {/* the full planned route, dashes moving towards the destination */}
              <Polyline
                positions={points}
                className="route-dash"
                pathOptions={{ color: routeColor, weight: 2.5, opacity: 0.6, dashArray: '2 10', lineCap: 'round' }}
              />
            </>
          )}
        </Pane>

        {points && <FlyingPlane points={points} color={routeColor} />}

        {departure && (
          <Marker position={[departure.lat, departure.lon]} icon={airportIcon(departure.code, DEPARTURE_COLOR, departure.city)} />
        )}
        {arrival && arrival.code !== departure?.code && (
          <Marker position={[arrival.lat, arrival.lon]} icon={airportIcon(arrival.code, ARRIVAL_COLOR, arrival.city)} />
        )}
      </MapContainer>

      <div className="pointer-events-none absolute left-4 top-4 z-[500] rounded-2xl bg-slate-950/80 px-4 py-3 text-white shadow-xl backdrop-blur">
        <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-cyan-200/80">Live route view</p>
        <p className="mt-2 text-sm font-semibold">
          {departureAirport} to {arrivalAirport}
        </p>
        <p className="mt-1 text-xs text-slate-300">
          {departure?.city} to {arrival?.city}
        </p>
      </div>

      <div
        className="absolute right-4 top-28 z-[500] flex rounded-full sm:top-4 bg-slate-950/80 p-1 text-xs shadow-xl backdrop-blur"
        role="radiogroup"
        aria-label="Map style"
      >
        {(Object.keys(STYLE_LABELS) as MapStyle[]).map((style) => (
          <button
            key={style}
            type="button"
            role="radio"
            aria-checked={mapStyle === style}
            onClick={() => chooseStyle(style)}
            className={`rounded-full px-3 py-1.5 font-medium transition ${
              mapStyle === style ? 'bg-cyan-400 text-slate-950' : 'text-slate-300 hover:text-white'
            }`}
          >
            {STYLE_LABELS[style]}
          </button>
        ))}
      </div>

      {notice && (
        <div className="absolute right-4 top-40 z-[500] max-w-xs sm:top-16 rounded-xl bg-amber-500/90 px-3 py-2 text-xs font-medium text-slate-950 shadow-lg">
          {notice}
        </div>
      )}

      <div className="pointer-events-none absolute bottom-4 left-4 z-[500] flex gap-4 rounded-2xl bg-slate-950/80 px-4 py-2 text-xs text-slate-200 backdrop-blur">
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full" style={{ background: DEPARTURE_COLOR }} /> Departure
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full" style={{ background: ARRIVAL_COLOR }} /> Arrival
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-0.5 w-4 rounded" style={{ background: routeColor }} /> Route (risk colour)
        </span>
      </div>
    </div>
  )
}
