// Reference data only (names and map coordinates) for the 20 airports in the training data.
// Delay statistics are NOT stored here: they come from the API, computed from the real dataset.
export interface AirportRecord {
  code: string
  name: string
  city: string
  state: string
  lat: number
  lon: number
}

export const AIRPORTS: AirportRecord[] = [
  { code: 'ATL', name: 'Hartsfield-Jackson Atlanta International', city: 'Atlanta', state: 'GA', lat: 33.6407, lon: -84.4277 },
  { code: 'BOS', name: 'Boston Logan International', city: 'Boston', state: 'MA', lat: 42.3656, lon: -71.0096 },
  { code: 'CLT', name: 'Charlotte Douglas International', city: 'Charlotte', state: 'NC', lat: 35.2144, lon: -80.9473 },
  { code: 'DCA', name: 'Ronald Reagan Washington National', city: 'Washington', state: 'DC', lat: 38.8512, lon: -77.0402 },
  { code: 'DEN', name: 'Denver International', city: 'Denver', state: 'CO', lat: 39.8561, lon: -104.6737 },
  { code: 'DFW', name: 'Dallas/Fort Worth International', city: 'Dallas', state: 'TX', lat: 32.8998, lon: -97.0403 },
  { code: 'DTW', name: 'Detroit Metropolitan Wayne County', city: 'Detroit', state: 'MI', lat: 42.2162, lon: -83.3554 },
  { code: 'EWR', name: 'Newark Liberty International', city: 'Newark', state: 'NJ', lat: 40.6895, lon: -74.1745 },
  { code: 'JFK', name: 'John F. Kennedy International', city: 'New York', state: 'NY', lat: 40.6413, lon: -73.7781 },
  { code: 'LAS', name: 'Harry Reid International', city: 'Las Vegas', state: 'NV', lat: 36.084, lon: -115.1537 },
  { code: 'LAX', name: 'Los Angeles International', city: 'Los Angeles', state: 'CA', lat: 33.9416, lon: -118.4085 },
  { code: 'LGA', name: 'LaGuardia', city: 'New York', state: 'NY', lat: 40.7769, lon: -73.874 },
  { code: 'MCO', name: 'Orlando International', city: 'Orlando', state: 'FL', lat: 28.4312, lon: -81.3081 },
  { code: 'MIA', name: 'Miami International', city: 'Miami', state: 'FL', lat: 25.7959, lon: -80.287 },
  { code: 'IAH', name: 'George Bush Intercontinental', city: 'Houston', state: 'TX', lat: 29.9902, lon: -95.3368 },
  { code: 'MSP', name: 'Minneapolis-Saint Paul International', city: 'Minneapolis', state: 'MN', lat: 44.8848, lon: -93.2223 },
  { code: 'ORD', name: "O'Hare International", city: 'Chicago', state: 'IL', lat: 41.9742, lon: -87.9073 },
  { code: 'PHX', name: 'Phoenix Sky Harbor International', city: 'Phoenix', state: 'AZ', lat: 33.4352, lon: -112.0101 },
  { code: 'SEA', name: 'Seattle-Tacoma International', city: 'Seattle', state: 'WA', lat: 47.4502, lon: -122.3088 },
  { code: 'SFO', name: 'San Francisco International', city: 'San Francisco', state: 'CA', lat: 37.6213, lon: -122.379 },
  { code: 'SLC', name: 'Salt Lake City International', city: 'Salt Lake City', state: 'UT', lat: 40.7899, lon: -111.9791 },
]

export const AIRPORTS_BY_CODE = Object.fromEntries(
  AIRPORTS.map((airport) => [airport.code, airport]),
) as Record<string, AirportRecord>

// Great-circle distance in miles, used only if a route is missing from the training data.
export function haversineMiles(a: AirportRecord, b: AirportRecord): number {
  const toRad = (d: number) => (d * Math.PI) / 180
  const dLat = toRad(b.lat - a.lat)
  const dLon = toRad(b.lon - a.lon)
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(toRad(a.lat)) * Math.cos(toRad(b.lat)) * Math.sin(dLon / 2) ** 2
  return Math.round(2 * 3958.8 * Math.asin(Math.sqrt(h)))
}
