import { useEffect, useMemo, useState } from 'react'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip as ChartTooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { TrendingDown, TrendingUp } from 'lucide-react'

import { ApiError, predictStock, type PredictionResponse } from '@/lib/api'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { Skeleton } from '@/components/ui/skeleton'

const daysAheadOptions = ['1', '3', '5', '7', '14', '30']
const quickTickers = ['MSFT', 'AAPL', 'NVDA', 'TSLA', 'GOOGL', 'AMZN']

function formatCurrency(value: number) {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 2,
  }).format(value)
}

function formatPercent(value: number) {
  const sign = value >= 0 ? '+' : ''
  return `${sign}${value.toFixed(2)}%`
}

function formatNumber(value: number) {
  return new Intl.NumberFormat('en-US', {
    maximumFractionDigits: 2,
  }).format(value)
}

function validatePrediction(result: PredictionResponse): string | null {
  const values = [
    result.current_price,
    result.predicted_price,
    result.price_change,
    result.price_change_percent,
    ...result.predictions,
    ...result.recent_prices,
  ]
  if (values.some((value) => !Number.isFinite(value))) {
    return 'Prediction returned invalid numeric values.'
  }
  if (result.current_price <= 0 || result.predicted_price <= 0) {
    return 'Prediction returned a non-positive price.'
  }
  const absChangePercent = Math.abs(result.price_change_percent)
  if (absChangePercent > 200) {
    return 'Prediction change is unrealistically large. Try another ticker or horizon.'
  }
  return null
}

function App() {
  const [ticker, setTicker] = useState('MSFT')
  const [daysAhead, setDaysAhead] = useState('5')
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [history, setHistory] = useState<PredictionResponse[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const stored = localStorage.getItem('prediction-history')
    if (stored) {
      try {
        const parsed = JSON.parse(stored) as PredictionResponse[]
        setHistory(parsed)
      } catch {
        localStorage.removeItem('prediction-history')
      }
    }
  }, [])

  const chartData = useMemo(() => {
    if (!result) return []
    const recent = result.recent_prices.map((price, index) => ({
      label: `D-${result.recent_prices.length - index}`,
      price,
      forecast: null,
    }))
    const forecasts = result.predictions.map((price, index) => ({
      label: `P${index + 1}`,
      price: null,
      forecast: price,
    }))
    return [...recent, ...forecasts]
  }, [result])

  const delta = result ? result.price_change : 0
  const deltaUp = delta >= 0
  const rangeStats = useMemo(() => {
    if (!result) return null
    const min = Math.min(...result.recent_prices)
    const max = Math.max(...result.recent_prices)
    return { min, max }
  }, [result])

  const averagePrice = useMemo(() => {
    if (!result) return null
    const sum = result.recent_prices.reduce((acc, value) => acc + value, 0)
    return sum / result.recent_prices.length
  }, [result])

  const predictionDeltas = useMemo(() => {
    if (!result) return []
    return result.predictions.map((price, index) => {
      const deltaValue = price - result.current_price
      const deltaPercent = (deltaValue / result.current_price) * 100
      return {
        label: `P${index + 1}`,
        price,
        deltaValue,
        deltaPercent,
      }
    })
  }, [result])

  const onSubmit = async (event: React.FormEvent) => {
    event.preventDefault()
    setError(null)
    setLoading(true)
    try {
      const payload = {
        ticker: ticker.trim().toUpperCase(),
        days_ahead: Number(daysAhead),
      }
      const response = await predictStock(payload)
      const validationError = validatePrediction(response)
      if (validationError) {
        setResult(null)
        setError(validationError)
        return
      }
      setResult(response)
      setHistory((prev) => {
        const next = [response, ...prev].slice(0, 5)
        localStorage.setItem('prediction-history', JSON.stringify(next))
        return next
      })
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.details || 'Prediction request failed.')
      } else if (err instanceof Error) {
        setError(err.message)
      } else {
        setError('Something went wrong.')
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <TooltipProvider>
      <div className="min-h-screen bg-background text-foreground">
        <div className="mx-auto flex max-w-6xl flex-col gap-8 px-6 py-10">
          <header className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-[0.3em] text-muted-foreground">
                  Stock Predictor
                </span>
                <Badge variant="secondary">2026 UI</Badge>
              </div>
              <h1 className="text-3xl font-semibold md:text-4xl">
                LSTM Market Forecasts
              </h1>
              <p className="text-sm text-muted-foreground">
                Predict next-day to 30-day prices with a modern, dark-first
                experience.
              </p>
            </div>
            <div className="flex items-center gap-3">
              <Badge variant="outline">API: /api</Badge>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Badge variant="secondary">Backend Ready</Badge>
                </TooltipTrigger>
                <TooltipContent>FastAPI backend is expected at /api.</TooltipContent>
              </Tooltip>
            </div>
          </header>

          <div className="grid gap-6 lg:grid-cols-[380px_1fr]">
            <Card className="h-fit">
              <CardHeader>
                <CardTitle>Prediction Console</CardTitle>
                <CardDescription>
                  Enter a ticker and forecast horizon to get a new prediction.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <form className="space-y-5" onSubmit={onSubmit}>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Ticker</label>
                    <Input
                      value={ticker}
                      onChange={(event) => setTicker(event.target.value)}
                      placeholder="AAPL, MSFT, TCS.NS"
                    />
                    <p className="text-xs text-muted-foreground">
                      Example tickers: AAPL, MSFT, TCS.NS, NVDA, TSLA
                    </p>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Days Ahead</label>
                    <Select value={daysAhead} onValueChange={setDaysAhead}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select" />
                      </SelectTrigger>
                      <SelectContent>
                        {daysAheadOptions.map((value) => (
                          <SelectItem key={value} value={value}>
                            {value} days
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <Button
                    className="w-full border border-input transition-transform hover:translate-y-[-1px] active:translate-y-0"
                    type="submit"
                    disabled={loading}
                  >
                    {loading ? 'Running prediction...' : 'Predict'}
                  </Button>
                  <div className="flex flex-wrap gap-2">
                    {quickTickers.map((symbol) => (
                      <Button
                        key={symbol}
                        type="button"
                        variant="outline"
                        size="sm"
                        className="transition hover:border-primary/60 hover:text-foreground"
                        onClick={() => setTicker(symbol)}
                      >
                        {symbol}
                      </Button>
                    ))}
                  </div>
                  {error ? (
                    <div className="rounded-md border border-destructive/50 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                      {error}
                    </div>
                  ) : null}
                </form>
                <Separator className="my-6" />
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-medium">Recent Runs</p>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => {
                        setHistory([])
                        localStorage.removeItem('prediction-history')
                      }}
                    >
                      Clear
                    </Button>
                  </div>
                  {history.length === 0 ? (
                    <p className="text-xs text-muted-foreground">
                      No recent predictions yet.
                    </p>
                  ) : (
                    <div className="space-y-2">
                      {history.map((item) => (
                        <button
                          key={item.timestamp}
                          className="flex w-full items-center justify-between rounded-md border px-3 py-2 text-left text-sm transition hover:bg-accent"
                          onClick={() => {
                            setResult(item)
                            setTicker(item.ticker)
                            setDaysAhead(String(item.days_ahead))
                          }}
                        >
                          <div>
                            <p className="font-medium">{item.ticker}</p>
                            <p className="text-xs text-muted-foreground">
                              {new Date(item.timestamp).toLocaleString()}
                            </p>
                          </div>
                          <span className="text-xs text-muted-foreground">
                            {formatPercent(item.price_change_percent)}
                          </span>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Forecast Overview</CardTitle>
                <CardDescription>
                  Latest predictions, price change, and timeline.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {loading ? (
                  <div className="space-y-4">
                    <div className="flex gap-4">
                      <Skeleton className="h-16 w-40" />
                      <Skeleton className="h-16 w-40" />
                    </div>
                    <Skeleton className="h-6 w-36" />
                    <Skeleton className="h-[260px] w-full" />
                  </div>
                ) : !result ? (
                  <div className="flex min-h-[240px] items-center justify-center rounded-lg border border-dashed border-muted-foreground/30">
                    <p className="text-sm text-muted-foreground">
                      Run a prediction to see insights here.
                    </p>
                  </div>
                ) : (
                  <>
                    <div className="flex flex-wrap items-center gap-6">
                      <div>
                        <p className="text-xs uppercase tracking-[0.3em] text-muted-foreground">
                          Current Price (Close)
                        </p>
                        <p className="text-3xl font-semibold">
                          {formatCurrency(result.current_price)}
                        </p>
                      </div>
                      <div>
                        <p className="text-xs uppercase tracking-[0.3em] text-muted-foreground">
                          Predicted Price (P1)
                        </p>
                        <p className="text-3xl font-semibold">
                          {formatCurrency(result.predicted_price)}
                        </p>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant={deltaUp ? 'default' : 'destructive'}>
                          {formatPercent(result.price_change_percent)}
                        </Badge>
                        {deltaUp ? (
                          <TrendingUp className="h-5 w-5 text-emerald-400" />
                        ) : (
                          <TrendingDown className="h-5 w-5 text-rose-400" />
                        )}
                      </div>
                      <div className="flex flex-col gap-1 text-xs text-muted-foreground">
                        <span>{result.ticker}</span>
                        <span>{new Date(result.timestamp).toLocaleString()}</span>
                      </div>
                    </div>

                    <Separator />

                    <div className="grid gap-3 md:grid-cols-3">
                      <div className="rounded-lg border px-3 py-2">
                        <p className="text-xs text-muted-foreground">Prediction Horizon</p>
                        <p className="text-lg font-semibold">{result.days_ahead} days</p>
                        <p className="text-xs text-muted-foreground">
                          Next {result.days_ahead} trading days
                        </p>
                      </div>
                      <div className="rounded-lg border px-3 py-2">
                        <p className="text-xs text-muted-foreground">Original Range (30d)</p>
                        <p className="text-lg font-semibold">
                          {rangeStats
                            ? `${formatCurrency(rangeStats.min)} – ${formatCurrency(
                                rangeStats.max
                              )}`
                            : '—'}
                        </p>
                        <p className="text-xs text-muted-foreground">Last 30 closes</p>
                      </div>
                      <div className="rounded-lg border px-3 py-2">
                        <p className="text-xs text-muted-foreground">Delta (P1 - Current)</p>
                        <p className="text-lg font-semibold">
                          {formatCurrency(result.price_change)}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {formatPercent(result.price_change_percent)}
                        </p>
                      </div>
                    </div>

                    <Tabs defaultValue="chart" className="w-full">
                      <TabsList>
                        <TabsTrigger value="chart">Chart</TabsTrigger>
                        <TabsTrigger value="recent">Original Data</TabsTrigger>
                        <TabsTrigger value="forecast">Prediction Deltas</TabsTrigger>
                      </TabsList>
                      <TabsContent value="chart">
                        <div className="space-y-3">
                          <p className="text-xs text-muted-foreground">
                            Solid line: historical closes (last 30). Dashed line: model forecast.
                          </p>
                          <div className="h-[340px] w-full rounded-lg border bg-card/50 p-3">
                            <ResponsiveContainer width="100%" height="100%">
                              <LineChart
                                data={chartData}
                                margin={{ top: 16, right: 24, left: 12, bottom: 8 }}
                              >
                                <CartesianGrid
                                  stroke="hsl(var(--border))"
                                  strokeDasharray="4 4"
                                  opacity={0.4}
                                />
                                <XAxis
                                  dataKey="label"
                                  tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 11 }}
                                  tickLine={false}
                                  axisLine={false}
                                />
                                <YAxis
                                  domain={['dataMin - 5', 'dataMax + 5']}
                                  tickFormatter={(value) => formatCurrency(value)}
                                  tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 11 }}
                                  width={90}
                                  axisLine={false}
                                  tickLine={false}
                                />
                                <ChartTooltip
                                  formatter={(value: number) => formatCurrency(value)}
                                  contentStyle={{
                                    background: 'hsl(var(--popover))',
                                    borderRadius: '8px',
                                    borderColor: 'hsl(var(--border))',
                                  }}
                                  labelStyle={{ color: 'hsl(var(--muted-foreground))' }}
                                />
                                <Legend
                                  formatter={(value) =>
                                    value === 'price'
                                      ? 'Historical Close'
                                      : 'Forecast'
                                  }
                                />
                                {averagePrice !== null ? (
                                  <ReferenceLine
                                    y={averagePrice}
                                    stroke="hsl(var(--muted-foreground))"
                                    strokeDasharray="6 6"
                                    label={{
                                      value: `Avg ${formatCurrency(averagePrice)}`,
                                      fill: 'hsl(var(--muted-foreground))',
                                      fontSize: 11,
                                      position: 'insideTopLeft',
                                    }}
                                  />
                                ) : null}
                                <Line
                                  type="monotone"
                                  dataKey="price"
                                  stroke="hsl(var(--primary))"
                                  strokeWidth={2}
                                  dot={false}
                                />
                                <Line
                                  type="monotone"
                                  dataKey="forecast"
                                  stroke="hsl(var(--accent-foreground))"
                                  strokeDasharray="6 4"
                                  strokeWidth={2}
                                  dot={false}
                                />
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                        </div>
                      </TabsContent>
                      <TabsContent value="recent">
                        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                          {result.recent_prices.slice(-12).map((price, index) => (
                            <div
                              key={`${price}-${index}`}
                              className="rounded-lg border px-3 py-2"
                            >
                              <p className="text-xs text-muted-foreground">
                                Close D-{result.recent_prices.length - index}
                              </p>
                              <p className="text-lg font-semibold">
                                {formatCurrency(price)}
                              </p>
                              <p className="text-xs text-muted-foreground">
                                {formatNumber(price)}
                              </p>
                            </div>
                          ))}
                        </div>
                      </TabsContent>
                      <TabsContent value="forecast">
                        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                          {predictionDeltas.map((item) => (
                            <div
                              key={item.label}
                              className="rounded-lg border px-3 py-2"
                            >
                              <p className="text-xs text-muted-foreground">
                                {item.label} Prediction
                              </p>
                              <p className="text-lg font-semibold">
                                {formatCurrency(item.price)}
                              </p>
                              <p className="text-xs text-muted-foreground">
                                {formatCurrency(item.deltaValue)} (
                                {formatPercent(item.deltaPercent)})
                              </p>
                            </div>
                          ))}
                        </div>
                      </TabsContent>
                    </Tabs>

                    <div className="flex flex-wrap gap-2">
                      <Button
                        type="button"
                        variant="outline"
                        className="transition hover:border-primary/60 hover:text-foreground"
                        onClick={() => navigator.clipboard.writeText(JSON.stringify(result, null, 2))}
                      >
                        Copy JSON
                      </Button>
                      <Button
                        type="button"
                        variant="secondary"
                        className="transition hover:bg-secondary/80"
                        onClick={() => setResult(null)}
                      >
                        Clear Result
                      </Button>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </TooltipProvider>
  )
}

export default App
