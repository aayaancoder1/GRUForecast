export type PredictionRequest = {
  ticker: string
  days_ahead: number
}

export type PredictionResponse = {
  ticker: string
  current_price: number
  predicted_price: number
  predictions: number[]
  days_ahead: number
  recent_prices: number[]
  price_change: number
  price_change_percent: number
  timestamp: string
}

const API_BASE = '/api'

class ApiError extends Error {
  status?: number
  details?: string

  constructor(message: string, status?: number, details?: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.details = details
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const controller = new AbortController()
  const timeoutId = window.setTimeout(() => controller.abort(), 20000)

  try {
    const response = await fetch(`${API_BASE}${path}`, {
      headers: {
        'Content-Type': 'application/json',
      },
      signal: controller.signal,
      ...options,
    })

    const contentType = response.headers.get('content-type') || ''
    const isJson = contentType.includes('application/json')
    const data = isJson ? await response.json() : await response.text()

    if (!response.ok) {
      const detail =
        typeof data === 'object' && data?.detail ? data.detail : undefined
      throw new ApiError('Request failed', response.status, detail)
    }

    return data as T
  } finally {
    window.clearTimeout(timeoutId)
  }
}

export async function predictStock(
  payload: PredictionRequest
): Promise<PredictionResponse> {
  return request<PredictionResponse>('/predict', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export { ApiError }
