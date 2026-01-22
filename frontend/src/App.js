import React, { useState, useRef, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { TrendingUp, TrendingDown, Search, Loader2, X } from 'lucide-react';

const STOCK_SUGGESTIONS = [
  { symbol: 'AAPL', name: 'Apple Inc.' },
  { symbol: 'GOOGL', name: 'Alphabet Inc.' },
  { symbol: 'MSFT', name: 'Microsoft Corp.' },
  { symbol: 'TSLA', name: 'Tesla Inc.' },
  { symbol: 'AMZN', name: 'Amazon.com Inc.' },
  { symbol: 'META', name: 'Meta Platforms' },
  { symbol: 'NVDA', name: 'NVIDIA Corp.' },
  { symbol: 'JPM', name: 'JPMorgan Chase' },
  { symbol: 'V', name: 'Visa Inc.' },
  { symbol: 'JNJ', name: 'Johnson & Johnson' },
  { symbol: 'TCS.NS', name: 'Tata Consultancy' },
  { symbol: 'RELIANCE.NS', name: 'Reliance Industries' },
  { symbol: 'INFY.NS', name: 'Infosys Limited' },
  { symbol: 'HDFC.NS', name: 'HDFC Bank' },
  { symbol: 'BTC-USD', name: 'Bitcoin' },
  { symbol: 'ETH-USD', name: 'Ethereum' },
];

export default function StockPredictor() {
  const [ticker, setTicker] = useState('');
  const [days, setDays] = useState(30);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const searchRef = useRef(null);

  const handleClickOutside = (e) => {
    if (searchRef.current && !searchRef.current.contains(e.target)) {
      setShowSuggestions(false);
    }
  };

  useEffect(() => {
    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, []);

  const handleSearchChange = (e) => {
    const value = e.target.value.toUpperCase();
    setTicker(value);

    if (value.length > 0) {
      const filtered = STOCK_SUGGESTIONS.filter(s =>
        s.symbol.includes(value) || s.name.toUpperCase().includes(value)
      );
      setSuggestions(filtered);
      setShowSuggestions(true);
    } else {
      setSuggestions([]);
      setShowSuggestions(false);
    }
  };

  const handleSuggestionClick = (symbol) => {
    setTicker(symbol);
    setShowSuggestions(false);
    setSuggestions([]);
  };

  const handlePredict = async () => {
    if (!ticker.trim()) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const res = await fetch('http://localhost:8000/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticker: ticker.toUpperCase(),
          days_ahead: parseInt(days)
        })
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({ detail: 'Unable to fetch data. Check ticker symbol.' }));
        throw new Error(errorData.detail || 'Unable to fetch data. Check ticker symbol.');
      }
      
      const result = await res.json();
      
      // Validate response data
      if (!result || !result.predictions || !Array.isArray(result.predictions)) {
        throw new Error('Invalid response format from server');
      }

      // Format data for display
      const formatted = {
        ticker: result.ticker,
        current_price: result.current_price,
        predicted_price: result.predicted_price,
        price_change: result.price_change,
        price_change_percent: result.price_change_percent,
        predictions: result.predictions.map((p, i) => ({
          date: new Date(new Date().getTime() + (i + 1) * 24 * 60 * 60 * 1000).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
          predicted_price: p
        })),
        recent_prices: result.recent_prices || []
      };

      setData(formatted);
      setSuggestions([]);
      setShowSuggestions(false);
    } catch (err) {
      setError(err.message);
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handlePredict();
    }
  };

  const chartData = () => {
    if (!data) return [];
    
    const recentPrices = data.recent_prices || [];
    const recent = recentPrices.slice(-60).map((price, idx) => ({
      date: `${idx - (recentPrices.length - 60)}d`,
      price: parseFloat(price) || 0,
      type: 'actual'
    }));

    const pred = (data.predictions || []).map(d => ({
      date: d.date || '',
      price: parseFloat(d.predicted_price) || 0,
      type: 'forecast'
    }));

    return [...recent.slice(-30), ...pred];
  };

  const stats = data ? {
    current: data.current_price,
    predicted: data.predicted_price,
    change: data.price_change,
    percent: data.price_change_percent
  } : null;

  return (
    <div className="min-h-screen bg-zinc-50">
      <div className="max-w-6xl mx-auto px-4 py-8">
        
        <div className="mb-12">
          <h1 className="text-4xl font-semibold text-zinc-900 mb-2">Stock Forecaster</h1>
          <p className="text-zinc-600">ML-powered price predictions using LSTM</p>
        </div>

        <div className="bg-white rounded-xl border border-zinc-200 p-6 mb-8">
          <div className="grid md:grid-cols-3 gap-4">
            <div ref={searchRef} className="relative">
              <label className="block text-sm font-medium text-zinc-700 mb-2">Ticker</label>
              <div className="relative">
                <input
                  type="text"
                  value={ticker}
                  onChange={handleSearchChange}
                  onKeyPress={handleKeyPress}
                  onFocus={() => ticker && setSuggestions(STOCK_SUGGESTIONS.filter(s => s.symbol.includes(ticker) || s.name.toUpperCase().includes(ticker))) && setShowSuggestions(true)}
                  placeholder="Search stocks..."
                  className="w-full px-4 py-2.5 border border-zinc-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
                <Search className="absolute right-3 top-8 w-5 h-5 text-zinc-400" />
              </div>

              {showSuggestions && suggestions.length > 0 && (
                <div className="absolute top-full left-0 right-0 mt-2 bg-white border border-zinc-200 rounded-lg shadow-lg z-50">
                  {suggestions.slice(0, 8).map(s => (
                    <div
                      key={s.symbol}
                      onClick={() => handleSuggestionClick(s.symbol)}
                      className="px-4 py-3 hover:bg-zinc-100 cursor-pointer border-b border-zinc-100 last:border-b-0"
                    >
                      <div className="font-semibold text-zinc-900">{s.symbol}</div>
                      <div className="text-sm text-zinc-500">{s.name}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-zinc-700 mb-2">Forecast Days</label>
              <input
                type="number"
                value={days}
                onChange={(e) => setDays(Math.min(30, Math.max(1, parseInt(e.target.value) || 1)))}
                onKeyPress={handleKeyPress}
                min="1"
                max="30"
                className="w-full px-4 py-2.5 border border-zinc-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            <div className="flex items-end">
              <button
                onClick={handlePredict}
                disabled={loading || !ticker.trim()}
                className="w-full px-6 py-2.5 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Loading
                  </>
                ) : 'Forecast'}
              </button>
            </div>
          </div>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-8 flex items-center justify-between">
            <span>{error}</span>
            <X className="w-4 h-4 cursor-pointer" onClick={() => setError(null)} />
          </div>
        )}

        {stats && (
          <>
            <div className="grid md:grid-cols-4 gap-4 mb-8">
              <div className="bg-white border border-zinc-200 rounded-lg p-5">
                <div className="text-sm text-zinc-600 mb-1">Current Price</div>
                <div className="text-2xl font-semibold text-zinc-900">${stats.current.toFixed(2)}</div>
              </div>

              <div className="bg-white border border-zinc-200 rounded-lg p-5">
                <div className="text-sm text-zinc-600 mb-1">{days}d Forecast</div>
                <div className="text-2xl font-semibold text-zinc-900">${stats.predicted.toFixed(2)}</div>
              </div>

              <div className="bg-white border border-zinc-200 rounded-lg p-5">
                <div className="text-sm text-zinc-600 mb-1 flex items-center gap-1">
                  Expected Change
                  {stats.change >= 0 ? 
                    <TrendingUp className="w-4 h-4 text-green-600" /> : 
                    <TrendingDown className="w-4 h-4 text-red-600" />
                  }
                </div>
                <div className={`text-2xl font-semibold ${stats.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {stats.change >= 0 ? '+' : ''}{stats.percent.toFixed(2)}%
                </div>
              </div>

              <div className="bg-white border border-zinc-200 rounded-lg p-5">
                <div className="text-sm text-zinc-600 mb-1">Price Change</div>
                <div className={`text-2xl font-semibold ${stats.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {stats.change >= 0 ? '+' : ''}${stats.change.toFixed(2)}
                </div>
              </div>
            </div>

            <div className="bg-white border border-zinc-200 rounded-lg p-6 mb-8">
              <div className="mb-6">
                <h2 className="text-lg font-semibold text-zinc-900">{data.ticker} Price Chart</h2>
                <p className="text-sm text-zinc-500 mt-1">Historical prices and forecast</p>
              </div>
              
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={chartData()}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis 
                    dataKey="date" 
                    stroke="#71717a"
                    tick={{ fill: '#71717a', fontSize: 12 }}
                  />
                  <YAxis 
                    stroke="#71717a"
                    tick={{ fill: '#71717a', fontSize: 12 }}
                    domain={['auto', 'auto']}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#ffffff',
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px',
                      fontSize: '14px'
                    }}
                    formatter={(value) => [`$${value.toFixed(2)}`, 'Price']}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="price" 
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-white border border-zinc-200 rounded-lg overflow-hidden">
              <div className="px-6 py-4 border-b border-zinc-200">
                <h2 className="text-lg font-semibold text-zinc-900">Forecast Breakdown</h2>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-zinc-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-zinc-700 uppercase tracking-wider">Date</th>
                      <th className="px-6 py-3 text-right text-xs font-medium text-zinc-700 uppercase tracking-wider">Price</th>
                      <th className="px-6 py-3 text-right text-xs font-medium text-zinc-700 uppercase tracking-wider">vs Today</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-200">
                    {data.predictions.map((p, i) => {
                      const diff = p.predicted_price - stats.current;
                      const pct = ((diff / stats.current) * 100).toFixed(2);
                      return (
                        <tr key={i} className="hover:bg-zinc-50">
                          <td className="px-6 py-3 text-sm text-zinc-900">{p.date}</td>
                          <td className="px-6 py-3 text-sm text-zinc-900 text-right font-mono">${p.predicted_price.toFixed(2)}</td>
                          <td className={`px-6 py-3 text-sm text-right font-medium ${diff >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {diff >= 0 ? '+' : ''}{pct}%
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}

        <div className="mt-8 text-center text-xs text-zinc-500">
          Predictions are for educational purposes only and should not be considered financial advice
        </div>
      </div>
    </div>
  );
}
