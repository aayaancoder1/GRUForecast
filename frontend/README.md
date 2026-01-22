# Frontend Setup Instructions

## Quick Start

### 1. Install Dependencies
```bash
npm install
```

### 2. Start Development Server
```bash
npm start
```

App runs at: http://localhost:3000

---

## Build for Production

```bash
npm run build
```

Creates optimized build in `build/` folder.

---

## Configuration

### Update API URL

If backend is on a different port, update `API_URL` in `src/App.js`:

```javascript
const API_URL = 'http://localhost:8000/api';
```

---

## Tailwind CSS

Already configured with:
- Custom colors (primary, accent, success, danger)
- Glass morphism effects
- Custom animations
- Dark theme optimized

---

## Components

- **Header.js** - Navigation bar with branding
- **Footer.js** - Footer with links and disclaimer
- **PredictionForm.js** - Stock ticker input and prediction form
- **PredictionChart.js** - Recharts line chart for predictions
- **HistoricalChart.js** - Area chart for historical data
- **StatsCard.js** - Reusable stats card component

---

## Features

✨ Beautiful dark UI
📊 Interactive charts with Recharts
🎯 Real-time predictions
📈 Historical analysis
🎨 Glass morphism design
⚡ Responsive and fast

---

## Troubleshooting

**npm install stuck?**
```bash
npm install --legacy-peer-deps
```

**Port 3000 in use?**
```bash
# macOS/Linux
lsof -ti:3000 | xargs kill -9

# Windows
netstat -ano | findstr :3000
```

**API not connecting?**
- Check backend is running on port 8000
- Verify API_URL in App.js
- Check browser console for CORS errors

---

## Dependencies

- react: UI library
- react-dom: React DOM rendering
- recharts: Chart library
- lucide-react: Icons
- axios: HTTP client
- tailwindcss: Styling
- date-fns: Date utilities
