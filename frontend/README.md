# Stock Predictor Frontend (Vite + shadcn/ui)

Modern React + TypeScript frontend with Tailwind and shadcn/ui components.

## Setup

```bash
cd frontend
bun install
```

## Development

```bash
bun run dev
```

The dev server runs on http://localhost:5173 and proxies `/api` to `http://localhost:8000`.

## Build

```bash
bun run build
bun run preview
```

## Notes

- The frontend calls the backend via relative `/api` routes.
- Tailwind is configured via `@tailwindcss/vite`.
- Dark mode is enabled by default in `src/main.tsx`.
