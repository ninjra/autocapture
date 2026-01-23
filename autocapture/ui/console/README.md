# Autocapture web console

Source for the web console. Build output is written to `../web` and served by FastAPI
at `/` with assets under `/static`.

## Commands

```bash
npm install
npm run dev
npm run build
```

## Dev workflow

- Start the backend: `poetry run autocapture api` (or `poetry run autocapture ui open`).
- Run the Vite dev server: `npm run dev`.
- API requests are proxied to `http://127.0.0.1:8000` (see `vite.config.ts`).
