# Server Sizing Frontend

Next.js application for LLM hardware and application infrastructure sizing. DGX Spark is available in the GPU catalog.

The LLM workflow estimates weight, full/sliding/linear attention cache, runtime, capacity-floor GPU count, and per-device memory placement. Input-token presets cover 2K through 256K with a custom option. Capacity-floor results are aggregate planning estimates and do not certify a multi-GPU partition or runtime compatibility.

## Local Setup

```bash
npm ci
copy .env.example .env
npm run db:generate
npm run dev
```

The repository includes `data/model_dashboard.sqlite` with the current catalog. The application is available at `http://localhost:3000` without a sign-in step.

## Validation

```bash
npm test
npm run lint
npx tsc --noEmit
npm run build
```

The browser smoke test uses Microsoft Edge by default:

```bash
npm run test:browser
```

Set `PLAYWRIGHT_CHANNEL` to another installed Playwright browser channel when needed.

## Metadata

Model Data supports previewing and importing public Hugging Face repositories by repository ID, for example `Qwen/Qwen3.6-35B-A3B`. Imports combine a revision-pinned `config.json` with the compact Hugging Face model API response for topology, supported weight precision metadata, exact safetensors parameter totals, and revision. NVFP4 and MXFP4 are treated as 4-bit weight estimates with separate packing/runtime overhead. KV-cache precision remains independent and is left unset unless explicitly verified. Active MoE parameters are left empty unless an authoritative source provides them.

```bash
npm run metadata:refresh
```

`metadata:refresh` updates supported topology from pinned Hugging Face configurations and reports failures without stopping the complete batch.

## Docker

The Compose service mounts the committed database directory at `/app/data`.

```bash
docker compose up -d --build
```

Docker publishes the application only on `127.0.0.1:3000`.
