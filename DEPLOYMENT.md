# Deploying VoxShield

Read the first section before you start. It will save you an hour.

---

## The one constraint that shapes everything

**The frontend goes on Vercel. The backend cannot.**

Vercel runs serverless functions with a **250 MB unzipped** limit and no GPU.
Measured on this machine:

| | Size |
| --- | --- |
| Vercel's limit per function | **250 MB** |
| `models/voxshield_antispoof.pt` | **363 MB** |
| PyTorch (installed) | 2,931 MB |
| Whole backend environment | 3,796 MB |

The **checkpoint alone is 45% over the limit** before a single dependency is
added. A CPU-only PyTorch build is smaller than the CUDA one here but still
several hundred megabytes — it does not come close to fitting.

Two more blockers, either of which would be fatal on its own:

- **No WebSockets.** Vercel serverless functions cannot hold a socket open, and
  the live call is a WebSocket. Edge functions can, but they cannot run Python.
- **Cold starts.** Loading the checkpoint takes ~6 seconds. Every cold
  invocation would pay that.

So the shape of the deployment is:

```
   Vercel (HTTPS)                    somewhere that can run PyTorch
   ┌──────────────────┐              ┌──────────────────────────────┐
   │  Next.js UI      │  ── HTTPS ─► │  FastAPI + torch + 363 MB    │
   │  static, global  │  ── WSS ───► │  checkpoint + ECAPA          │
   └──────────────────┘              └──────────────────────────────┘
```

---

## Step 1 — Deploy the frontend to Vercel

### 1.1 Push your branch

```powershell
git push origin karthik
```

### 1.2 Import the project

1. Go to **vercel.com/new**
2. Import `SathvikGuttula/NullBox`
3. **Set Root Directory to `frontend`.** This is the one setting that matters.
   Click *Edit* next to Root Directory and pick `frontend`. Without it, Vercel
   looks at the repository root, finds no `package.json`, and fails.
4. Framework Preset should auto-detect as **Next.js**. Leave build and output
   settings alone — the defaults are correct.

### 1.3 Set the environment variables

Under **Environment Variables**, add both, for **all three** environments
(Production, Preview, Development):

| Name | Value |
| --- | --- |
| `NEXT_PUBLIC_API_URL` | `https://your-backend-url` |
| `NEXT_PUBLIC_WS_URL` | `wss://your-backend-url` |

You will not have that URL until Step 2. Either do Step 2 first, or deploy now
with placeholders and redeploy afterwards — but see the warning below.

> **`NEXT_PUBLIC_*` variables are baked into the JavaScript bundle at build
> time, not read at runtime.** Verified: building with
> `NEXT_PUBLIC_API_URL=https://api.example.com` puts that literal string into
> `.next/static/chunks/app/page-*.js`.
>
> **Changing the variable in the dashboard does nothing until you redeploy.**
> After editing it, go to Deployments → ⋯ → **Redeploy**.

### 1.4 Deploy

Press Deploy. It takes about a minute. Verified from a clean checkout of
exactly the 20 files git would hand Vercel: `npm ci` resolves from the lockfile
and `next build` succeeds.

If you leave the variables unset the UI still deploys and runs — it falls back
to `http://localhost:8000` and shows **"Backend unreachable"** with the fix on
screen. That is a legitimate mode: the deployed UI driving a backend on your
own laptop. See the HTTPS note in Step 3.

---

## Step 2 — Make the backend reachable

Pick one. For a hackathon demo, option A is the right answer and takes about
two minutes.

### Option A — Tunnel from your laptop (recommended)

Keeps the RTX 3050, so inference stays at ~30 ms instead of ~300 ms on a free
CPU host. Gives you real HTTPS and WSS, which is what Vercel needs.

```powershell
# once
winget install --id Cloudflare.cloudflared

# every time you demo — start the backend first
cd C:\Users\Karthikeya Varma\voxshield
.\start.ps1 -BackendOnly

# then, in a second terminal
cloudflared tunnel --url http://localhost:8000
```

It prints a URL like `https://random-words-here.trycloudflare.com`. That is
your backend URL. It supports WebSockets, so the live call works.

**Trade-off:** the URL changes every restart, and your laptop must stay on and
awake. For a judged demo that is fine. For anything longer, a named tunnel with
a Cloudflare account gives a stable hostname.

### Option B — Hugging Face Spaces (free, persistent, CPU)

Best if you need a URL that survives your laptop closing.

1. Create a Space, SDK **Docker**, hardware **CPU basic** (free).
2. Add a `Dockerfile` that installs **CPU-only** PyTorch:
   `pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio`
   — the CUDA build in this repo is 2.9 GB and pointless without a GPU.
3. The 363 MB checkpoint cannot go in a normal git repo. Either push it with
   Git LFS, or upload it to a HF model repo and download it at container start.
4. Expose port 7860 and run
   `uvicorn app.main:app --host 0.0.0.0 --port 7860`.

Expect roughly **10× slower** inference. Everything still works; the live call
just updates less crisply.

### Option C — Render / Railway / Fly.io

Workable but **not on free tiers** — PyTorch plus wav2vec2 plus ECAPA needs
around 2 GB of RAM, and Render's free instance has 512 MB. Budget for the
smallest paid tier.

---

## Step 3 — Wire the two together

### 3.1 Tell the backend to accept the Vercel origin

A browser will not let your Vercel page call the API unless the API says the
origin is allowed. Create `backend/.env`:

```ini
CORS_ORIGINS=http://localhost:3000,https://YOUR-PROJECT.vercel.app

# Vercel gives every preview deployment its own hostname, so an exact list
# breaks on every push. This covers them all:
CORS_ORIGIN_REGEX=^https://[a-z0-9-]+\.vercel\.app$
```

Restart the backend. Without this you get a CORS error in the browser console
and every request fails, while the backend logs look completely healthy.

### 3.2 Point the frontend at the backend

Set both variables in Vercel to the URL from Step 2, then **redeploy**:

| Name | Example |
| --- | --- |
| `NEXT_PUBLIC_API_URL` | `https://random-words.trycloudflare.com` |
| `NEXT_PUBLIC_WS_URL` | `wss://random-words.trycloudflare.com` |

Note `wss://`, not `ws://`. An HTTPS page cannot open an insecure socket.

---

## The HTTPS trap — read this one

Vercel always serves HTTPS. Browsers block an HTTPS page from calling a plain
`http://` address. This is the single most likely thing to break your demo.

| Backend URL | Works from Vercel? |
| --- | --- |
| `https://…trycloudflare.com` | **Yes** |
| `http://localhost:8000` | Yes in Chrome, Edge and Firefox — `localhost` is treated as a trusted origin. **No in Safari.** And only on the machine running the backend. |
| `http://203.0.113.10:8000` | **No.** Blocked as mixed content. |
| `http://my-server.com` | **No.** Blocked as mixed content. |

If the UI shows "Backend unreachable" but the backend is definitely running,
open the browser console. `Mixed Content: The page at 'https://…' was loaded
over HTTPS, but requested an insecure resource` means exactly this. The fix is
HTTPS on the backend, which is what the tunnel in Option A gives you.

---

## Step 4 — Verify

1. Open your Vercel URL.
2. Go to the **System** tab. It should say **Ready**, show the checkpoint at
   362.6 MB and **Calibrated: yes**. If it says Offline, the problem is in
   Step 3, not Step 1.
3. Go to **Analyse**, upload any file from `datasets\test-audio\spoof\`, press
   Analyse. Expect `>99.9%` synthetic, risk 100, HIGH RISK WORKFLOW.
4. Go to **Live call** and press Start. Grant microphone access. If the risk
   updates once a second, the WebSocket path works too.

Microphone access requires HTTPS or localhost. Vercel is HTTPS, so this is
fine — but it is why the live call cannot be tested over a plain-HTTP host.

---

## What will actually go wrong

| Symptom | Cause | Fix |
| --- | --- | --- |
| Build fails, "no package.json" | Root Directory not set | Set it to `frontend` |
| Deploys, but always "Backend unreachable" | Env vars set after the build | Redeploy — `NEXT_PUBLIC_*` is baked in at build time |
| Console: `Mixed Content … insecure resource` | Backend is HTTP | Use the tunnel from Option A |
| Console: `blocked by CORS policy` | Vercel origin not allowlisted | Set `CORS_ORIGINS` in `backend/.env`, restart backend |
| Everything works, live call does not | `ws://` instead of `wss://` | Fix `NEXT_PUBLIC_WS_URL`, redeploy |
| Preview deployments break, production fine | Preview hostnames are not in `CORS_ORIGINS` | Set `CORS_ORIGIN_REGEX` |
| Backend URL stops working next morning | Quick tunnels get a new URL each restart | Named Cloudflare tunnel, or Option B |
| Analysis times out first try | Cold start loading a 362 MB checkpoint | Expected, ~6 s. It is fast afterwards. |

---

## Honest limitations of a public deployment

State these if anyone asks — they are real and they are not hidden anywhere in
the product.

- **There is no authentication.** Every endpoint is open. Anyone with the URL
  can enrol a voice, verify against any profile, or delete someone else's
  profile. Do not put a public backend URL in a slide deck.
- **There is no rate limiting.** A public backend can be used to exhaust your
  GPU, or to brute-force voices against an enrolled profile.
- **Enrolled embeddings are stored unencrypted** in `models/speakers.json`.
  They are biometric data.
- Nothing about a call is persisted, so there is no audit trail of what a
  deployed instance decided.

For a judged demo on a tunnel that you shut down afterwards, these are
acceptable. For anything left running, they are not.
