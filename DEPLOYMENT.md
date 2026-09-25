# 🚀 Credence 2.0 — Dual Cloud Deployment Guide
### Backend on Render + Frontend on Vercel

This guide provides the exact steps to deploy your Credence 2.0 system with the **FastAPI ML Decision Engine hosted on Render** and the **Glassmorphic Single Page Application hosted on Vercel**.

---

## 🛠️ Step 1: Push Your Code to GitHub

Open PowerShell in `c:\Users\Admin\Desktop\paytm` and run:

```bash
git add .
git commit -m "feat: configure dual deployment for Render backend and Vercel frontend"
git push origin main
```

*(If you haven't linked your GitHub repository yet:)*
```bash
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git
git branch -M main
git push -u origin main
```

---

## 🐍 Step 2: Deploy Backend to Render (First)

You need to deploy the backend first to get your public API URL.

1. Go to [https://dashboard.render.com](https://dashboard.render.com) and log in.
2. Click **New +** → **Web Service**.
3. Select your GitHub repository.
4. Fill in the deployment details:
   - **Name**: `credence-backend` (or your choice)
   - **Region**: Choose closest to you (e.g., Singapore or Frankfurt)
   - **Branch**: `main`
   - **Root Directory**: `.` (leave blank / root)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: `Free`
5. Click **Advanced** and verify these environment variables:
   - `PYTHON_VERSION`: `3.11.0`
   - `DB_PATH`: `users.db`
6. Click **Create Web Service**.
7. Wait 2–3 minutes for the build to finish. Once live, note down your Render URL:
   > 🔗 **Example Render URL**: `https://credence-backend-xyz.onrender.com`
8. Verify it works by opening in your browser:
   `https://credence-backend-xyz.onrender.com/health`
   You should see:
   ```json
   {"status":"healthy","service":"Credence Underwriting Portal","version":"2.0.0"}
   ```

---

## ⚡ Step 3: Link Render URL & Deploy Frontend to Vercel

### Step 3.1: Update the Render Backend URL in `vercel.json`
Open [`frontend/vercel.json`](file:///c:/Users/Admin/Desktop/paytm/frontend/vercel.json) and [`vercel.json`](file:///c:/Users/Admin/Desktop/paytm/vercel.json):
Replace `https://YOUR-RENDER-BACKEND.onrender.com` with your actual Render URL (e.g. `https://credence-backend-xyz.onrender.com`).

Commit and push the update:
```bash
git add .
git commit -m "chore: set live render backend url"
git push origin main
```

### Step 3.2: Deploy on Vercel
1. Go to [https://vercel.com/dashboard](https://vercel.com/dashboard) and log in.
2. Click **Add New...** → **Project**.
3. Import your GitHub repository.
4. Under **Configure Project**:
   - **Framework Preset**: `Other`
   - **Root Directory**: Select `frontend` (or leave `./` since root `vercel.json` also routes `/` to the frontend).
   - **Build and Output Settings**: Leave defaults (no build command needed).
5. Click **Deploy**.
6. Within 15–30 seconds, Vercel will give you a live production URL:
   > 🌐 **Example Vercel URL**: `https://credence-loan.vercel.app`

---

## 💡 Alternative: No-Code Backend URL Switcher

Your frontend also has a built-in intelligent fetch interceptor!
If you ever want to point your Vercel frontend to a different backend without re-deploying, open your Vercel app in the browser, press `F12` (Developer Console), and run:

```javascript
localStorage.setItem('credence_backend_url', 'https://your-new-backend.onrender.com');
location.reload();
```

To reset back to default:
```javascript
localStorage.removeItem('credence_backend_url');
location.reload();
```

---

## ✅ Step 4: Verification Checklist

1. [ ] `/health` on Render returns `{"status": "healthy"}`.
2. [ ] Vercel URL loads the complete dark/light theme single-page portal.
3. [ ] Underwriting Form (`/predict`) calculates probability and renders the SHAP explanation chart.
4. [ ] Loan Simulator (`/simulate`) updates EMI and DTI sliders smoothly.
5. [ ] Repayment & Restructuring tabs load historical cycle data.
6. [ ] Voice Assistant accurately synthesizes voice audio via `/speak`.
