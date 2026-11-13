/**
 * config.js — Runtime Backend URL Configuration
 * ==============================================
 * This file is loaded BEFORE script.js so that window.BACKEND_URL is
 * available when script.js reads it.
 *
 * ── LOCAL DEVELOPMENT ──────────────────────────────────────────────────────
 * Leave the value as-is. Uvicorn must be running on port 8000.
 *
 * ── RENDER DEPLOYMENT ──────────────────────────────────────────────────────
 * After your Render backend service is live, replace the URL below with the
 * actual service URL shown in the Render dashboard, e.g.:
 *   https://student-mental-health-api.onrender.com
 *
 * Then commit + push — the static site will redeploy automatically.
 */

// TODO: Replace this URL after your Render backend service is deployed.
window.BACKEND_URL = 'http://127.0.0.1:https://student-mental-health-project-2093.onrender.com';

