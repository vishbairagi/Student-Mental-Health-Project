/**
 * Student Social Media & Mental Health Prediction System — Frontend Script
 * Handles form validation, Fetch API integration with FastAPI, UI state transitions,
 * and live model health diagnostic polling.
 */

// ── Backend Configuration ────────────────────────────────────────────────────
// Priority order:
//   1. window.BACKEND_URL  — set by a <script> tag before this file (best for static hosts)
//   2. Same-origin /api    — if frontend is served from the same domain as the API
//   3. Localhost fallback  — for local development only
//
// For Render deployment, create a file  frontend/config.js  with the single line:
//   window.BACKEND_URL = 'https://student-mental-health-api.onrender.com';
// and include it BEFORE this script in index.html.
const BACKEND_URL = window.BACKEND_URL || 'http://127.0.0.1:8000';

// ── DOM Element Selectors ───────────────────────────────────────────────────
const predictionForm = document.getElementById('prediction-form');
const btnPredict = document.getElementById('btn-predict');
const btnSpinner = document.getElementById('btn-spinner');
const btnReset = document.getElementById('btn-reset');

// Status Pill
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');

// Result Views
const resultPlaceholder = document.getElementById('result-placeholder');
const resultLoading = document.getElementById('result-loading');
const resultContent = document.getElementById('result-content');

// Result Output Elements
const scoreNumber = document.getElementById('score-number');
const scoreCircle = document.getElementById('score-circle');
const meterFill = document.getElementById('meter-fill');
const riskBadge = document.getElementById('risk-badge');
const interpretationText = document.getElementById('interpretation-text');
const modelVal = document.getElementById('model-val');
const rawVal = document.getElementById('raw-val');
const usageVal = document.getElementById('usage-val');
const sleepVal = document.getElementById('sleep-val');
const predTimestamp = document.getElementById('pred-timestamp');

// Error Alert
const errorAlert = document.getElementById('error-alert');
const errorTitle = document.getElementById('error-title');
const errorMessage = document.getElementById('error-message');
const errorClose = document.getElementById('error-close');

// Presets
const btnPresetHealthy = document.getElementById('preset-healthy');
const btnPresetDistressed = document.getElementById('preset-distressed');

// ── Initial Health Check on Page Load ───────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  checkBackendHealth();
});

/**
 * Ping the FastAPI /health endpoint to verify server status.
 */
async function checkBackendHealth() {
  try {
    const response = await fetch(`${BACKEND_URL}/health`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' }
    });

    if (response.ok) {
      const data = await response.json();
      statusDot.className = 'status-dot online';
      statusText.textContent = `API Connected (${data.model_type || 'Model Ready'})`;
      statusText.style.color = '#10b981';
    } else {
      markBackendOffline(`Server responded with HTTP ${response.status}`);
    }
  } catch (err) {
    markBackendOffline('Backend offline. Run Uvicorn server.');
  }
}

function markBackendOffline(msg) {
  statusDot.className = 'status-dot offline';
  statusText.textContent = msg;
  statusText.style.color = '#ef4444';
}

// ── Form Submission Handler ─────────────────────────────────────────────────
predictionForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  hideError();

  // 1. Extract values with explicit type casting matching backend schemas
  const payload = {
    Age: parseInt(document.getElementById('Age').value, 10),
    Gender: document.getElementById('Gender').value.trim(),
    Country: document.getElementById('Country').value.trim(),
    Academic_Level: document.getElementById('Academic_Level').value.trim(),
    Most_Used_Platform: document.getElementById('Most_Used_Platform').value.trim(),
    Purpose_Of_Use: document.getElementById('Purpose_Of_Use').value.trim(),
    Avg_Daily_Usage_Hours: parseFloat(document.getElementById('Avg_Daily_Usage_Hours').value),
    Daily_Unlocks: parseInt(document.getElementById('Daily_Unlocks').value, 10),
    Study_Hours: parseFloat(document.getElementById('Study_Hours').value),
    Physical_Activity_Hours: parseFloat(document.getElementById('Physical_Activity_Hours').value),
    Sleep_Hours_Per_Night: parseFloat(document.getElementById('Sleep_Hours_Per_Night').value),
    Stress_Level: document.getElementById('Stress_Level').value.trim()
  };

  // 2. Client-side sanity checks
  const clientErrors = validateClientInput(payload);
  if (clientErrors.length > 0) {
    showError('Please check your input values', clientErrors.join('<br>'));
    return;
  }

  // 3. UI State: Transition to loading
  setLoadingState(true);

  try {
    // 4. Send asynchronous HTTP POST request via Fetch API
    const response = await fetch(`${BACKEND_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    const data = await response.json();

    if (!response.ok) {
      // Handle known HTTP error codes
      if (response.status === 422 && data.details) {
        // Pydantic validation error
        const issues = data.details.map(d => `<strong>${d.field}:</strong> ${d.message}`).join('<br>');
        showError('Validation Error (HTTP 422)', issues);
      } else if (response.status === 503) {
        showError('Service Unavailable (HTTP 503)', data.detail || 'The ML pipeline is not loaded on the backend.');
      } else {
        showError(`Server Error (HTTP ${response.status})`, data.detail || data.error || 'An unexpected error occurred during prediction.');
      }
      resetResultToPlaceholder();
      return;
    }

    // 5. Success: Render predictions on the UI
    renderPredictionResult(data, payload);
    // Refresh backend status pill to online
    statusDot.className = 'status-dot online';
    statusText.textContent = `API Connected (${data.model_name || 'Model Ready'})`;
    statusText.style.color = '#10b981';

  } catch (networkError) {
    // Connection refused / network failure
    showError(
      'Cannot Connect to Backend Server',
      `Failed to reach <code>${BACKEND_URL}/predict</code>.<br><br>` +
      `<strong>Troubleshooting:</strong><br>` +
      `1. Make sure Uvicorn is running: <code>uvicorn backend.main:app --host 127.0.0.1 --port 8000</code><br>` +
      `2. Verify that your browser permits local network requests.`
    );
    markBackendOffline('Connection Refused');
    resetResultToPlaceholder();
  } finally {
    setLoadingState(false);
  }
});

/**
 * Validate values before dispatching to backend.
 */
function validateClientInput(p) {
  const errors = [];
  if (isNaN(p.Age) || p.Age < 10 || p.Age > 100) errors.push('Age must be between 10 and 100 years.');
  if (!p.Gender) errors.push('Gender is required.');
  if (!p.Country) errors.push('Country is required.');
  if (isNaN(p.Avg_Daily_Usage_Hours) || p.Avg_Daily_Usage_Hours < 0 || p.Avg_Daily_Usage_Hours > 24) {
    errors.push('Daily social media usage must be between 0.0 and 24.0 hours.');
  }
  if (isNaN(p.Daily_Unlocks) || p.Daily_Unlocks < 0 || p.Daily_Unlocks > 1000) {
    errors.push('Daily unlocks must be between 0 and 1000.');
  }
  if (isNaN(p.Study_Hours) || p.Study_Hours < 0 || p.Study_Hours > 24) {
    errors.push('Daily study hours must be between 0.0 and 24.0 hours.');
  }
  if (isNaN(p.Physical_Activity_Hours) || p.Physical_Activity_Hours < 0 || p.Physical_Activity_Hours > 24) {
    errors.push('Physical activity hours must be non-negative (&ge; 0.0) and &le; 24.0.');
  }
  if (isNaN(p.Sleep_Hours_Per_Night) || p.Sleep_Hours_Per_Night < 0 || p.Sleep_Hours_Per_Night > 24) {
    errors.push('Sleep hours per night must be between 0.0 and 24.0 hours.');
  }
  return errors;
}

/**
 * Update UI with the predicted score, interpretation, risk badge, and styling.
 */
function renderPredictionResult(data, inputPayload) {
  const score = data.predicted_mental_health_score;

  // 1. Text elements
  scoreNumber.textContent = score.toFixed(2);
  interpretationText.textContent = data.interpretation;
  modelVal.textContent = data.model_name || 'RandomForestRegressor';
  rawVal.textContent = (data.raw_prediction !== undefined) ? data.raw_prediction.toFixed(4) : score.toFixed(4);
  usageVal.textContent = `${inputPayload.Avg_Daily_Usage_Hours} hrs/day (${inputPayload.Most_Used_Platform})`;
  sleepVal.textContent = `${inputPayload.Sleep_Hours_Per_Night} hrs/night`;

  // 2. Formatted timestamp
  const dateObj = data.timestamp ? new Date(data.timestamp) : new Date();
  predTimestamp.textContent = dateObj.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) +
    ' on ' + dateObj.toLocaleDateString();

  // 3. Score percentage fill for the meter bar
  const pct = Math.min(100, Math.max(0, (score / 10.0) * 100));
  meterFill.style.width = `${pct}%`;

  // 4. Color-code risk badge and score ring
  const risk = data.risk_level || 'Moderate Risk';
  riskBadge.textContent = risk;

  // Remove old classes
  riskBadge.className = 'risk-badge';
  scoreCircle.style.borderColor = '';
  meterFill.style.background = '';

  if (score >= 7.5) {
    riskBadge.classList.add('low');
    scoreCircle.style.borderColor = 'var(--risk-low)';
    meterFill.style.background = 'linear-gradient(90deg, #10b981, #059669)';
  } else if (score >= 6.0) {
    riskBadge.classList.add('moderate');
    scoreCircle.style.borderColor = 'var(--risk-moderate)';
    meterFill.style.background = 'linear-gradient(90deg, #0ea5e9, #0284c7)';
  } else if (score >= 4.5) {
    riskBadge.classList.add('elevated');
    scoreCircle.style.borderColor = 'var(--risk-elevated)';
    meterFill.style.background = 'linear-gradient(90deg, #f59e0b, #d97706)';
  } else {
    riskBadge.classList.add('high');
    scoreCircle.style.borderColor = 'var(--risk-high)';
    meterFill.style.background = 'linear-gradient(90deg, #ef4444, #dc2626)';
  }

  // 5. Display the result view
  resultPlaceholder.style.display = 'none';
  resultLoading.style.display = 'none';
  resultContent.style.display = 'block';

  // Smooth scroll to result on mobile screens
  if (window.innerWidth <= 1024) {
    resultContent.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

// ── Reset Form Handler ──────────────────────────────────────────────────────
btnReset.addEventListener('click', () => {
  predictionForm.reset();
  // Restore sensible defaults
  document.getElementById('Age').value = '21';
  document.getElementById('Gender').value = 'Female';
  document.getElementById('Country').value = 'USA';
  document.getElementById('Academic_Level').value = 'Undergraduate';
  document.getElementById('Most_Used_Platform').value = 'Instagram';
  document.getElementById('Purpose_Of_Use').value = 'Entertainment';
  document.getElementById('Avg_Daily_Usage_Hours').value = '4.5';
  document.getElementById('Daily_Unlocks').value = '150';
  document.getElementById('Study_Hours').value = '3.5';
  document.getElementById('Physical_Activity_Hours').value = '1.5';
  document.getElementById('Sleep_Hours_Per_Night').value = '7.0';
  document.getElementById('Stress_Level').value = 'Medium';

  hideError();
  resetResultToPlaceholder();
});

// ── Quick-Fill Preset Buttons ───────────────────────────────────────────────
btnPresetHealthy.addEventListener('click', () => {
  document.getElementById('Age').value = '21';
  document.getElementById('Gender').value = 'Male';
  document.getElementById('Country').value = 'USA';
  document.getElementById('Academic_Level').value = 'Undergraduate';
  document.getElementById('Most_Used_Platform').value = 'YouTube';
  document.getElementById('Purpose_Of_Use').value = 'Education';
  document.getElementById('Avg_Daily_Usage_Hours').value = '2.0';
  document.getElementById('Daily_Unlocks').value = '90';
  document.getElementById('Study_Hours').value = '4.5';
  document.getElementById('Physical_Activity_Hours').value = '2.0';
  document.getElementById('Sleep_Hours_Per_Night').value = '8.0';
  document.getElementById('Stress_Level').value = 'Low';
  hideError();
});

btnPresetDistressed.addEventListener('click', () => {
  document.getElementById('Age').value = '20';
  document.getElementById('Gender').value = 'Female';
  document.getElementById('Country').value = 'UK';
  document.getElementById('Academic_Level').value = 'Undergraduate';
  document.getElementById('Most_Used_Platform').value = 'TikTok';
  document.getElementById('Purpose_Of_Use').value = 'Entertainment';
  document.getElementById('Avg_Daily_Usage_Hours').value = '7.5';
  document.getElementById('Daily_Unlocks').value = '240';
  document.getElementById('Study_Hours').value = '1.5';
  document.getElementById('Physical_Activity_Hours').value = '0.5';
  document.getElementById('Sleep_Hours_Per_Night').value = '4.5';
  document.getElementById('Stress_Level').value = 'Very High';
  hideError();
});

// ── Helper: UI States & Error Handling ──────────────────────────────────────
function setLoadingState(isLoading) {
  btnPredict.disabled = isLoading;
  if (isLoading) {
    btnSpinner.classList.add('active');
    resultPlaceholder.style.display = 'none';
    resultContent.style.display = 'none';
    resultLoading.style.display = 'block';
  } else {
    btnSpinner.classList.remove('active');
    resultLoading.style.display = 'none';
  }
}

function resetResultToPlaceholder() {
  resultPlaceholder.style.display = 'block';
  resultLoading.style.display = 'none';
  resultContent.style.display = 'none';
}

function showError(title, msg) {
  errorTitle.textContent = title;
  errorMessage.innerHTML = msg;
  errorAlert.style.display = 'flex';
  errorAlert.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideError() {
  errorAlert.style.display = 'none';
}

errorClose.addEventListener('click', hideError);

