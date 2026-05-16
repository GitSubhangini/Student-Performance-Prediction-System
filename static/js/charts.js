/* charts.js — Fetch Plotly JSON from Flask API and render charts */

const PLOTLY_CONFIG = { responsive: true, displayModeBar: false };

function loadChart(url, elementId) {
  const el = document.getElementById(elementId);
  if (!el) return;

  // Loading skeleton
  el.innerHTML = '<div class="d-flex align-items-center justify-content-center h-100">' +
    '<div class="spinner-border text-accent" role="status"></div></div>';

  fetch(url)
    .then(r => r.json())
    .then(fig => {
      el.innerHTML = "";
      Plotly.newPlot(elementId, fig.data || [], fig.layout || {}, PLOTLY_CONFIG);
    })
    .catch(() => {
      el.innerHTML = '<div class="text-center text-muted py-4"><i class="bi bi-exclamation-circle me-2"></i>Chart unavailable</div>';
    });
}
