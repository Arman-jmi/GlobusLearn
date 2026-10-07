// GlobusLearn UI interactions: hero parallax and card tilt.
// Runs once per browser tab (Streamlit reruns re-send this element); uses delegated listeners so it keeps
// working as Streamlit replaces the DOM. Disabled for prefers-reduced-motion and touch-only devices.
(function () {
  if (window.__globusLearnUI) return;
  window.__globusLearnUI = true;

  // Streamlit keeps the scroll offset when the script swaps chat -> landing; start the landing page at the top.
  let lastView = null;
  setInterval(function () {
    const view = document.querySelector(".st-key-topbar") ? "chat" : document.querySelector(".gl-hero-copy") ? "home" : null;
    if (view === "home" && lastView === "chat") {
      const main = document.querySelector('[data-testid="stMain"]');
      if (main) main.scrollTo({ top: 0 });
    }
    if (view) lastView = view;
  }, 400);

  const motionOK = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const finePointer = window.matchMedia("(pointer: fine)").matches;
  if (!motionOK || !finePointer) return;

  const TILT_HOST = '[class*="st-key-tilt_"]';
  const MAX_CORE = 14;   // degrees the Science Core turns toward the cursor
  const MAX_TILT = 7;    // degrees a card tilts under the cursor

  let mx = 0, my = 0, frame = 0, tiltHost = null, tiltEvent = null;

  function clamp(v) { return Math.max(-1, Math.min(1, v)); }

  function update() {
    frame = 0;
    const core = document.querySelector(".gl-core");
    if (core) {
      const r = core.getBoundingClientRect();
      if (r.bottom > 0 && r.top < window.innerHeight) {
        const dx = clamp((mx - (r.left + r.width / 2)) / (window.innerWidth / 2));
        const dy = clamp((my - (r.top + r.height / 2)) / (window.innerHeight / 2));
        core.style.setProperty("--ry", (dx * MAX_CORE).toFixed(2) + "deg");
        core.style.setProperty("--rx", (-dy * MAX_CORE).toFixed(2) + "deg");
      }
    }
    if (tiltHost && tiltEvent) {
      const r = tiltHost.getBoundingClientRect();
      const px = (tiltEvent.clientX - r.left) / r.width - 0.5;
      const py = (tiltEvent.clientY - r.top) / r.height - 0.5;
      tiltHost.style.setProperty("--ty", (px * 2 * MAX_TILT).toFixed(2) + "deg");
      tiltHost.style.setProperty("--tx", (-py * 2 * MAX_TILT).toFixed(2) + "deg");
      tiltHost.style.setProperty("--gx", ((px + 0.5) * 100).toFixed(1) + "%");
      tiltHost.style.setProperty("--gy", ((py + 0.5) * 100).toFixed(1) + "%");
    }
  }

  function schedule() { if (!frame) frame = requestAnimationFrame(update); }

  function resetTilt(host) {
    if (!host) return;
    host.style.removeProperty("--tx");
    host.style.removeProperty("--ty");
  }

  document.addEventListener("pointermove", function (e) {
    mx = e.clientX; my = e.clientY;
    const host = e.target instanceof Element ? e.target.closest(TILT_HOST) : null;
    if (host !== tiltHost) { resetTilt(tiltHost); tiltHost = host; }
    tiltEvent = host ? e : null;
    schedule();
  }, { passive: true });

  document.addEventListener("pointerleave", function () { resetTilt(tiltHost); tiltHost = null; }, { passive: true });
})();
