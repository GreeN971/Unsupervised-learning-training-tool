"""Shared neumorphic theme: design tokens, Quasar overrides, and the ambient
background canvas, injected once for every page via NiceGUI's shared
head/body HTML hooks. Pages call `install()` once at startup and `stepper()`
/ `panel()` to lay out their own content consistently.
"""

from contextlib import contextmanager
from typing import Iterator

from nicegui import ui

STEPS = ("Data", "Algorithm", "Configure", "Topology", "Save location", "Run")

_FONTS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
"""

_THEME_INIT = """
<script>
(function(){
  try {
    var t = localStorage.getItem('ng-theme');
    if(t === 'light' || t === 'dark'){
      document.documentElement.setAttribute('data-theme', t);
    }
  } catch(e) {}
})();
</script>
"""

_STYLE = """
<style>
:root{
  --ng-bg:#131722;
  --ng-surface:#1a2033;
  --ng-surface-raised:#1d2438;
  --ng-shadow-light:#262f4a;
  --ng-shadow-dark:#0a0d16;
  --ng-text-primary:#e9edf9;
  --ng-text-secondary:#8e98b8;
  --ng-text-muted:#5c6584;
  --ng-accent:#3fe0e8;
  --ng-accent-soft:#3fe0e81f;
  --ng-accent-2:#9c8dff;
  --ng-danger:#ff7a8a;
  --ng-warning:#f5c26b;
  --ng-success:#4fe0a8;
  --ng-border-glow:#3fe0e855;
  --ng-font-display:'Chakra Petch',ui-sans-serif,sans-serif;
  --ng-font-body:'IBM Plex Sans',ui-sans-serif,system-ui,sans-serif;
  --ng-font-mono:'IBM Plex Mono',ui-monospace,monospace;
}

:root[data-theme="light"]{
  --ng-bg:#e6eaf2;
  --ng-surface:#eaeef5;
  --ng-surface-raised:#eef1f7;
  --ng-shadow-light:#ffffff;
  --ng-shadow-dark:#b7c0d6;
  --ng-text-primary:#1c2438;
  --ng-text-secondary:#5a6483;
  --ng-text-muted:#8a92ab;
  --ng-accent:#0c8fa3;
  --ng-accent-soft:#0c8fa31f;
  --ng-accent-2:#5b4fd6;
  --ng-danger:#c8404f;
  --ng-warning:#8f5400;
  --ng-success:#188a63;
  --ng-border-glow:#0c8fa355;
}

html, body{
  background:var(--ng-bg) !important;
  color:var(--ng-text-primary);
  font-family:var(--ng-font-body);
}

/* Each of the 5 wizard steps is a full page load (not an SPA route), so this
   is what makes arriving at one feel like a transition rather than a hard
   cut - the background canvas state is restored separately in JS so the
   dot/line pattern itself carries over rather than jumping too. */
body{animation:ng-fade-in .35s ease both;}
@keyframes ng-fade-in{from{opacity:0;}to{opacity:1;}}
@media (prefers-reduced-motion: reduce){body{animation:none;}}

/* Opt-in to the browser's native cross-document view transition where
   supported (a soft crossfade between the old and new page); a no-op,
   harmless at-rule on engines that don't understand it yet. */
@view-transition{navigation:auto;}

/* Transparent, not var(--ng-bg): these are normal-flow boxes sitting between
   body and the canvas, so an opaque background here would paint over it. */
#app, .nicegui-content, .q-page, .q-layout, .q-page-container{
  background:transparent !important;
  color:var(--ng-text-primary);
  font-family:var(--ng-font-body);
}

/* #bg-canvas sits behind the UI using plain positive-z-index stacking, not
   z-index:-1. A negative z-index relies on the canvas being painted in the
   "before normal-flow content" stacking step, which some Chromium builds
   (confirmed here) simply never composite to screen - the element draws
   into its own buffer correctly but nothing ever reaches the display. Every
   opaque surface (.neu-panel, .stepper, .neu-theme-toggle) is promoted to
   its own stacking position above the canvas instead, which is the more
   universally reliable way to layer a background under real UI. */
#bg-canvas{position:fixed;inset:0;width:100vw;height:100vh;z-index:0;pointer-events:none;display:block;}

.text-h5,.text-h4,.text-h6{
  font-family:var(--ng-font-display) !important;
  font-weight:600 !important;
  letter-spacing:.01em;
  color:var(--ng-text-primary) !important;
}

.neu-panel{
  position:relative;z-index:1;
  background:var(--ng-surface);
  border-radius:18px;
  box-shadow:10px 10px 24px var(--ng-shadow-dark), -10px -10px 24px var(--ng-shadow-light);
  padding:2.5rem !important;
}

/* NiceGUI's .nicegui-content wrapper carries its own 1rem top+bottom padding
   (2rem total). A plain 100vh min-height nested inside that padding is
   therefore always 2rem taller than the viewport - producing a scrollbar
   even on pages whose actual content would otherwise fit. This is used
   instead of Tailwind's min-h-screen for the page-centering column. */
.neu-viewport{min-height:calc(100vh - 2rem);}

.stepper{
  position:relative;z-index:1;
  display:flex;align-items:flex-start;gap:0;
  background:var(--ng-surface);
  border-radius:16px;
  padding:1.1rem 1.4rem;
  box-shadow:7px 7px 16px var(--ng-shadow-dark), -7px -7px 16px var(--ng-shadow-light);
  font-family:var(--ng-font-body);
}
/* Fixed width (not auto) so every step's node sits at the same offset
   regardless of label length ("Save location" vs "Run") - otherwise the
   connecting lines, which run edge-to-edge between step boxes, attach at a
   different distance from each node and the whole row reads as misaligned. */
.stepper .step{display:flex;flex-direction:column;align-items:center;gap:.4rem;flex:0 0 84px;width:84px;}
.stepper .label{text-align:center;}
.stepper .node{
  width:26px;height:26px;border-radius:9px;
  display:flex;align-items:center;justify-content:center;
  font-family:var(--ng-font-mono);font-size:12px;font-weight:500;
  background:var(--ng-surface);color:var(--ng-text-muted);
  box-shadow:4px 4px 8px var(--ng-shadow-dark), -4px -4px 8px var(--ng-shadow-light);
}
.stepper .step.done .node{color:var(--ng-accent);box-shadow:4px 4px 8px var(--ng-shadow-dark), -4px -4px 8px var(--ng-shadow-light), 0 0 10px var(--ng-accent-soft);}
.stepper .step.active .node{
  background:var(--ng-accent);color:var(--ng-bg);
  box-shadow:0 0 0 4px var(--ng-accent-soft), 0 0 16px var(--ng-border-glow);
  animation:ng-pulse 2.2s ease-in-out infinite;
}
.stepper .label{font-size:11.5px;color:var(--ng-text-muted);white-space:nowrap;}
.stepper .step.active .label,.stepper .step.done .label{color:var(--ng-text-primary);}
/* margin-top puts the line's *center* (not its top edge) at 13px, the node's
   own vertical center (half of the node's 26px height). */
.stepper .line{flex:1 1 auto;height:4px;background:var(--ng-shadow-dark);margin-top:11px;border-radius:2px;min-width:16px;}
.stepper .line.done{background:var(--ng-accent);box-shadow:0 0 8px var(--ng-border-glow);}
@keyframes ng-pulse{0%,100%{box-shadow:0 0 0 4px var(--ng-accent-soft),0 0 16px var(--ng-border-glow);}50%{box-shadow:0 0 0 7px var(--ng-accent-soft),0 0 22px var(--ng-border-glow);}}
@media (prefers-reduced-motion: reduce){.stepper .step.active .node{animation:none;}}

.q-btn{
  font-family:var(--ng-font-body) !important;
  font-weight:500 !important;
  text-transform:none !important;
  border-radius:12px !important;
  background:var(--ng-surface) !important;
  color:var(--ng-text-primary) !important;
  box-shadow:6px 6px 13px var(--ng-shadow-dark), -6px -6px 13px var(--ng-shadow-light);
  transition:box-shadow .15s ease, transform .15s ease;
}
.q-btn:before{box-shadow:none !important;}
.q-btn:hover{transform:translateY(-1px);}
.q-btn:active,.q-btn.q-btn--active{
  box-shadow:inset 3px 3px 7px var(--ng-shadow-dark), inset -3px -3px 7px var(--ng-shadow-light) !important;
  transform:translateY(0);
}
.q-btn.neu-primary{color:var(--ng-accent) !important;box-shadow:6px 6px 13px var(--ng-shadow-dark), -6px -6px 13px var(--ng-shadow-light), 0 0 0 1px var(--ng-border-glow);}
.q-btn.neu-primary:hover{box-shadow:6px 6px 13px var(--ng-shadow-dark), -6px -6px 13px var(--ng-shadow-light), 0 0 18px var(--ng-border-glow);}
.q-btn.neu-ghost{background:transparent !important;box-shadow:none !important;color:var(--ng-text-secondary) !important;}
.q-btn.neu-ghost:hover{color:var(--ng-text-primary) !important;transform:none;}
.q-btn.neu-danger{color:var(--ng-danger) !important;}

.q-field .q-field__control{
  background:var(--ng-surface) !important;
  border-radius:12px !important;
  box-shadow:inset 4px 4px 9px var(--ng-shadow-dark), inset -4px -4px 9px var(--ng-shadow-light);
  padding:0 14px !important;
  min-height:46px !important;
}
.q-field__control:before,.q-field__control:after{border:none !important;}
.q-field--focused .q-field__control{
  box-shadow:inset 4px 4px 9px var(--ng-shadow-dark), inset -4px -4px 9px var(--ng-shadow-light), 0 0 0 1px var(--ng-border-glow), 0 0 14px var(--ng-border-glow) !important;
}
.q-field__label{
  font-family:var(--ng-font-mono) !important;
  font-size:11px !important;
  letter-spacing:.09em;
  text-transform:uppercase;
  color:var(--ng-text-muted) !important;
}
.q-field--focused .q-field__label,.q-field--float .q-field__label{color:var(--ng-text-secondary) !important;}
.q-field__native,.q-field__input{
  font-family:var(--ng-font-mono) !important;
  color:var(--ng-text-primary) !important;
  font-size:14.5px !important;
}
.q-field__marginal{color:var(--ng-text-muted) !important;}
/* The native OS/browser spin-button widget can't be reliably reskinned (it
   renders completely differently in the browser vs. the desktop webview,
   and neither honors a filter-based tint) - replaced with real, themed
   up/down controls (.neu-spin) overlaid on the field instead. */
input[type=number]::-webkit-inner-spin-button,input[type=number]::-webkit-outer-spin-button{
  -webkit-appearance:none;margin:0;
}
input[type=number]{-moz-appearance:textfield;}
.neu-number .q-field__control{padding-right:34px !important;}

.neu-spin{
  position:absolute;top:50%;right:8px;transform:translateY(-50%);
  display:flex;flex-direction:column;
  border-radius:6px;overflow:hidden;
  box-shadow:2px 2px 5px var(--ng-shadow-dark), -2px -2px 5px var(--ng-shadow-light);
}
.neu-spin .q-icon{
  width:20px;height:13px;
  display:flex;align-items:center;justify-content:center;
  background:var(--ng-surface);color:var(--ng-text-muted);
  font-size:13px;line-height:1;cursor:pointer;
  transition:color .15s ease, background .15s ease;
}
.neu-spin .q-icon:hover{color:var(--ng-accent);background:var(--ng-surface-raised);}
.neu-spin .q-icon:active{background:var(--ng-shadow-dark);}

.q-menu{
  background:var(--ng-surface-raised) !important;
  box-shadow:8px 8px 18px var(--ng-shadow-dark), -8px -8px 18px var(--ng-shadow-light) !important;
  border-radius:12px !important;
}
.q-menu .q-item{color:var(--ng-text-primary) !important;font-family:var(--ng-font-mono) !important;font-size:14px;}
.q-menu .q-item.q-manual-focusable--focused,.q-menu .q-item:hover{background:var(--ng-accent-soft) !important;}

.q-toggle__track{background:var(--ng-shadow-dark) !important;opacity:1 !important;}
.q-toggle__thumb{background:var(--ng-surface) !important;box-shadow:2px 2px 5px var(--ng-shadow-dark), -2px -2px 5px var(--ng-shadow-light);}
.q-toggle__thumb:after{background:transparent !important;}
.q-toggle__inner--truthy .q-toggle__track{background:var(--ng-accent) !important;opacity:.35 !important;box-shadow:0 0 10px var(--ng-accent-soft);}
.q-toggle__inner--truthy .q-toggle__thumb{background:var(--ng-accent) !important;box-shadow:0 0 8px var(--ng-accent) !important;}
.q-toggle__label{font-family:var(--ng-font-body) !important;color:var(--ng-text-primary) !important;}

.q-checkbox__bg{border-color:var(--ng-shadow-dark) !important;border-radius:4px !important;}
.q-checkbox__inner--truthy .q-checkbox__bg{background:var(--ng-accent) !important;border-color:var(--ng-accent) !important;box-shadow:0 0 8px var(--ng-accent-soft);}
.q-checkbox__label{font-family:var(--ng-font-body) !important;color:var(--ng-text-primary) !important;}

.q-linear-progress{
  border-radius:7px !important;overflow:hidden;
  background:var(--ng-surface) !important;
  box-shadow:inset 3px 3px 7px var(--ng-shadow-dark), inset -3px -3px 7px var(--ng-shadow-light);
  height:14px !important;
}
.q-linear-progress__track{background:transparent !important;opacity:1 !important;}
.q-linear-progress__model{
  background:linear-gradient(90deg,var(--ng-accent),var(--ng-accent-2)) !important;
  box-shadow:0 0 12px var(--ng-border-glow);
}

.q-tooltip{
  font-family:var(--ng-font-body) !important;font-size:11.5px !important;line-height:1.5;
  background:var(--ng-surface-raised) !important;color:var(--ng-text-secondary) !important;
  border-radius:8px !important;
  box-shadow:5px 5px 12px var(--ng-shadow-dark), -5px -5px 12px var(--ng-shadow-light);
  padding:.5rem .7rem !important;
  max-width:220px;
}

.neu-info{
  width:18px;height:18px;border-radius:50%;
  display:inline-flex;align-items:center;justify-content:center;
  background:var(--ng-surface) !important;color:var(--ng-text-muted) !important;
  box-shadow:2px 2px 5px var(--ng-shadow-dark), -2px -2px 5px var(--ng-shadow-light);
  font-size:12px !important;cursor:help;
}
.neu-info:hover{color:var(--ng-accent) !important;}

.neu-readout{
  font-family:var(--ng-font-mono) !important;font-size:38px;font-weight:500;
  color:var(--ng-accent) !important;letter-spacing:.02em;
  text-shadow:0 0 18px var(--ng-border-glow);
}
.neu-status{font-family:var(--ng-font-mono) !important;font-size:13px;color:var(--ng-text-muted) !important;}
.neu-status.neu-ok{color:var(--ng-success) !important;}
.neu-status.neu-error{color:var(--ng-danger) !important;}

/* What a finished run shows: the saved path and the model's size. The path
   wraps anywhere, so all of it can be read. */
.neu-result{
  width:100%;background:var(--ng-surface);border-radius:14px;
  border-left:4px solid var(--ng-success);
  padding:.9rem 1.1rem;
  box-shadow:inset 4px 4px 9px var(--ng-shadow-dark), inset -4px -4px 9px var(--ng-shadow-light);
}
.neu-result-value{
  font-family:var(--ng-font-mono) !important;font-size:22px;font-weight:600;line-height:1.4;
  color:var(--ng-success) !important;overflow-wrap:anywhere;user-select:text;
}

.neu-path{
  width:100%;
  font-family:var(--ng-font-mono) !important;font-size:12.5px;
  color:var(--ng-text-secondary);
  background:var(--ng-surface);
  border-radius:12px;
  padding:.85rem 1rem;
  box-shadow:inset 4px 4px 9px var(--ng-shadow-dark), inset -4px -4px 9px var(--ng-shadow-light);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
}
.neu-path.neu-set{color:var(--ng-accent);}

/* Algorithm notices (app/pages/notices.py): blocking problems, warnings and
   the algorithm's standing limitations, as inset rows with a colored icon
   and left edge. */
.neu-notices{width:100%;display:flex;flex-direction:column;gap:.5rem;}
.neu-notices-title{
  font-family:var(--ng-font-display) !important;font-size:12px;font-weight:600;
  letter-spacing:.08em;text-transform:uppercase;color:var(--ng-text-muted) !important;
  margin-top:.25rem;
}
/* Configure page: titles over the PyOD detector and Keras surrogate groups,
   with an etched rule underneath. */
.neu-section-title{
  width:100%;
  font-family:var(--ng-font-display) !important;font-size:12px;font-weight:600;
  letter-spacing:.08em;text-transform:uppercase;color:var(--ng-text-muted) !important;
  padding-bottom:.35rem;margin-bottom:.25rem;
  border-bottom:1px solid var(--ng-shadow-dark);
  box-shadow:0 1px 0 var(--ng-shadow-light);
}
.neu-notice{
  display:flex;align-items:flex-start;gap:.6rem;width:100%;
  font-size:13px;line-height:1.45;color:var(--ng-text-secondary) !important;
  background:var(--ng-surface);border-radius:12px;
  border-left:3px solid var(--ng-text-muted);
  padding:.6rem .85rem;
  box-shadow:inset 3px 3px 7px var(--ng-shadow-dark), inset -3px -3px 7px var(--ng-shadow-light);
}
.neu-notice .q-icon{flex:none;font-size:18px;margin-top:1px;color:var(--ng-text-muted);}
.neu-notice.neu-notice-block{border-left-color:var(--ng-danger);color:var(--ng-text-primary) !important;}
.neu-notice.neu-notice-block .q-icon{color:var(--ng-danger);}
.neu-notice.neu-notice-warn{border-left-color:var(--ng-warning);}
.neu-notice.neu-notice-warn .q-icon{color:var(--ng-warning);}
.neu-notice.neu-notice-info .q-icon{color:var(--ng-accent);}

/* CSV wizard sub-tabs and its column include list. */
.neu-tabs .q-tab{color:var(--ng-text-secondary) !important;border-radius:10px;}
.neu-tabs .q-tab--active{color:var(--ng-accent) !important;}
.neu-tabs .q-tab__indicator{background:var(--ng-accent) !important;}
.neu-tab-panels,.neu-tab-panels .q-tab-panel{background:transparent !important;padding:0 !important;}
.neu-columns{
  width:100%;max-height:320px;overflow-y:auto;
  border-radius:12px;padding:.5rem 1rem;
  box-shadow:inset 4px 4px 9px var(--ng-shadow-dark), inset -4px -4px 9px var(--ng-shadow-light);
}
.neu-column-name{font-family:var(--ng-font-mono) !important;color:var(--ng-text-primary);}

.neu-theme-toggle{
  position:fixed;top:20px;right:20px;z-index:1000;
  width:44px;height:44px;border-radius:12px;border:none;cursor:pointer;
  background:var(--ng-surface);
  box-shadow:6px 6px 13px var(--ng-shadow-dark), -6px -6px 13px var(--ng-shadow-light);
  display:flex;align-items:center;justify-content:center;
  color:var(--ng-text-secondary);
  transition:box-shadow .15s ease, transform .15s ease, color .15s ease;
}
.neu-theme-toggle:hover{transform:translateY(-1px);color:var(--ng-accent);}
.neu-theme-toggle:active{
  box-shadow:inset 3px 3px 7px var(--ng-shadow-dark), inset -3px -3px 7px var(--ng-shadow-light);
  transform:translateY(0);
}
.neu-theme-toggle svg{width:20px;height:20px;}
.neu-theme-toggle .ng-icon-sun{display:none;}
.neu-theme-toggle .ng-icon-moon{display:block;}
:root[data-theme="light"] .neu-theme-toggle .ng-icon-sun{display:block;}
:root[data-theme="light"] .neu-theme-toggle .ng-icon-moon{display:none;}
</style>
"""

_TOGGLE_BUTTON = """
<button id="ng-theme-toggle" class="neu-theme-toggle" aria-label="Toggle light and dark theme" onclick="window.__ngToggleTheme()">
  <svg class="ng-icon-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true">
    <circle cx="12" cy="12" r="4.2"/>
    <path d="M12 2.5v2.4M12 19.1v2.4M4.2 4.2l1.7 1.7M18.1 18.1l1.7 1.7M2.5 12h2.4M19.1 12h2.4M4.2 19.8l1.7-1.7M18.1 5.9l1.7-1.7"/>
  </svg>
  <svg class="ng-icon-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true">
    <path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a6.8 6.8 0 0 0 10.5 10.5Z"/>
  </svg>
</button>
"""

_CANVAS = '<canvas id="bg-canvas" aria-hidden="true"></canvas>'

_SCRIPT = """
<script>
window.__ngToggleTheme = function(){
  var root = document.documentElement;
  var current = root.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
  var next = current === 'light' ? 'dark' : 'light';
  root.setAttribute('data-theme', next);
  try { localStorage.setItem('ng-theme', next); } catch(e) {}
};

(function(){
  // Belt-and-braces: some environments (embedded webviews in particular)
  // asynchronously re-apply an opaque background to these wrapper elements
  // after our stylesheet has already been parsed, which paints over
  // #bg-canvas even though it's transparent at first paint. Keep re-forcing
  // transparency on them for as long as the page lives, using inline
  // !important so no later stylesheet or inline-style write can win.
  function forceTransparent(el){
    el.style.setProperty('background', 'transparent', 'important');
    el.style.setProperty('background-color', 'transparent', 'important');
  }
  var TRANSPARENT_SELECTORS = ['#app', '.q-layout', '.q-page-container', '.q-page', '.nicegui-content'];
  function enforceTransparency(){
    TRANSPARENT_SELECTORS.forEach(function(sel){
      document.querySelectorAll(sel).forEach(forceTransparent);
    });
  }
  enforceTransparency();
  [0, 50, 150, 300, 600, 1200, 2500].forEach(function(delay){
    setTimeout(enforceTransparency, delay);
  });
  // childList-only: watching 'style'/'class' attributes here would re-fire
  // on our own forceTransparent() writes, creating a self-triggering loop
  // that floods the microtask queue and starves requestAnimationFrame -
  // exactly the "animates for a moment then freezes" bug this caused once.
  // Element insertions (e.g. Quasar mounting .q-page late) don't trigger
  // forceTransparent(), so this stays a one-way reaction with no feedback.
  if (window.MutationObserver) {
    new MutationObserver(enforceTransparency).observe(document.documentElement, {
      childList: true, subtree: true,
    });
  }

  function boot(){
    var canvas = document.getElementById('bg-canvas');
    if(!canvas || canvas.dataset.wired) return;
    canvas.dataset.wired = '1';
    var ctx = canvas.getContext('2d');
    var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    var w, h, nodes, pulses = [], pulseTimer = 0;

    function hexToRgb(hex){
      hex = hex.trim().replace('#','');
      if(hex.length===3) hex = hex.split('').map(function(c){return c+c;}).join('');
      var num = parseInt(hex,16);
      return ((num>>16)&255)+','+((num>>8)&255)+','+(num&255);
    }
    function currentRgb(){
      var v = getComputedStyle(document.documentElement).getPropertyValue('--ng-accent') || '#3fe0e8';
      return hexToRgb(v);
    }
    // Each wizard step is a full page load, so without this the dot/line
    // pattern would jump to a brand new random layout on every navigation -
    // jarring, since the network reads as one continuous thing. Persisting
    // positions (and velocities, so drift keeps its direction) across the
    // sessionStorage boundary makes it look like the same background simply
    // continued underneath the new page.
    var STORE_KEY = 'ng-bg-nodes';
    function loadSavedNodes(){
      try {
        var raw = sessionStorage.getItem(STORE_KEY);
        var arr = raw ? JSON.parse(raw) : null;
        return Array.isArray(arr) && arr.length ? arr : null;
      } catch(e) { return null; }
    }
    function saveNodes(){
      if(!nodes) return;
      try { sessionStorage.setItem(STORE_KEY, JSON.stringify(nodes)); } catch(e) {}
    }
    window.addEventListener('pagehide', saveNodes);
    window.addEventListener('beforeunload', saveNodes);
    setInterval(saveNodes, 2000);

    function freshNodes(){
      var count = Math.max(30, Math.min(60, Math.floor((w*h)/18000)));
      var result = [];
      for(var i=0;i<count;i++){
        result.push({x:Math.random()*w, y:Math.random()*h, vx:(Math.random()-0.5)*0.14, vy:(Math.random()-0.5)*0.14});
      }
      return result;
    }
    var initialized = false, lastW = 0, lastH = 0;
    function resize(){
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = window.innerWidth; h = window.innerHeight;
      canvas.width = w*dpr; canvas.height = h*dpr;
      canvas.style.width = w+'px'; canvas.style.height = h+'px';
      ctx.setTransform(dpr,0,0,dpr,0,0);
      var sizeChanged = Math.abs(w-lastW) > 4 || Math.abs(h-lastH) > 4;
      if(!initialized){
        initialized = true;
        nodes = loadSavedNodes() || freshNodes();
        lastW = w; lastH = h;
      } else if(sizeChanged){
        // A real window resize (rare for this app) - the old layout no
        // longer matches, so start a new one rather than stretching it.
        nodes = freshNodes();
        lastW = w; lastH = h;
      }
    }
    window.addEventListener('resize', resize);
    resize();
    // Some webviews report a 0x0 or stale window size on the very first
    // layout pass, before the native window has finished sizing itself.
    setTimeout(resize, 300);

    function linkDistance(){ return Math.min(170, w/5); }

    function maybeSpawnPulse(){
      pulseTimer++;
      if(pulseTimer < 90 || pulses.length >= 2) return;
      var maxD = linkDistance();
      var candidates = [];
      for(var i=0;i<nodes.length;i++){
        for(var j=i+1;j<nodes.length;j++){
          var dx=nodes[i].x-nodes[j].x, dy=nodes[i].y-nodes[j].y;
          if(Math.sqrt(dx*dx+dy*dy) < maxD) candidates.push([i,j]);
        }
      }
      if(candidates.length){
        var pick = candidates[Math.floor(Math.random()*candidates.length)];
        pulses.push({a:pick[0], b:pick[1], t:0});
        pulseTimer = 0;
      }
    }

    function draw(){
      ctx.clearRect(0,0,w,h);
      var rgb = currentRgb();
      var maxD = linkDistance();
      for(var i=0;i<nodes.length;i++){
        for(var j=i+1;j<nodes.length;j++){
          var dx=nodes[i].x-nodes[j].x, dy=nodes[i].y-nodes[j].y;
          var dist=Math.sqrt(dx*dx+dy*dy);
          if(dist<maxD){
            ctx.strokeStyle = 'rgba('+rgb+','+((1-dist/maxD)*0.28)+')';
            ctx.lineWidth = 0.8;
            ctx.beginPath();
            ctx.moveTo(nodes[i].x,nodes[i].y);
            ctx.lineTo(nodes[j].x,nodes[j].y);
            ctx.stroke();
          }
        }
      }
      nodes.forEach(function(n){
        ctx.beginPath();
        ctx.arc(n.x,n.y,2.4,0,Math.PI*2);
        ctx.fillStyle = 'rgba('+rgb+',0.6)';
        ctx.fill();
      });
      pulses.forEach(function(p){
        var na = nodes[p.a], nb = nodes[p.b];
        if(!na || !nb) return;
        var x = na.x + (nb.x-na.x)*p.t;
        var y = na.y + (nb.y-na.y)*p.t;
        var fade = Math.sin(Math.PI*p.t);
        ctx.beginPath();
        ctx.arc(x,y,3.2,0,Math.PI*2);
        ctx.fillStyle = 'rgba('+rgb+','+(0.9*fade)+')';
        ctx.fill();
      });
    }

    function step(){
      nodes.forEach(function(n){
        n.x += n.vx; n.y += n.vy;
        if(n.x < 0 || n.x > w) n.vx *= -1;
        if(n.y < 0 || n.y > h) n.vy *= -1;
      });
      pulses.forEach(function(p){ p.t += 0.012; });
      pulses = pulses.filter(function(p){ return p.t <= 1; });
      maybeSpawnPulse();
      draw();
      requestAnimationFrame(step);
    }

    if(reduceMotion){ draw(); } else { step(); }
  }
  if(document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
  document.addEventListener('DOMContentLoaded', boot);
  setTimeout(boot, 0);
})();
</script>
"""


def install() -> None:
    """Register the theme's fonts/CSS/background canvas on every page.

    Call once, before `ui.run()`.
    """
    ui.add_head_html(_THEME_INIT + _FONTS + _STYLE, shared=True)
    ui.add_body_html(_TOGGLE_BUTTON + _CANVAS + _SCRIPT, shared=True)


def _stepper_html(active: int) -> str:
    parts = []
    for i, name in enumerate(STEPS):
        state = "done" if i < active else "active" if i == active else ""
        node = "&#10003;" if i < active else str(i + 1)
        parts.append(
            f'<div class="step {state}"><div class="node">{node}</div>'
            f'<span class="label">{name}</span></div>'
        )
        if i < len(STEPS) - 1:
            parts.append(f'<div class="line {"done" if i < active else ""}"></div>')
    return f'<div class="stepper">{"".join(parts)}</div>'


@contextmanager
def panel(step: int | None, wide: bool = False) -> Iterator[ui.column]:
    """A centered neumorphic panel, with the wizard stepper above it, that
    page content is placed inside. `step` is the 0-based active step index,
    or None for a page outside the steps (the CSV wizard) - no stepper;
    `wide` makes room for fields beside the centered ones.
    """
    with ui.column().classes("w-full neu-viewport items-center justify-center gap-6 py-10"):
        # max-w-3xl: 6 fixed-width (84px) steps plus padding and minimum
        # line widths no longer fit in max-w-xl/2xl.
        if step is not None:
            ui.html(_stepper_html(step)).classes("w-full max-w-3xl")
        width = "max-w-2xl" if wide else "max-w-xl"
        with ui.column().classes(f"neu-panel items-center gap-4 w-full {width}") as col:
            yield col
