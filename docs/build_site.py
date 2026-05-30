"""
Build a static GitHub Pages site from the executed demo notebook.

Extracts every plot (image/png) and table (styled HTML + rich console output)
from ``PyWaveletPanel_Demo.ipynb`` and renders them, in document order, inside a
clean light-themed page with author info, links, and a short theory section.

Run from the repo root:  python docs/build_site.py
Output:  docs/index.html  and  docs/assets/*.png
"""
import base64
import json
import re
from pathlib import Path

import markdown as md

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "PyWaveletPanel_Demo.ipynb"
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def join(x):
    return "".join(x) if isinstance(x, list) else x


# --------------------------------------------------------------------------- #
#  Author / project metadata
# --------------------------------------------------------------------------- #
AUTHOR = "Dr. Merwan Roudane"
EMAIL = "merwanroudane920@gmail.com"
GITHUB = "https://github.com/merwanroudane/pywaveletpanel"
PYPI = "https://pypi.org/project/pywaveletpanel/"
ISSUES = "https://github.com/merwanroudane/pywaveletpanel/issues"
GH_USER = "https://github.com/merwanroudane"

# --------------------------------------------------------------------------- #
#  Theory section (hand-written, concise)
# --------------------------------------------------------------------------- #
THEORY_HTML = r"""
<h4>A. The discrete wavelet transform &amp; multiresolution analysis</h4>
<p>A wavelet transform decomposes a time series $x_t$ into components localised in
<strong>both time and frequency (scale)</strong>. Using a father wavelet $\phi$ (smooth,
low-pass) and a mother wavelet $\psi$ (oscillatory, high-pass), the series is written as</p>
<p>$$x_t = \underbrace{S_{J,t}}_{\text{smooth trend}} + \sum_{j=1}^{J}\underbrace{D_{j,t}}_{\text{detail at scale } j}.$$</p>
<p>Each <strong>detail</strong> band $D_j$ captures fluctuations with period roughly
$2^{j}$&ndash;$2^{j+1}$ time units; the <strong>smooth</strong> $S_J$ holds the long-run
trend. The library uses the <strong>maximal-overlap DWT (MODWT)</strong>, which is
shift-invariant and works for any sample length $T$ (unlike the classical DWT, which
requires $T = 2^{J}$).</p>

<h4>B. Scale-by-scale panel regression</h4>
<p>Classical panel regression estimates a <em>single</em> slope. Real economic
relationships, however, can differ across horizons: short-run noise vs. long-run
co-movement. We therefore run the fixed-effects regression <strong>separately at each
wavelet scale</strong>,</p>
<p>$$\tilde y_{it}^{(j)} = \alpha_i + \beta_j\,\tilde x_{it}^{(j)} + \varepsilon_{it}^{(j)},$$</p>
<p>where $\tilde y^{(j)},\tilde x^{(j)}$ are the scale-$j$ detail series and $\beta_j$ is
the <strong>scale-specific elasticity</strong>. Standard errors use the Newey&ndash;West
HAC estimator to remain valid under serial correlation introduced by the transform.</p>

<h4>C. SAW structural-break detection</h4>
<p>The <strong>Structure-Adapted Wavelet (SAW)</strong> estimator of Bada et al. (2021)
detects <em>jumps</em> in a time-varying coefficient $\gamma_t$. After first-differencing
out the fixed effects, the coefficient path is expanded in a <strong>Haar basis</strong>;
small detail coefficients (noise) are removed by hard-thresholding at the universal level</p>
<p>$$\lambda = \hat\sigma\sqrt{2\log T}, \qquad \hat\sigma = \operatorname{median}(|W_1|)/0.6745,$$</p>
<p>and breaks are read off the <strong>reconstructed</strong> path wherever it jumps by
more than $\lambda$. A Post-SAW step re-estimates each stable segment and runs Chow tests
across consecutive intervals.</p>

<h4>D. Wavelet panel unit-root tests</h4>
<p>To ask whether series are <strong>stationary</strong> or contain a <strong>unit
root</strong>, the package provides the classical IPS test alongside three wavelet-based
tests (Wavelet Ratio IPS, and Wald tests on DWT / MODWT energy). Critical values are
obtained by Monte Carlo simulation under the null of independent random walks. Wavelet
tests gain power when breaks or cross-sectional dependence are present.</p>
"""

# --------------------------------------------------------------------------- #
#  CSS — light journal/paper theme
# --------------------------------------------------------------------------- #
CSS = r"""
:root{
  --bg:#ffffff; --surface:#f6f8fb; --surface2:#eef2f7;
  --ink:#1a1f29; --muted:#5b6573; --line:#e3e8ef;
  --accent:#0072B2; --accent2:#009E73; --accent3:#E69F00;
  --shadow:0 1px 3px rgba(20,30,50,.06),0 8px 24px rgba(20,30,50,.06);
  --radius:14px;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  margin:0;color:var(--ink);background:var(--bg);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  line-height:1.65;font-size:17px;
}
.wrap{max-width:1060px;margin:0 auto;padding:0 22px}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
h1,h2,h3,h4{font-family:Georgia,"Times New Roman",serif;line-height:1.25;color:#10151d}
h2{font-size:1.9rem;margin:2.6rem 0 1rem;padding-bottom:.4rem;border-bottom:2px solid var(--line)}
h3{font-size:1.35rem;margin:1.8rem 0 .6rem}
h4{font-size:1.1rem;margin:1.2rem 0 .4rem;color:var(--accent)}
p{margin:.7rem 0}
code{font-family:"SFMono-Regular",Consolas,"Liberation Mono",Menlo,monospace;
  background:var(--surface2);padding:.12em .4em;border-radius:6px;font-size:.88em}
pre{background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--accent);
  border-radius:10px;padding:14px 16px;overflow:auto;font-size:.82rem;line-height:1.45}
pre code{background:none;padding:0}

/* hero */
.hero{background:
  radial-gradient(1200px 400px at 80% -10%,rgba(0,114,178,.10),transparent 60%),
  radial-gradient(900px 400px at 0% 0%,rgba(0,158,115,.08),transparent 55%),
  linear-gradient(180deg,#fbfdff,#ffffff);
  border-bottom:1px solid var(--line);padding:64px 0 40px}
.hero .wave{font-size:3rem}
.hero h1{font-size:2.8rem;margin:.2rem 0 .3rem}
.hero .tag{font-size:1.25rem;color:var(--muted);margin:0 0 1.2rem}
.badges{display:flex;flex-wrap:wrap;gap:8px;margin:.6rem 0 1.2rem}
.badges img{height:22px}
.btnrow{display:flex;flex-wrap:wrap;gap:10px;margin-top:6px}
.btn{display:inline-flex;align-items:center;gap:7px;padding:9px 16px;border-radius:10px;
  font-weight:600;font-size:.92rem;border:1px solid var(--line);background:#fff;color:var(--ink);
  box-shadow:var(--shadow)}
.btn.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
.btn.alt{background:var(--accent2);color:#fff;border-color:var(--accent2)}
.btn:hover{text-decoration:none;transform:translateY(-1px);transition:.15s}

/* author card */
.author{display:flex;gap:16px;align-items:center;margin-top:26px;padding:16px 18px;
  background:#fff;border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow);max-width:560px}
.avatar{width:54px;height:54px;border-radius:50%;flex:0 0 auto;
  background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff;
  display:flex;align-items:center;justify-content:center;font-weight:700;font-size:1.3rem;font-family:Georgia,serif}
.author .nm{font-weight:700}
.author .mu{color:var(--muted);font-size:.9rem}

/* nav */
nav.toc{position:sticky;top:0;z-index:10;background:rgba(255,255,255,.92);
  backdrop-filter:saturate(160%) blur(8px);border-bottom:1px solid var(--line)}
nav.toc .wrap{display:flex;gap:18px;flex-wrap:wrap;padding:11px 22px;font-size:.9rem;font-weight:600}
nav.toc a{color:var(--muted)}
nav.toc a:hover{color:var(--accent);text-decoration:none}

/* content blocks */
section{padding:6px 0}
blockquote{margin:1rem 0;padding:.6rem 1rem;border-left:4px solid var(--accent3);
  background:#fffaf0;border-radius:8px;color:#5a4a26}
figure{margin:1.6rem 0;text-align:center}
figure img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:12px;
  box-shadow:var(--shadow);background:#fff;padding:6px}
figcaption{color:var(--muted);font-size:.88rem;margin-top:.5rem;font-style:italic}

/* tables */
.tablewrap{overflow-x:auto;margin:1.2rem 0;border:1px solid var(--line);border-radius:12px;
  box-shadow:var(--shadow);background:#fff;padding:6px}
table{border-collapse:collapse;width:100%;font-size:.9rem}
table th,table td{padding:8px 12px;border-bottom:1px solid var(--line);text-align:right}
table th{background:var(--surface2);color:#10151d;font-weight:700;text-align:right}
table caption{caption-side:top;font-weight:700;padding:8px;color:#10151d;text-align:left}
table tr:hover td{background:var(--surface)}

/* console (rich) output */
.console{background:#fbfcfe;border:1px solid var(--line);border-left:4px solid var(--accent2);
  border-radius:10px;padding:12px 14px;overflow:auto;font-size:.74rem;line-height:1.35;
  font-family:"SFMono-Regular",Consolas,monospace;color:#2a3340;white-space:pre}

.codecell{margin:1.1rem 0}
.codecell .src{background:#0b1c2c0d}
details.code{margin:.6rem 0}
details.code summary{cursor:pointer;color:var(--muted);font-size:.85rem;font-weight:600}

footer{margin-top:60px;padding:34px 0;border-top:1px solid var(--line);background:var(--surface);
  color:var(--muted);font-size:.9rem}
footer .wrap{display:flex;justify-content:space-between;flex-wrap:wrap;gap:14px}
.kbd{font-size:.8rem;color:var(--muted)}
@media(max-width:640px){.hero h1{font-size:2.1rem}body{font-size:16px}}
"""


def md2html(text: str) -> str:
    return md.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )


def main():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    fig_n = 0
    body = []

    for cell in nb["cells"]:
        ctype = cell["cell_type"]
        if ctype == "markdown":
            src = join(cell["source"])
            # skip the notebook's own title/author block (we have a custom hero)
            if "# 🌊 PyWaveletPanel" in src or src.strip().startswith("## 0. Setup"):
                continue
            body.append(f'<div class="md">{md2html(src)}</div>')
            continue

        # code cell -------------------------------------------------------- #
        src = join(cell["source"])
        parts = []
        # collapsible source code
        if src.strip():
            esc = (src.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            parts.append(
                f'<details class="code"><summary>Show code</summary>'
                f'<pre><code>{esc}</code></pre></details>'
            )

        for out in cell.get("outputs", []):
            data = out.get("data", {})
            if "image/png" in data:
                fig_n += 1
                raw = join(data["image/png"])
                fn = f"fig_{fig_n:02d}.png"
                (ASSETS / fn).write_bytes(base64.b64decode(raw))
                parts.append(
                    f'<figure><img src="assets/{fn}" alt="Figure {fig_n}">'
                    f'<figcaption>Figure {fig_n}</figcaption></figure>'
                )
            elif "text/html" in data:
                html = join(data["text/html"])
                parts.append(f'<div class="tablewrap">{html}</div>')
            elif out.get("output_type") == "stream":
                txt = strip_ansi(join(out.get("text", "")))
                if txt.strip():
                    esc = txt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    parts.append(f'<div class="console">{esc}</div>')
            elif "text/plain" in data and "image/png" not in data:
                # skip bare repr lines like "<Styler ...>"
                t = join(data["text/plain"]).strip()
                if t and not t.startswith("<") and "Styler" not in t:
                    esc = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    parts.append(f'<div class="console">{esc}</div>')

        if parts:
            body.append('<div class="codecell">' + "\n".join(parts) + "</div>")

    initials = "".join(w[0] for w in AUTHOR.replace("Dr.", "").split()[:2]).upper()

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PyWaveletPanel — Wavelet-Based Panel Data Econometrics</title>
<meta name="description" content="PyWaveletPanel: wavelet-based panel data econometrics in Python — scale-by-scale regression, SAW structural breaks, and panel unit-root tests. By {AUTHOR}.">
<link rel="preconnect" href="https://cdn.jsdelivr.net">
<script>
  window.MathJax={{tex:{{inlineMath:[['$','$']],displayMath:[['$$','$$']]}}}};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>{CSS}</style>
</head>
<body>

<header class="hero">
  <div class="wrap">
    <div class="wave">🌊</div>
    <h1>PyWaveletPanel</h1>
    <p class="tag">Wavelet-Based Panel Data Econometrics in Python</p>
    <div class="badges">
      <a href="{PYPI}"><img src="https://img.shields.io/pypi/v/pywaveletpanel.svg?color=blue" alt="PyPI"></a>
      <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python">
      <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT">
    </div>
    <div class="btnrow">
      <a class="btn primary" href="{GITHUB}">★ GitHub Repository</a>
      <a class="btn alt" href="{PYPI}">⬇ Install from PyPI</a>
      <a class="btn" href="{ISSUES}">Issue Tracker</a>
    </div>
    <div class="author">
      <div class="avatar">{initials}</div>
      <div>
        <div class="nm">{AUTHOR}</div>
        <div class="mu">📧 <a href="mailto:{EMAIL}">{EMAIL}</a></div>
        <div class="mu">🔗 <a href="{GH_USER}">github.com/merwanroudane</a></div>
      </div>
    </div>
  </div>
</header>

<nav class="toc"><div class="wrap">
  <a href="#about">Overview</a>
  <a href="#theory">Theory</a>
  <a href="#demo">Worked Demo</a>
  <a href="#install">Install</a>
  <a href="#refs">References</a>
</div></nav>

<main class="wrap">
  <section id="about">
    <h2>Overview</h2>
    <p><strong>PyWaveletPanel</strong> brings wavelet methods to panel-data
    econometrics. It implements five published methodologies behind a clean,
    consistent API and produces journal-quality tables and publication-grade,
    light-themed plots. The worked example below uses the real
    <em>Grunfeld (1958)</em> investment panel (11 U.S. firms, 1935–1954).</p>
    <ul>
      <li><strong>Multiresolution decomposition</strong> (MODWT / DWT, Haar &amp; LA(8)).</li>
      <li><strong>Scale-by-scale panel regression</strong> with fixed effects and Newey–West HAC errors.</li>
      <li><strong>SAW structural-break detection</strong> with Post-SAW Chow tests.</li>
      <li><strong>Panel unit-root tests</strong>: classical IPS plus three wavelet-based tests.</li>
    </ul>
  </section>

  <section id="theory">
    <h2>Methodology &amp; Theory</h2>
    {THEORY_HTML}
  </section>

  <section id="demo">
    <h2>Worked Demonstration — Grunfeld Investment Panel</h2>
    <p>Every figure and table below is generated automatically from the executed
    notebook <code>PyWaveletPanel_Demo.ipynb</code>.</p>
    {''.join(body)}
  </section>

  <section id="install">
    <h2>Installation</h2>
    <pre><code>pip install pywaveletpanel</code></pre>
    <p>Or from source: <a href="{GITHUB}">{GITHUB}</a></p>
  </section>

  <section id="refs">
    <h2>References</h2>
    <ol>
      <li>Bada et al. (2021). <em>A Wavelet Method for Panel Models with Jump Discontinuities</em>. arXiv:2109.10950.</li>
      <li>Karlsson et al. (2020). <em>Oil Prices &amp; Exchange Rates: A Wavelet Panel Analysis</em>. The Energy Journal 41(1).</li>
      <li>Almasri et al. (2016). <em>Wavelet-based Panel Unit-root Test with Structural Breaks</em>. Applied Economics.</li>
      <li>Gallegati et al. (2015). <em>Productivity and Unemployment: Scale-by-scale Panel Analysis</em>. SNDE.</li>
      <li>Li &amp; Shukur (2013). <em>Unit Roots in Panel Data Using a Wavelet Ratio</em>. Computational Economics 41.</li>
      <li>Grunfeld, Y. (1958). <em>The Determinants of Corporate Investment</em>. PhD thesis, University of Chicago.</li>
    </ol>
  </section>
</main>

<footer><div class="wrap">
  <div>
    <div><strong>{AUTHOR}</strong></div>
    <div class="kbd">📧 <a href="mailto:{EMAIL}">{EMAIL}</a> · 🔗 <a href="{GH_USER}">github.com/merwanroudane</a></div>
  </div>
  <div class="kbd">
    <a href="{GITHUB}">GitHub</a> · <a href="{PYPI}">PyPI</a> · <a href="{ISSUES}">Issues</a><br>
    Built with PyWaveletPanel · MIT License
  </div>
</div></footer>

</body>
</html>"""

    (DOCS / "index.html").write_text(html, encoding="utf-8")
    print(f"Wrote {DOCS/'index.html'}  ({len(html):,} bytes)")
    print(f"Extracted {fig_n} figures into {ASSETS}")


if __name__ == "__main__":
    main()
