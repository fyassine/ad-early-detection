# Publication-Ready Figures in Python — Style Guide & Journal Audit Checklist

> Applies when creating, modifying, or auditing plots and figure generation scripts for `THESIS/` (and repository-wide publications).
>
> **Visual Reference**: Follow the style example illustrated in [`DOCS/figures_style/image.png`](file:///mnt/e/fyassine/ad-early-detection/DOCS/figures_style/image.png).
>
> ("Simple & elegant figures for research papers, with examples in Julia and Python") and the author's
>
> **Provenance caveat:** the repo's `.py` file bodies could not be rendered through the GitHub connector at
> retrieval time. The principles below come from the author's own description of the repo; the Python code is
> a faithful, annotated reconstruction of the repo's method, not a verbatim copy of
> `01_main_simple_example/Python/scripts/beautiful_figure_example.py`. Verify against the repo before citing
> parameter values as "theirs".
>
> **Notebook Location & Creation Policy**: All standalone plotting workflows and figure generator notebooks must be stored in [`DOCS/plots_notebooks/`](file:///mnt/e/fyassine/ad-early-detection/DOCS/plots_notebooks). If no notebook exists for a requested or needed plot, always create a new Jupyter notebook inside [`DOCS/plots_notebooks/`](file:///mnt/e/fyassine/ad-early-detection/DOCS/plots_notebooks) to generate, document, and reproduce the figure.

---

## Part 1 — The eight design principles

The rules the author followed when building the figures in the repo:

1. **Overall quality** — use vector graphics (SVG, PDF) for pixel-free, scalable, publication-ready figures.
2. **Readability** — use clear fonts of proper sizes and optimize the layout of figures.
3. **Simplify and declutter** — remove unnecessary elements; make the figure simple and effective.
4. **Colours** — use fewer colours, apply them strategically, and select harmonious schemes.
5. **Message and story** — decide what you want to tell the reader before polishing anything.
6. **Consistent style** — maintain consistent fonts, colours, and formatting across all figures.
7. **To avoid** — no pie charts and no 3D plots; they mislead the reader and hinder analysis
   (which is why the repo's main example is 2D).
8. **Time** — allocate enough time to iterate, refine, and polish.

These converge with the canonical Rougier et al. *Ten Simple Rules for Better Figures*
([PLOS Comput. Biol.](https://pmc.ncbi.nlm.nih.gov/articles/PMC4161295/)): "do not trust the defaults",
"avoid chartjunk", "message trumps beauty".

---

## Part 2 — The repo's working method

Core idea: **iterative tuning**. Start from a fully default plot and improve parameters one by one —
"the answer lies in the number of plotting parameters tuned."

The 2D example folder (`01_main_simple_example/Python`) mirrors that pedagogy:

| File | Role |
|---|---|
| `scripts/not_a_very_beautiful_figure_example.py` | The all-defaults starting point |
| `scripts/beautiful_figure_example.py` | The same data, fully tuned |
| `visualisation_steps_python.png` | The step-by-step progression |
| `purple_teal_palette.png` | The harmonious two-hue palette (principle 4) |
| `manuscript_screenshot_with_two_python_figures.png` | The final in-context test |

**The full workflow:**

1. Tune parameters in Python (matplotlib).
2. Export SVG/PDF.
3. Post-process in vector software (CorelDRAW, or the free Inkscape).
4. Test how the figure looks inside the actual LaTeX / Word manuscript at real print size.

---

## Part 3 — Master matplotlib rcParams template

Define the style **once** and reuse it for every figure in the paper (principle 6). Drop this into a
shared `plotstyle.py` and import it in every figure script.

```python
# plotstyle.py — one style block for the whole paper (Principle 6)
import matplotlib.pyplot as plt

MM = 1 / 25.4  # mm → inches conversion for exact journal sizing

plt.rcParams.update({
    # Typography (Principle 2: BeautifulFigures style with Courier New)
    'font.family': 'Courier New',
    'font.size': 8,                 # body text inside figures: 7–9 pt typical
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.8,

    # Frame & ticks (Principle 3: declutter)
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'legend.frameon': False,

    # Vector output (Principle 1)
    'pdf.fonttype': 42,             # TrueType embedding — text stays text
    'svg.fonttype': 'none',         # text editable in Inkscape/CorelDRAW
})

def journal_fig(width_mm, height_mm=None):
    """Figure sized exactly to the journal column (Principle 2)."""
    height_mm = height_mm or width_mm * 0.68   # pleasant default aspect
    return plt.subplots(figsize=(width_mm * MM, height_mm * MM))
```

---

## Part 4 — Exact journal sizing rules

Design each figure at its **final print width** — never rescale in the manuscript.

| Publisher | Single column | Intermediate | Double column | Fonts | Resolution / format |
|---|---|---|---|---|---|
| **IEEE** ([guidelines](https://proceedingsoftheieee.ieee.org/resources/guidelines-for-figures-and-tables/), [Author Center](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/resolution-and-size/)) | 3.5 in / 88.9 mm | — | 7.16 in / 182 mm | **9–10 pt**; Helvetica, Times New Roman, Arial, Cambria, Symbol | >300 dpi colour/greyscale, >600 dpi B/W line art; PS, EPS, PDF, PNG, TIFF |

Rule of thumb across publishers: (IEEE); double column ≈ 17–19 cm.

---

## Part 5 — Colour: strategic, minimal, colorblind-safe

- The repo uses a **two-hue purple–teal pair** instead of the full default cycle (principle 4).
- Avoid red–green combinations — indistinguishable for readers with colour-vision deficiency
  (explicitly called out in IEEE's guidelines).
- The standard colorblind-safe categorical palette is **Okabe–Ito**
  ([hex/RGB/CMYK reference](https://sci-draw.com/figure-accessibility-kit)):

| Name | HEX |
|---|---|
| Black | `#000000` |
| Orange | `#E69F00` |
| Sky blue | `#56B4E9` |
| Bluish green | `#009E73` |
| Yellow | `#F0E442` |
| Blue | `#0072B2` |
| Vermillion | `#D55E00` |
| Reddish purple | `#CC79A7` |

```python
# Colorblind-safe defaults for categorical data
OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
             '#0072B2', '#D55E00', '#CC79A7', '#000000']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=OKABE_ITO)

# The repo's two-hue approach (Principle 4) for a two-series plot
PURPLE, TEAL = '#7b2d8b', '#1b9e97'

# For continuous maps: perceptually uniform + colorblind-safe
# plt.imshow(data, cmap='viridis')   # or 'cividis' — never 'jet'
```

---

## Part 6 — Before / after example (the repo's core demo)

### Before — defaults everywhere (what `not_a_very_beautiful_figure_example.py` demonstrates)

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 200)
y1 = np.sin(x) * np.exp(-0.2 * x)
y2 = np.cos(x) * np.exp(-0.2 * x)

plt.figure()                      # default size, default fonts
plt.plot(x, y1)                   # default colour cycle
plt.plot(x, y2)
plt.xlabel('Time, s')
plt.ylabel('Amplitude')
plt.legend(['Series 1', 'Series 2'])
plt.savefig('figure.png')         # low-res raster — Principle 1 violated
```

### After — same data, every rule applied

```python
import numpy as np
import matplotlib.pyplot as plt
from plotstyle import journal_fig, PURPLE, TEAL  # Part 3 block (Principle 6)

x = np.linspace(0, 10, 200)
y1 = np.sin(x) * np.exp(-0.2 * x)
y2 = np.cos(x) * np.exp(-0.2 * x)

# Principle 2: sized to the journal column (Nature single = 89 mm)
fig, ax = journal_fig(width_mm=89, height_mm=60)

ax.plot(x, y1, color=PURPLE, lw=1.5, label='Series 1')   # Principle 4
ax.plot(x, y2, color=TEAL,  lw=1.5, label='Series 2')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude (a.u.)')

ax.spines[['top', 'right']].set_visible(False)           # Principle 3
ax.legend(loc='upper right')

fig.tight_layout(pad=0.3)
fig.savefig('figure.pdf')          # Principle 1: vector for the journal
fig.savefig('figure.svg')          # for the Inkscape/CorelDRAW pass
fig.savefig('figure.png', dpi=600) # sharp raster fallback (IEEE: >300 dpi)
```

---

## Part 7 — Journal figure audit checklist

Run this over every figure (and every script) before submission.

### Sizing
- [ ] Figure width equals the target journal's column width exactly (see Part 4 table)
- [ ] Figure created at final size — no `\includegraphics[width=...]` rescaling that shrinks fonts
- [ ] Height leaves room for the caption (Nature: ≤170 mm incl. legend; IEEE: ≤8.5 in depth)
- [ ] Multi-panel figures sized to double column, panels legible at that width

### Typography
- [ ] Font matching style guidance: **`Courier New`** (monospace, matching BeautifulFigures reference `image.png`), with fallbacks `DejaVu Sans Mono`, `Liberation Mono`
- [ ] Font size within journal range (Nature 5–7 pt; IEEE 9–10 pt; generic 7–9 pt)
- [ ] All labels, ticks, and legends readable at 100% print size — no sub-5 pt text
- [ ] Bold/italic used sparingly

### Colour
- [ ] ≤3–4 hues per figure; colours carry meaning (principle 4)
- [ ] No red–green pairings; palette is colorblind-safe (Okabe–Ito or equivalent)
- [ ] Continuous data uses `viridis`/`cividis`, never `jet`
- [ ] Figure still interpretable when printed in greyscale

### Format & output
- [ ] Vector export (PDF/SVG/EPS) for all line art — no rasterized plots
- [ ] `pdf.fonttype = 42` / `svg.fonttype = 'none'` so text stays editable
- [ ] Any unavoidable raster is ≥300 dpi (colour), ≥600 dpi (B/W line art)
- [ ] File format is on the journal's accepted list (IEEE: PS/EPS/PDF/PNG/TIFF)

### Content & consistency (principles 3, 5, 6)
- [ ] One clear message per figure — you can state it in one sentence
- [ ] No chartjunk: top/right spines off, no legend frames, minimal gridlines
- [ ] No pie charts, no decorative 3D (principle 7)
- [ ] Identical fonts, colours, and line widths across ALL figures in the paper
- [ ] Figure viewed inside the compiled manuscript (LaTeX/Word) at final size — the repo's last step
- [ ] Caption is self-contained (symbols, abbreviations, colour codes defined — IEEE requirement)

### Quick CLI audit across your scripts

```bash
# Find raster saves missing an explicit dpi
grep -rn "savefig" --include="*.py" . | grep "\.png" | grep -v "dpi"

# Find scripts that bypass the shared style block
grep -rL "plotstyle" --include="*.py" ./figures

# Catch the usual suspects: jet colormap, pie charts, 3D
grep -rn "cmap=.jet\|plt.pie\|projection=.3d" --include="*.py" .
```

---

## Part 8 — Text Visibility & Anti-Occlusion Rules

> **Canonical location**: [`.agents/rules/plots.md § Part 8`](file:///mnt/e/fyassine/ad-early-detection/.agents/rules/plots.md)
> is the **single source of truth** for Part 8.  Do **not** edit or duplicate Part 8 here —
> update only the canonical file.  This avoids rule drift between the two files.
>
> **Summary for quick reference** (full spec with code examples is in the canonical file):
>
> - Every `ax.text()` / `ax.annotate()` that overlaps data **must** have either a white bbox
>   (`alpha 0.7–0.9`) or a halo path-effect (`pe.withStroke`), plus `zorder ≥ 8`.
> - Use `textcoords='offset points'` with `xytext=(0, 12)` style offsets for callouts;
>   reserve data-coordinate arithmetic for geometric placements only.
> - The arrow patch and text label in a single `ax.annotate` are one artist — `zorder` lifts
>   both together; that is harmless but means independent layering requires a separate
>   `FancyArrowPatch`.
> - For direction labels on dense scatter (e.g. UMAP), choose the perpendicular side with
>   fewer nearby points and document the choice with an inline comment.
> - Run the Part 8.4 checklist and inspect every figure at 100 % zoom before saving.