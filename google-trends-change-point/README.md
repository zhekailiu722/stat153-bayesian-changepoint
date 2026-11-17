# Google Trends Change-Point Analysis

A single-change-point analysis of U.S. Google Trends search interest for **mask**, with posterior uncertainty visualized alongside the observed series.

The script searches for a shift in the series mean, computes a discrete posterior over possible change points, and draws 1,000 posterior samples of the change point, regression coefficients, and noise standard deviation.

## Results

![Google Trends change-point fit using the included original data](figures/change_point_fit.png)

The figure above uses the original CSV included in this repository: **254 monthly observations from January 2004 through February 2025**, for `mask` in the United States, with category `All categories`.

| Output | Result |
| --- | --- |
| Minimum-RSS change point, `c_est` | `193` |
| Calendar interpretation | The first regime ends in January 2020; the second starts in February 2020 |
| Sampled 95% percentile interval for `c`, `ci_c` | `[187, 194]` |
| Posterior draws / seed | `1,000 / 42` |

The index interval describes possible last observations of the first regime, from July 2019 through February 2020. The plotted dashed line marks the minimum-RSS split. The model estimates a single shift in the mean; the later peak and decline remain visible in the observed series.

## Run the analysis

Use Python 3.12, the version used for validation.

```bash
python -m pip install -r requirements.txt
```

The original Google Trends export is already included at `data/multiTimeline.csv`; no separate data download or path edit is needed. Its bytes are unchanged from the supplied `multiTimeline (1).csv`. The script skips the first two lines and reads the numeric column `mask: (United States)`. The `Month` column supplies calendar labels for the plot.

From the repository directory, run:

```bash
python change_point_analysis.py
```

The script prints `c_est` and `ci_c`, opens the figure, and saves:

- `figures/change_point_fit.png` — 300 dpi image.
- `figures/change_point_fit.svg` — scalable text and axes; the dense posterior curves are rasterized to keep file size manageable.

For an environment without a display, use a non-interactive Matplotlib backend, for example on macOS/Linux:

```bash
MPLBACKEND=Agg python change_point_analysis.py
```

Input and output paths are resolved relative to the script, so the script also works when invoked from another directory.

## Reading the plot

| Visual | Meaning |
| --- | --- |
| Dark line | Observed search interest |
| Faint teal lines | All 1,000 sampled fitted-mean curves |
| Bold teal line | Pointwise posterior median of the fitted mean |
| Pale teal band | Pointwise 2.5th–97.5th percentiles of the sampled fitted means |
| Dashed amber line | Split selected by minimizing residual sum of squares (RSS) |

The credible band describes uncertainty in the **fitted mean**, not future observations. It is a pointwise interval, not a simultaneous band for the whole curve. Faint posterior draws can extend outside it.

The horizontal axis displays years from the CSV's `Month` column, with one observation per month. The model still uses observation indices `1, ..., n`. A change point `c` means observations `1, ..., c` precede the change and observations `c + 1, ..., n` follow it. Step curves and the dashed boundary are drawn between the corresponding observations, at `c + 0.5`. The annotation names the last month before the change.

The median curve is a pointwise summary of many sampled step functions, so it need not itself have only one change point.

## Model and computation

The model is

$$
y_t = \beta_0 + \beta_1\mathbf{1}(t > c) + \varepsilon_t,
\qquad \varepsilon_t \sim N(0,\sigma^2).
$$

The original implementation is retained:

1. Fit an OLS model at each candidate `c = 1, ..., n - 1` and minimize RSS.
2. Compute and normalize the change-point log weights
   `((p - n) / 2) * log(RSS) - 0.5 * log(det(X.T @ X))`, with `p = 2`.
3. Sample 1,000 change points using random seed `42`.
4. Sample the noise standard deviation and regression coefficients conditionally on each sampled change point.

`ci_c` is the 2.5th–97.5th percentile interval from the sampled change points. It is distinct from the fitted-mean band shown in the plot.

The included data contain finite numeric values, no missing entries, and a complete sequence of monthly dates. The original statistical assumptions and numerical edge cases are unchanged. If replacing the dataset, supply a finite numeric series with more than two observations and positive residual sum of squares at every candidate split, plus a chronological `Month` column in `YYYY-MM` format. Values such as `<1`, missing entries, and other nonnumeric cells require an explicit preprocessing decision before using this script.

## Validation

The script was run on the included original dataset. Its RSS values, normalized change-point posterior, sampled change points, `ci_c`, and sampled regression/noise parameters were compared with the original script and were exactly equal. Both exported figures were generated successfully. Numerical inference, sample count, and seed are unchanged.

## Repository contents

| File | Purpose |
| --- | --- |
| `change_point_analysis.py` | Original inference code with the revised plotting section and a portable input path |
| `requirements.txt` | Versions used to validate the script |
| `.gitignore` | Excludes Python caches, virtual environments, and local editor files |
| `data/multiTimeline.csv` | Original monthly Google Trends data, copied without changes |
| `data/README.md` | Data coverage, format, and provenance |
| `figures/change_point_fit.png` | Actual analysis figure at 300 dpi |
| `figures/change_point_fit.svg` | Scalable version of the actual analysis figure |

Plotting adds readable legends, transparent posterior curves, a fitted-mean summary and interval, calendar labels from the original data, and automatic image export. Outside plotting, the only functional change is replacing the machine-specific CSV path with `data/multiTimeline.csv` and an informative missing-file message.
