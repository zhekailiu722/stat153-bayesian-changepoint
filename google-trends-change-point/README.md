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

The model assumes independent Gaussian errors:

$$
y_t = \beta_0 + \beta_1\mathbf{1}(t > c) + \varepsilon_t,
\qquad \varepsilon_t \overset{\mathrm{iid}}{\sim} N(0,\sigma^2),
\qquad t=1,\ldots,n.
$$

Here, $\beta_0$ is the mean before the change, $\beta_1$ is the change in the mean, and $\sigma$ is the common noise standard deviation. The mean after the change is $\beta_0+\beta_1$. The change point $c$ is the last observation in the first regime.

### Priors implied by the implementation

The code does not declare prior distributions separately. Its posterior formulas correspond to the following prior specification, using the same prior convention for every candidate change point:

| Parameter | Prior | Interpretation |
| --- | --- | --- |
| $c$ | Discrete uniform on $\{1,\ldots,n-1\}$ | Every split with two nonempty regimes has equal prior probability |
| $\boldsymbol\beta=(\beta_0,\beta_1)^\top$ | $\pi(\boldsymbol\beta\mid\sigma^2,c)\propto 1$ on $\mathbb R^2$ | Flat prior on the intercept and mean shift |
| $v=\sigma^2$ | $\pi(v\mid c)\propto 1/v$, for $v>0$ | Scale prior on the variance; equivalently, $\pi(\sigma\mid c)\propto 1/\sigma$ |

Thus the joint prior density/mass is proportional to $1/\sigma^2$ over the stated support. The coefficient and variance priors are **improper**: they do not integrate to one. The same unspecified multiplicative constant is used across candidate splits. No proper normal-prior variance or inverse-gamma-prior hyperparameters are specified. The posterior is proper when $n>2$, each design matrix has full column rank, and each candidate's RSS is positive, as in the included dataset.

### Change-point posterior

For a fixed $c$, let $X_c$ have rows $[1,\mathbf{1}(t>c)]$, let $p=2$ be its number of columns, and define

$$
\widehat{\boldsymbol\beta}_c=(X_c^\top X_c)^{-1}X_c^\top\mathbf y,
\qquad
\mathrm{RSS}_c=\|\mathbf y-X_c\widehat{\boldsymbol\beta}_c\|^2.
$$

Integrating out $\boldsymbol\beta$ and $\sigma^2$ gives

$$
P(c\mid\mathbf y)\propto
|X_c^\top X_c|^{-1/2}\,
\mathrm{RSS}_c^{-(n-p)/2},
\qquad c=1,\ldots,n-1.
$$

This is the formula implemented in `logpost(c)`:

```text
log_weight(c) = ((p - n) / 2) * log(RSS_c) - 0.5 * log(det(X_c.T @ X_c))
```

The script subtracts the maximum log weight before exponentiating for numerical stability, then normalizes the weights to sum to one. The separate estimate `c_est` minimizes RSS; it need not equal the posterior mode because the posterior also includes the determinant factor.

### Posterior sampling

Using seed `42`, the script generates 1,000 joint posterior draws in this order:

1. **Change point:** draw $c^{(i)}$ from the discrete probabilities $P(c\mid\mathbf y)$ using `rng.choice`.
**2. Noise variance:** conditional on the sampled change point, with the coefficients integrated out,

$$
\sigma^2 \mid c,\mathbf{y}
\sim \mathrm{InvGamma}\left(
\frac{n-p}{2},
\frac{\mathrm{RSS}_c}{2}
\right).
$$

Here $\mathrm{InvGamma}(a,b)$ has density proportional to
$v^{-a-1}\exp(-b/v)$ for $v>0$.

Equivalently, draw $Z^{(i)} \sim \chi^2_{n-p}$ and set

$$
\sigma^{(i)}
=
\sqrt{\frac{\mathrm{RSS}_{c^{(i)}}}{Z^{(i)}}}.
$$

This is exactly the `chirv` / `sig_sample` calculation.
For the included data, $n-p=254-2=252$.

3. **Regression coefficients:** conditional on the sampled change point and variance,

   $$
   \boldsymbol\beta\mid\sigma^2,c,\mathbf y\sim
   N_2\left(\widehat{\boldsymbol\beta}_c,\;\sigma^2(X_c^\top X_c)^{-1}\right).
   $$

   This is the distribution passed to `rng.multivariate_normal`. Each row of `post_samples` stores $[c^{(i)},\beta_0^{(i)},\beta_1^{(i)},\sigma^{(i)}]$; the last entry is the standard deviation, not the variance.

These are direct draws from the joint posterior through its marginal and conditional distributions, rather than an MCMC chain; no burn-in or thinning is needed. Each plotted curve is the sampled mean $\beta_0^{(i)}+\beta_1^{(i)}\mathbf{1}(t>c^{(i)})$, without added observation noise.

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

## Publishing

Suggested repository name: `google-trends-change-point`

Suggested description: `Single-change-point analysis of Google Trends search interest with posterior sampling and uncertainty visualization.`

Extract the package and upload the **contents** of its `google-trends-change-point` folder to the repository root, rather than uploading the ZIP itself. Include `.gitignore`, `data/`, and `figures/`. The repository contains the input data, script, dependencies, and generated figures needed to reproduce the displayed result.
