from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# The original Google Trends CSV is included in data/multiTimeline.csv.
DATA_PATH = Path(__file__).resolve().parent / "data" / "multiTimeline.csv"
if not DATA_PATH.is_file():
    raise FileNotFoundError(
        f"Missing input: {DATA_PATH}\n"
        "Copy your Google Trends CSV here, keeping its first two metadata rows."
    )

df = pd.read_csv(DATA_PATH, skiprows=2)
y = df['mask: (United States)'].values
n=len(y)
def rss(c):
    x = np.arange(1, n+1)
    xc = (x > c).astype(float)
    X = np.column_stack([np.ones(n), xc])

    md = sm.OLS(y, X).fit()
    rss = np.sum(md.resid ** 2)
    return rss
allcvals = np.arange(1, n)
rssvals = np.array([rss(c) for c in allcvals])
c_est = allcvals[np.argmin(rssvals)]
print(f'c_est: {c_est}')

def logpost(c):
    x = np.arange(1, n + 1)
    xc = (x > c).astype(float)
    X = np.column_stack([np.ones(n), xc])
    p = X.shape[1]

    md = sm.OLS(y, X).fit()
    rss = np.sum(md.resid ** 2)
    sgn, log_det = np.linalg.slogdet(np.dot(X.T, X))
    logval = ((p - n) / 2) * np.log(rss) - 0.5 * log_det
    return logval
logpostvals = np.array([logpost(c) for c in allcvals])
postvals_unnormalized = np.exp(logpostvals - np.max(logpostvals))
postvals = postvals_unnormalized / (np.sum(postvals_unnormalized))

N = 1000
rng = np.random.default_rng(seed = 42)
cpostsamples = rng.choice(allcvals, N, replace = True, p = postvals)
ci_c = np.percentile(cpostsamples, [2.5, 97.5])
print(f'ci_c: {ci_c}')

#b
post_samples = np.zeros(shape = (N, 4))
post_samples[:, 0] = cpostsamples

for i in range(N):
    c = cpostsamples[i]

    x = np.arange(1, n + 1)
    xc = (x > c).astype(float)
    X = np.column_stack([np.ones(n), xc])
    p = X.shape[1]
    
    md_c = sm.OLS(y, X).fit()
    chirv = rng.chisquare(df = n - p)
    sig_sample = np.sqrt(np.sum(md_c.resid ** 2) / chirv)
    post_samples[i, (p + 1)] = sig_sample

    covmat = (sig_sample ** 2) * np.linalg.inv(np.dot(X.T, X))
    beta_sample = rng.multivariate_normal(mean = md_c.params, cov = covmat, size = 1)
    post_samples[i, 1:(p + 1)] = beta_sample

# --- Plot: observations and posterior fitted means ---
# All quantities below summarize existing draws; estimation is unchanged.
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

plot_x = np.arange(1, n + 1)
plot_months = pd.to_datetime(df['Month'], format='%Y-%m')
date_range = f'{plot_months.iloc[0]:%b %Y}–{plot_months.iloc[-1]:%b %Y}'
fitted_draws = (
    post_samples[:, 1, None]
    + post_samples[:, 2, None]
    * (plot_x[None, :] > cpostsamples[:, None])
)
lower, median, upper = np.percentile(fitted_draws, [2.5, 50, 97.5], axis=0)

colors = {
    'ink': '#243746',
    'muted': '#617280',
    'teal': '#087F8C',
    'draw': '#5AA7AE',
    'band': '#D9EDEE',
    'accent': '#C57735',
    'grid': '#E7EDF0',
}

with plt.rc_context({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.labelcolor': colors['ink'],
    'xtick.color': colors['muted'],
    'ytick.color': colors['muted'],
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
}):
    fig = plt.figure(figsize=(13.2, 6.9), facecolor='white')
    ax = fig.add_axes([0.085, 0.20, 0.87, 0.56])

    fig.text(0.085, 0.925, 'Search interest, before and after a change',
             fontsize=22, fontweight='bold', color=colors['ink'])
    fig.text(0.085, 0.882,
             f'Google Trends  /  mask  /  United States  /  {date_range}',
             fontsize=11, color=colors['muted'])
    fig.text(0.955, 0.882, f'{N:,} posterior draws',
             ha='right', fontsize=11, color=colors['muted'])

    ax.set_axisbelow(True)
    ax.grid(axis='y', color=colors['grid'], linewidth=0.8)
    ax.fill_between(plot_x, lower, upper, step='mid',
                    color=colors['band'], linewidth=0, zorder=1)

    # Keep every posterior curve, but prevent the ensemble from hiding the data.
    for fitted in fitted_draws:
        ax.step(plot_x, fitted, where='mid', color=colors['draw'],
                alpha=0.018, linewidth=0.55, rasterized=True, zorder=2)

    ax.step(plot_x, median, where='mid', color=colors['teal'],
            linewidth=2.5, zorder=3)
    ax.plot(plot_x, y, color=colors['ink'], linewidth=1.55, zorder=4)

    # c is the last observation before the change. The boundary falls between
    # observations c and c + 1, consistent with the model indicator (x > c).
    split_x = c_est + 0.5
    ax.axvline(split_x, color=colors['accent'], linewidth=1.25,
               linestyle=(0, (4, 4)), zorder=5)
    label_on_right = split_x < (n + 1) / 2
    split_month = plot_months.iloc[c_est - 1].strftime('%b %Y')
    ax.annotate(f'After {split_month} (c = {c_est})',
                xy=(split_x, 0.97), xycoords=('data', 'axes fraction'),
                xytext=(7 if label_on_right else -7, 0),
                textcoords='offset points',
                ha='left' if label_on_right else 'right', va='top',
                fontsize=10, color=colors['accent'],
                bbox={'facecolor': 'white', 'edgecolor': 'none',
                      'alpha': 0.90, 'pad': 3}, zorder=6)

    legend_handles = [
        Line2D([0], [0], color=colors['ink'], linewidth=1.8,
               label='Observed interest'),
        Line2D([0], [0], color=colors['draw'], linewidth=1.0,
               alpha=0.65, label='Posterior draws'),
        Line2D([0], [0], color=colors['teal'], linewidth=2.5,
               label='Posterior median'),
        Patch(facecolor=colors['band'], edgecolor='none',
              label='95% pointwise credible interval'),
    ]
    fig.legend(handles=legend_handles, loc='lower left',
               bbox_to_anchor=(0.085, 0.795), ncol=4, frameon=False,
               fontsize=10, handlelength=2.5, columnspacing=2.0,
               labelcolor=colors['muted'], borderaxespad=0)

    ax.set_xlabel('Year (monthly observations)', labelpad=12)
    ax.set_ylabel('Search interest (index)', labelpad=12)
    ax.set_xlim(0.5, n + 0.5)
    ax.margins(y=0.15)
    # Keep the model's one-based positions; display the supplied calendar dates.
    tick_indices = np.flatnonzero(
        (plot_months.dt.month == 1)
        & ((plot_months.dt.year - plot_months.iloc[0].year) % 4 == 0)
    )
    ax.set_xticks(plot_x[tick_indices])
    ax.set_xticklabels(plot_months.iloc[tick_indices].dt.strftime('%Y'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(axis='both', length=0, pad=8)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('bottom', 'left'):
        ax.spines[side].set_color(colors['grid'])
        ax.spines[side].set_linewidth(0.8)

    fig.text(0.085, 0.080,
             'Shading summarizes uncertainty in the fitted mean, not future observations.',
             fontsize=9.5, color=colors['muted'])
    fig.text(0.085, 0.050,
             'The dashed line marks the minimum-RSS split; the change occurs after observation c.',
             fontsize=9.5, color=colors['muted'])

    output_dir = Path(__file__).resolve().parent / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'change_point_fit.png', dpi=300, facecolor='white')
    fig.savefig(output_dir / 'change_point_fit.svg', facecolor='white')
    plt.show()
