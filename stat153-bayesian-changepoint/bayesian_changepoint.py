"""STAT 153 HW2, Problem 3: direct posterior sampling for a change point.

Based on Zhekai Liu's coursework script and the course Lab 5 example:
https://stat153.berkeley.edu/spring-2025/lab5
See README.md for the model, priors, provenance, and limitations.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures without requiring a desktop window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parent


def design_matrix(n, c):
    """c is the one-based index of the last observation before the change."""
    t = np.arange(1, n + 1)
    return np.column_stack([np.ones(n), (t > c).astype(float)])


def fit_posterior(y, draws=1000, seed=42):
    """Enumerate c, then sample c -> sigma^2 -> beta (not Gibbs/MCMC)."""
    n = len(y)
    p = 2
    candidates = np.arange(1, n)
    rss = np.empty(n - 1)
    means = np.empty((n - 1, p))
    inverse_xtx = np.empty((n - 1, p, p))
    log_weights = np.empty(n - 1)

    for k, c in enumerate(candidates):
        X = design_matrix(n, c)
        fit = sm.OLS(y, X).fit()
        rss[k] = np.sum(fit.resid**2)
        if rss[k] <= 0:
            raise ValueError("Positive residual sum of squares is required.")
        means[k] = fit.params
        xtx = X.T @ X
        sign, logdet = np.linalg.slogdet(xtx)
        if sign <= 0:
            raise ValueError("The design matrix must have full column rank.")
        inverse_xtx[k] = np.linalg.inv(xtx)
        # beta and sigma^2 have been integrated out here.
        log_weights[k] = -0.5 * logdet - (n - p) / 2 * np.log(rss[k])

    probabilities = np.exp(log_weights - log_weights.max())
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(seed)
    c_samples = rng.choice(candidates, size=draws, p=probabilities)
    samples = np.zeros((draws, 4))  # c, beta0, beta1, sigma
    samples[:, 0] = c_samples
    for i, c in enumerate(c_samples):
        k = c - 1
        # sigma^2 | c,y ~ IG((n-p)/2, RSS(c)/2).
        sigma = np.sqrt(rss[k] / rng.chisquare(n - p))
        beta = rng.multivariate_normal(means[k], sigma**2 * inverse_xtx[k])
        samples[i, 1:3] = beta
        samples[i, 3] = sigma

    # Exact discrete equal-tailed quantiles avoid Monte Carlo error in the CI.
    ci = candidates[np.searchsorted(np.cumsum(probabilities), [0.025, 0.975])]
    return candidates, rss, probabilities, samples, ci


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/mask_us_google_trends.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.draws < 1:
        parser.error("--draws must be positive")

    data = pd.read_csv(args.data, skiprows=2)
    dates = pd.to_datetime(data["Month"], format="%Y-%m")
    y = pd.to_numeric(data["mask: (United States)"], errors="raise").to_numpy(dtype=float)
    if len(y) <= 2 or not np.isfinite(y).all():
        raise ValueError("At least three finite observations are required.")
    periods = pd.PeriodIndex(dates, freq="M")
    if not np.all(np.diff(periods.asi8) == 1):
        raise ValueError("Observations must be consecutive months in chronological order.")

    candidates, rss, probabilities, samples, ci = fit_posterior(y, args.draws, args.seed)
    mle = int(candidates[np.argmin(rss)])
    map_c = int(candidates[np.argmax(probabilities)])
    date_at = lambda c: dates.iloc[int(c) - 1].strftime("%Y-%m")
    summary = {
        "observations": len(y), "start_month": date_at(1), "end_month": date_at(len(y)),
        "draws": args.draws, "seed": args.seed,
        "c_definition": "One-based index of the last month before the change; new level starts at c+1.",
        "mle_c": mle, "mle_last_prechange_month": date_at(mle),
        "mle_first_postchange_month": date_at(mle + 1),
        "posterior_mode_c": map_c,
        "posterior_mode_last_prechange_month": date_at(map_c),
        "credible_interval_95_c": ci.tolist(),
        "credible_interval_95_last_prechange_month": [date_at(c) for c in ci],
        "posterior_probability_at_mode": float(probabilities.max()),
        "sample_mean_beta0": float(samples[:, 1].mean()),
        "sample_mean_beta1": float(samples[:, 2].mean()),
        "sample_mean_sigma": float(samples[:, 3].mean()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame({
        "c": candidates, "last_prechange_month": [date_at(c) for c in candidates],
        "rss": rss, "posterior_probability": probabilities,
    }).to_csv(args.output_dir / "changepoint_posterior.csv", index=False)
    sample_df = pd.DataFrame(samples, columns=["c", "beta0", "beta1", "sigma"])
    sample_df["c"] = sample_df["c"].astype(int)
    sample_df.to_csv(args.output_dir / "posterior_samples.csv", index=False)

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    t = np.arange(1, len(y) + 1)
    functions = samples[:, 1, None] + samples[:, 2, None] * (t[None, :] > samples[:, 0, None])
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    for curve in functions[:min(200, args.draws)]:
        ax.plot(dates, curve, color="#dc7043", alpha=0.025, linewidth=0.7)
    ax.plot([], [], color="#dc7043", alpha=0.6, label="Posterior mean-function draws (up to 200)")
    ax.plot(dates, y, color="#34495e", linewidth=1.2, label="Observed search interest")
    ax.plot(dates, functions.mean(axis=0), color="#a52d20", linewidth=2, label="Posterior average mean function")
    ax.set(title="US search interest in 'mask': a single-change-point model", xlabel="Month", ylabel="Google Trends search interest")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.15)
    fig.savefig(args.output_dir / "posterior_fit.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
    ax.plot(dates.iloc[:-1], probabilities, color="#246e89", linewidth=1.5)
    ax.axvline(dates.iloc[map_c - 1], color="#a52d20", linestyle="--", label=f"Posterior mode: {date_at(map_c)}")
    ax.set(title="Posterior probability of the change point", xlabel="Last month before the change (c)", ylabel="Posterior probability")
    ax.legend()
    ax.grid(alpha=0.15)
    fig.savefig(args.output_dir / "changepoint_posterior.png", dpi=170)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
