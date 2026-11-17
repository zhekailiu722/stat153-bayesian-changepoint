import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

df =pd.read_csv("D:\\google download\\multiTimeline (1).csv", skiprows = 2)
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

plt.figure(figsize=(15,6))
plt.plot(y)

for i in range(N):
    c = cpostsamples[i]
    beta0 = post_samples[i, 1]
    beta1 = post_samples[i, 2]

    ftdval = beta0 +beta1 * (x > c).astype(float)
    plt.plot(ftdval, color = 'red')
plt.show()