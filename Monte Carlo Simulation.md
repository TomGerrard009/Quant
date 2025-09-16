# Quant
All quant related coding projects, open to improvements from communities.

### Commands to be used throughout
%config InlineBackend.figure_format = 'retina'
%matplotlib inline

import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

plt.rcParams.update({'font.size':14})

## ------- MONTE CARLO SIM FOR 3-ASSET PORTFOLIO ------------- #
n_assets = 3  

tickers = ["AAPL", "TSLA", "NVDA"]  #example tickers
prices = yf.download(tickers, period='1y', interval='1d', auto_adjust='True')['Close']

def pricing(symbol):
    ticker = yf.Ticker(symbol)
    last = ticker.history(period="1d")
    return last['Close'].iloc[-1]

### Defining parameters
T = 1 # time in years
N = 252 # trading days
dt = T/N # timestamp of steps 
M = 10000 # no of simulations


### Log returns of prices 
log_ret = np.log(prices / prices.shift(1)).dropna()  

### Drift (mu) and volatility (sigma) annualised
mu = np.array(log_ret.mean() * 252)   
sigma = np.array(log_ret.std() * np.sqrt(252))          

### Portfolio weights (sum to 1)          ### random weights for simulation and then optimise weightings.
weights = np.array([0.35, 0.37, 0.28])


### Correlation matrix
corr_matrix = log_ret.corr()

### Covariance Matrix calculated using Cholesky decomposition
D = np.diag(sigma)
cov_matrix = D @ corr_matrix @ D    
L = np.linalg.cholesky(cov_matrix) 

print("L shape:", L.shape)  
print("log_ret shape:2", log_ret.shape)        # checks shapes to see if calculations have been carried out properly.


# So is generated with real latest data
S0 = np.array([pricing(t) for t in tickers])

# --------- SIMULATING GBM PATHS ------------- #
sim_paths = np.zeros((M, N+1, n_assets))
sim_paths[:,0,:] = S0

# Simulate GBM with correlated shocks
for m in range(M):
    Z = np.random.normal(size=(N, n_assets))
    corr = Z @ L.T
    for t in range(1, N + 1):
        sim_paths[m, t, :] = sim_paths[m, t - 1, :] * np.exp(
            (mu - 0.5 * sigma**2) * dt + np.sqrt(dt) * corr[t - 1]
        )

# Compute weighted portfolio value per simulation & timestep
port_val = np.sum(sim_paths * weights, axis=2)  # shape (M, N+1)

# Scale portfolio so initial value = 100,000
initial_port_val= np.sum(S0 * weights)
scale_factor = 100000 / initial_port_val
port_val *= scale_factor

final_values = port_val[:, -1]
port_mean = np.mean(final_values)

# ------------ PLOTTING RESULTS ---------------- #
# Plotting side by side
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# ---- Left plot: Simulation paths ----
for i in range(10000):
    axs[0].plot(port_val[i])
axs[0].set_title("Sample Simulated Portfolio Paths")
axs[0].set_xlabel("Time Step (Day)")
axs[0].set_ylabel("Portfolio Value ($)")
axs[0].grid(True)

# Add portfolio weights and starting prices as text box
weights_percent = weights * 100
textstr = '\n'.join((
    "Portfolio Weights & Starting Prices:",
    *(f"{tickers[i]}: {weights_percent[i]:.1f}% @ ${S0[i]:.2f}" for i in range(n_assets))
))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
axs[0].text(0.05, 0.95, textstr, transform=axs[0].transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

# ---- Right plot: Histogram of final portfolio values ----
axs[1].hist(final_values, bins=50, edgecolor='black')
axs[1].set_title("Distribution of Final Portfolio Values")
axs[1].set_xlabel("Portfolio Value at T ($)")
axs[1].set_ylabel("Frequency")
axs[1].grid(True)

# Compute VaR at 95% confidence
VaR_95 = np.percentile(final_values, 5)

# Plot VaR vertical line
axs[1].axvline(VaR_95, color='r', linestyle='--', label=f'VaR (5%): ${VaR_95:,.0f}')

# Calculate % chance portfolio ends below initial 100k (loss)
pct_loss = np.mean(final_values < 100000) * 100
axs[1].axvline(100000, color='orange', linestyle='--', label='Initial Portfolio Value')
axs[1].legend()

# Add text box for VaR and loss %
textstr2 = f"Value at Risk (5% quantile): ${VaR_95:,.2f}\n" \
           f"Chance of Loss (below $100k): {pct_loss:.2f}%"
# Add text box for VaR and loss % in top right, just above legend
axs[1].text(0.95, 0.95, textstr2, transform=axs[1].transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=props)


plt.tight_layout()
plt.show()

# text box code used from k-dickinson, monte carlo simulation code as I love the clean look and the use of a VaR percentile as a risk measure.

print("Portfolio mean value: ${:.3f}".format(port_mean))
print("Return: {:.3f}%".format(port_mean / 100000))

# --------------- WEIGHT OPTIMISATION METHOD 1 (MAXIMISING SHARPE RATIO) ----------------- #
returns = log_ret
Rf = 4.08     # latest could be taken from T10 bonds

def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.0):    # defining portfolio statistics
    returns = np.dot(weights, mean_returns)
    std_dev = np.sqrt(weights.T @ cov_matrix @ weights)
    sharpe_ratio = (returns - risk_free_rate) / std_dev
    return returns, std_dev, sharpe_ratio

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=Rf):           # setting up optimisation: Maximise Sharpe Ratio
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

mean_returns = returns.mean()           # running optimisation
num_assets = len(mean_returns)
init_guess = [1. / num_assets] * num_assets
bounds = tuple((0, 1) for _ in range(num_assets))
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

opt_result = minimize(negative_sharpe, init_guess,
                      args=(mean_returns, cov_matrix, 0.0),
                      method='SLSQP', bounds=bounds, constraints=constraints)

opt_weights = opt_result.x       # new optimal weightings

# ---------------- NEW PLOTS AND SIMULATED PATHS WITH OPTIMAL WEIGHTINGS --------------- # 
sim_paths = np.zeros((M, N+1, n_assets))
sim_paths[:,0,:] = S0

# Simulate GBM with correlated shocks
for m in range(M):
    Z = np.random.normal(size=(N, n_assets))
    corr = Z @ L.T
    for t in range(1, N + 1):
        sim_paths[m, t, :] = sim_paths[m, t - 1, :] * np.exp(
            (mu - 0.5 * sigma**2) * dt + np.sqrt(dt) * corr[t - 1]
        )

# Compute weighted portfolio value per simulation & timestep
opt_port_val = np.sum(sim_paths * opt_weights, axis=2)  # shape (M, N+1)

# Scale portfolio so initial value = 100,000
opt_initial_port_val= np.sum(S0 * opt_weights)
opt_scale_factor = 100000 / opt_initial_port_val
opt_port_val *= opt_scale_factor

opt_final_values = opt_port_val[:, -1]
opt_port_mean = np.mean(opt_final_values)

# Plotting side by side
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# ---- Left plot: Simulation paths ----
for i in range(10000):
    axs[0].plot(opt_port_val[i])
axs[0].set_title("Sample Simulated Portfolio Paths")
axs[0].set_xlabel("Time Step (Day)")
axs[0].set_ylabel("Portfolio Value ($)")
axs[0].grid(True)

# Add portfolio weights and starting prices as text box
opt_weights_percent = opt_weights * 100
textstr = '\n'.join((
    "Portfolio Weights & Starting Prices:",
    *(f"{tickers[i]}: {opt_weights_percent[i]:.1f}% @ ${S0[i]:.2f}" for i in range(n_assets))
))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
axs[0].text(0.05, 0.95, textstr, transform=axs[0].transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

# ---- Right plot: Histogram of final portfolio values ----
axs[1].hist(opt_final_values, bins=50, edgecolor='black')
axs[1].set_title("Distribution of Final Optimal Portfolio Values")
axs[1].set_xlabel("Portfolio Value at T ($)")
axs[1].set_ylabel("Frequency")
axs[1].grid(True)

# Compute VaR at 95% confidence
opt_VaR_95 = np.percentile(opt_final_values, 5)

# Plot VaR vertical line
axs[1].axvline(opt_VaR_95, color='r', linestyle='--', label=f'VaR (5%): ${opt_VaR_95:,.0f}')

# Calculate % chance portfolio ends below initial 100k (loss)
opt_pct_loss = np.mean(opt_final_values < 100000) * 100
axs[1].axvline(100000, color='orange', linestyle='--', label='Initial Portfolio Value')
axs[1].legend()

# Add text box for VaR and loss %
textstr2 = f"Value at Risk (5% quantile): ${opt_VaR_95:,.2f}\n" \
           f"Chance of Loss (below $100k): {opt_pct_loss:.2f}%"
# Add text box for VaR and loss % in top right, just above legend
axs[1].text(0.95, 0.95, textstr2, transform=axs[1].transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=props)


plt.tight_layout()
plt.show()

print("Portfolio mean value: ${:.3f}".format(port_mean))
print("Return: {:.3f}%".format(port_mean / 100000))
print(opt_weights)


# -------------- WEIGHT OPTIMISATION METHOD 2 (MAX RETURN PER UNIT RISK) -------------------- #
def portfolio_performance(weights, mean_returns, cov_matrix):
    port_return = np.dot(weights, mean_returns)
    port_std = np.sqrt(weights.T @ cov_matrix @ weights)
    return port_return, port_std

# Maximise return per unit risk
def negative_return_risk(weights, mean_returns, cov_matrix):
    port_return, port_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return - (port_return / port_std)   # negative Sharpe-like ratio

n_assets = len(mean_returns)
init_guess = np.array([1.0 / n_assets] * n_assets)
bounds = tuple((0, 1) for _ in range(n_assets))
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # weights sum to 1


alt_result = minimize(
    negative_return_risk,
    init_guess,
    args=(mean_returns, cov_matrix),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

alt_weights = alt_result.x
alt_return, alt_risk = portfolio_performance(alt_weights, mean_returns, cov_matrix)

print("Optimal Weights:", alt_weights.round(4))
print(f"Expected Annual Return: {port_mean:.2%}")
print(f"Annual Volatility (Risk): {alt_risk:.2%}")

# ---------- NEW PLOTS AND SIMULATION WITH WEIGHTS ------------- #
sim_paths = np.zeros((M, N+1, n_assets))
sim_paths[:,0,:] = S0

# Simulate GBM with correlated shocks
for m in range(M):
    Z = np.random.normal(size=(N, n_assets))
    corr = Z @ L.T
    for t in range(1, N + 1):
        sim_paths[m, t, :] = sim_paths[m, t - 1, :] * np.exp(
            (mu - 0.5 * sigma**2) * dt + np.sqrt(dt) * corr[t - 1]
        )

# Compute weighted portfolio value per simulation & timestep
alt_port_val = np.sum(sim_paths * alt_weights, axis=2)  # shape (M, N+1)

# Scale portfolio so initial value = 100,000
alt_initial_port_val= np.sum(S0 * alt_weights)
alt_scale_factor = 100000 / alt_initial_port_val
alt_port_val *= alt_scale_factor

alt_final_values = alt_port_val[:, -1]

# Plotting side by side
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# ---- Left plot: Simulation paths ----
for i in range(10000):
    axs[0].plot(alt_port_val[i])
axs[0].set_title("Sample Simulated Portfolio Paths")
axs[0].set_xlabel("Time Step (Day)")
axs[0].set_ylabel("Portfolio Value ($)")
axs[0].grid(True)

# Add portfolio weights and starting prices as text box
alt_weights_percent = alt_weights * 100
textstr = '\n'.join((
    "Portfolio Weights & Starting Prices:",
    *(f"{tickers[i]}: {alt_weights_percent[i]:.1f}% @ ${S0[i]:.2f}" for i in range(n_assets))
))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
axs[0].text(0.05, 0.95, textstr, transform=axs[0].transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

# ---- Right plot: Histogram of final portfolio values ----
axs[1].hist(alt_final_values, bins=50, edgecolor='black')
axs[1].set_title("Distribution of Final Optimal Portfolio Values")
axs[1].set_xlabel("Portfolio Value at T ($)")
axs[1].set_ylabel("Frequency")
axs[1].grid(True)

# Compute VaR at 95% confidence
alt_VaR_95 = np.percentile(alt_final_values, 5)

# Plot VaR vertical line
axs[1].axvline(alt_VaR_95, color='r', linestyle='--', label=f'VaR (5%): ${alt_VaR_95:,.0f}')

# Calculate % chance portfolio ends below initial 100k (loss)
alt_pct_loss = np.mean(alt_final_values < 100000) * 100
axs[1].axvline(100000, color='orange', linestyle='--', label='Initial Portfolio Value')
axs[1].legend()

# Add text box for VaR and loss %
textstr2 = f"Value at Risk (5% quantile): ${alt_VaR_95:,.2f}\n" \
           f"Chance of Loss (below $100k): {alt_pct_loss:.2f}%"
# Add text box for VaR and loss % in top right, just above legend
axs[1].text(0.95, 0.95, textstr2, transform=axs[1].transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=props)


plt.tight_layout()
plt.show()

print("Portfolio mean value: ${:.3f}".format(port_mean))
print("Return: {:.3f}%".format(port_mean / 100000))
print(alt_weights)

# -------- EVALUATION OF WEIGHT OPTIMISATION EFFECT ------------ #
- Disappointing to see that both weight optimisation techniques made no effect at all to the returns, but both techniques produces a decrease in value at risk. Most effect at decreasing VaR (by ~$5000) at the 5% level was optimisation by maximising the sharpe ratio.
- It is also important to consider that the number of simulations outweighs the effect of the optimisations applied, as new simulated paths may under/over perform and cause an unaltered mean as randomness still exists.
- A possible improvement is adjusted the weight throughout the simulation so that it is theoretically, constantly optimal. However, in real markets, adjusting the weights so often would be unrealistic due to position size and most likely detrimental to profit because of transaction costs. This would also increase the runtime of the Monte Carlo simulation by an massive amount as the weight is being adjusted for 10,000 simulations 253 times.

- I am also sure both weight optimisation codes do a very similar thing and therefore the difference in results (that can be viewed graphically), might be due to alterations in simulated paths for each Monte Carlo simulation and not solely down to the effect of the technique
