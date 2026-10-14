# MonteCarlo-Option-Pricer

A comprehensive, object-oriented Python engine for pricing financial derivatives using Monte Carlo simulations. This repository provides robust implementations for both European and American options, alongside variance-reduced computations for the Greeks.

## Features

- **European Option Pricing:** Fast, vectorized Monte Carlo engine for Call and Put options.
- **American Option Pricing (LSMC):** Implementation of the Longstaff-Schwartz algorithm using polynomial regression to estimate the conditional expectation of continuation values and compute early exercise premiums.
- **Greeks Computation:** Calculation of Delta and Gamma using Central Finite Differences coupled with **Common Random Numbers (CRN)** to drastically reduce statistical variance.
- **Path Visualization:** Generates and plots discrete Geometric Brownian Motion (GBM) trajectories for intuitive analysis.

## Mathematical Background

The underlying asset dynamics are modeled using a **Geometric Brownian Motion (GBM)** under the risk-neutral measure:

$$S_T = S_0 \exp\left(\left(r - \frac{\sigma^2}{2}\right)T + \sigma \sqrt{T} Z\right)$$

Where $Z \sim \mathcal{N}(0,1)$. 

For American options, the **Longstaff-Schwartz Algorithm** utilizes backward induction and Ordinary Least Squares (OLS) regression on In-The-Money (ITM) paths. This resolves the optimal stopping problem by comparing the immediate exercise payoff against the estimated continuation value at each discrete time step.

## Dependencies

- `numpy`: Core library for vectorized array operations and standard normal random number generation.
- `scipy`: Used strictly for analytical Black-Scholes benchmark comparisons.
- `matplotlib`: For stochastic trajectory visualizations.

## Quick Start

Clone the repository and run the main script to execute the pricing engines and compare the outputs.

```bash
git clone [https://github.com/your-username/MonteCarlo-Option-Pricer.git](https://github.com/your-username/MonteCarlo-Option-Pricer.git)
cd MonteCarlo-Option-Pricer
python pricer.py
