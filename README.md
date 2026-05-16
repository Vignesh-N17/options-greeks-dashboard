# Options Greeks Dashboard

Interactive dashboard for visualizing Black-Scholes options Greeks in real time.

Built with Dash + Plotly. Core Black-Scholes math implemented from scratch using NumPy and SciPy — no options pricing library used.

## Features

- Real-time Greeks calculation — Delta, Gamma, Theta, Vega, Rho
- Interactive sliders for spot price, strike, volatility, time to expiry, risk-free rate
- 2D line charts + 3D surface plots via Plotly
- Pure analytical Black-Scholes via `bs_greeks` function

## Tech Stack

```
Python · Dash · Plotly · NumPy · SciPy
```

## Run locally

```bash
pip install dash plotly numpy scipy
python app.py
```

Then open `http://localhost:8050` in browser.

## How it works

All Greeks computed analytically inside `bs_greeks` using Black-Scholes closed-form solutions. No QuantLib or mibian — just math.

```
Delta  = N(d1)
Gamma  = N'(d1) / (S · σ · √T)
Theta  = -(S · N'(d1) · σ) / (2√T) - r · K · e^(-rT) · N(d2)
Vega   = S · N'(d1) · √T
Rho    = K · T · e^(-rT) · N(d2)
```
