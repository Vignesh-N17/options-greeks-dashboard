"""
app.py
Real-time Greeks Dashboard using Dash.
"""

import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import warnings
warnings.filterwarnings("ignore")

# ── Black-Scholes engine ──────────────────────────────────────────────────────
def bs_greeks(S, K, T, r, q, sigma, is_call=True):
    if T <= 0 or sigma <= 0:
        return {k: 0.0 for k in ["price","delta","gamma",
                                   "vega","theta","rho","vanna","volga"]}
    d1   = (np.log(S/K) + (r-q+0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2   = d1 - sigma*np.sqrt(T)
    nd1  = norm.cdf(d1);  nd2  = norm.cdf(d2)
    nnd1 = norm.cdf(-d1); nnd2 = norm.cdf(-d2)
    npd1 = norm.pdf(d1)
    disc_r = np.exp(-r*T); disc_q = np.exp(-q*T)
    sqrtT  = np.sqrt(T)

    if is_call:
        price = S*disc_q*nd1  - K*disc_r*nd2
        delta = disc_q*nd1
        rho   = K*T*disc_r*nd2 / 100.0
    else:
        price = K*disc_r*nnd2 - S*disc_q*nnd1
        delta = -disc_q*nnd1
        rho   = -K*T*disc_r*nnd2 / 100.0

    gamma = disc_q*npd1 / (S*sigma*sqrtT)
    vega  = S*disc_q*npd1*sqrtT / 100.0
    theta = (-S*disc_q*npd1*sigma/(2*sqrtT)
             - r*K*disc_r*(nd2  if is_call else -nnd2)
             + q*S*disc_q*(nd1  if is_call else -nnd1)) / 365.0
    vanna = -disc_q*npd1*d2/sigma
    volga =  S*disc_q*npd1*sqrtT*d1*d2/sigma

    return {"price":price,"delta":delta,"gamma":gamma,
            "vega":vega,"theta":theta,"rho":rho,
            "vanna":vanna,"volga":volga}

# ── Styles ────────────────────────────────────────────────────────────────────
CARD = {"background":"#1e1e2e","border":"1px solid #444",
        "borderRadius":"8px","padding":"15px","margin":"8px"}
LBL  = {"color":"#cdd6f4","fontSize":"13px","marginBottom":"4px"}

# ── App layout ────────────────────────────────────────────────────────────────
app = dash.Dash(__name__)
app.title = "Greeks Dashboard"

app.layout = html.Div(
    style={"background":"#13131f","minHeight":"100vh",
           "fontFamily":"monospace","color":"#cdd6f4"},
    children=[

    html.H2("⚡ Real-Time Options Greeks Dashboard",
            style={"textAlign":"center","color":"#89b4fa",
                   "padding":"20px 0 5px 0"}),

    # Controls
    html.Div(
        style={"display":"flex","flexWrap":"wrap",
               "justifyContent":"center","padding":"10px"},
        children=[

        html.Div(children=[
            html.Label("Spot Price (S)", style=LBL),
            dcc.Slider(id="S", min=400, max=700, step=1, value=560,
                marks={400:"400",500:"500",600:"600",700:"700"},
                tooltip={"placement":"bottom"}),
        ], style={**CARD, "width":"280px"}),

        html.Div(children=[
            html.Label("Strike (K)", style=LBL),
            dcc.Slider(id="K", min=400, max=700, step=1, value=560,
                marks={400:"400",500:"500",600:"600",700:"700"},
                tooltip={"placement":"bottom"}),
        ], style={**CARD, "width":"280px"}),

        html.Div(children=[
            html.Label("Time to Expiry (years)", style=LBL),
            dcc.Slider(id="T", min=0.02, max=2.0, step=0.01, value=0.25,
                marks={0.02:"1wk",0.25:"3m",0.5:"6m",1.0:"1y",2.0:"2y"},
                tooltip={"placement":"bottom"}),
        ], style={**CARD, "width":"280px"}),

        html.Div(children=[
            html.Label("Implied Volatility (%)", style=LBL),
            dcc.Slider(id="sigma", min=5, max=80, step=1, value=18,
                marks={5:"5%",20:"20%",40:"40%",60:"60%",80:"80%"},
                tooltip={"placement":"bottom"}),
        ], style={**CARD, "width":"280px"}),

        html.Div(children=[
            html.Label("Risk-Free Rate (%)", style=LBL),
            dcc.Slider(id="r", min=0, max=10, step=0.1, value=4.3,
                marks={0:"0%",5:"5%",10:"10%"},
                tooltip={"placement":"bottom"}),
        ], style={**CARD, "width":"200px"}),

        html.Div(children=[
            html.Label("Option Type", style=LBL),
            dcc.RadioItems(id="flag",
                options=[{"label":"Call","value":"call"},
                         {"label":"Put", "value":"put"}],
                value="call",
                style={"color":"#cdd6f4"},
                labelStyle={"marginRight":"15px"}),
        ], style={**CARD, "width":"150px"}),
    ]),

    # Greeks cards
    html.Div(id="greeks-display",
             style={"display":"flex","flexWrap":"wrap",
                    "justifyContent":"center","padding":"5px"}),

    # Charts
    html.Div(
        style={"display":"flex","flexWrap":"wrap","justifyContent":"center"},
        children=[
            dcc.Graph(id="greeks-chart",  style={"width":"95%","height":"450px"}),
            dcc.Graph(id="pnl-chart",     style={"width":"95%","height":"380px"}),
            dcc.Graph(id="surface-chart", style={"width":"95%","height":"450px"}),
        ]
    ),
])

# ── Callback ──────────────────────────────────────────────────────────────────
@app.callback(
    Output("greeks-display", "children"),
    Output("greeks-chart",   "figure"),
    Output("pnl-chart",      "figure"),
    Output("surface-chart",  "figure"),
    Input("S",     "value"),
    Input("K",     "value"),
    Input("T",     "value"),
    Input("sigma", "value"),
    Input("r",     "value"),
    Input("flag",  "value"),
)
def update(S, K, T, sigma_pct, r_pct, flag):
    sigma   = sigma_pct / 100.0
    r       = r_pct     / 100.0
    q       = 0.013
    is_call = (flag == "call")
    g       = bs_greeks(S, K, T, r, q, sigma, is_call)

    # ── Greeks cards ──────────────────────────────────────────────────────
    items = [
        ("Price",  f"${g['price']:.3f}", "#89b4fa"),
        ("Delta",  f"{g['delta']:.4f}",  "#a6e3a1"),
        ("Gamma",  f"{g['gamma']:.6f}",  "#fab387"),
        ("Vega",   f"{g['vega']:.4f}",   "#f9e2af"),
        ("Theta",  f"{g['theta']:.4f}",  "#f38ba8"),
        ("Rho",    f"{g['rho']:.4f}",    "#cba6f7"),
        ("Vanna",  f"{g['vanna']:.4f}",  "#94e2d5"),
        ("Volga",  f"{g['volga']:.4f}",  "#eba0ac"),
    ]
    cards = [
        html.Div(children=[
            html.Div(name, style={"color":"#888","fontSize":"11px"}),
            html.Div(val,  style={"color":col,"fontSize":"18px",
                                  "fontWeight":"bold","marginTop":"5px"}),
        ], style={**CARD,"textAlign":"center","width":"110px"})
        for name, val, col in items
    ]

    # ── Greeks vs Spot ────────────────────────────────────────────────────
    S_range = np.linspace(S*0.70, S*1.30, 100)
    prices  = [bs_greeks(s,K,T,r,q,sigma,is_call)["price"] for s in S_range]
    deltas  = [bs_greeks(s,K,T,r,q,sigma,is_call)["delta"] for s in S_range]
    gammas  = [bs_greeks(s,K,T,r,q,sigma,is_call)["gamma"] for s in S_range]
    thetas  = [bs_greeks(s,K,T,r,q,sigma,is_call)["theta"] for s in S_range]

    fig1 = make_subplots(rows=2, cols=2,
        subplot_titles=("Price","Delta","Gamma","Theta"),
        vertical_spacing=0.15)

    kw = dict(x=S_range, mode="lines", showlegend=False)
    fig1.add_trace(go.Scatter(**kw,y=prices,line=dict(color="#89b4fa",width=2)),1,1)
    fig1.add_trace(go.Scatter(**kw,y=deltas,line=dict(color="#a6e3a1",width=2)),1,2)
    fig1.add_trace(go.Scatter(**kw,y=gammas,line=dict(color="#fab387",width=2)),2,1)
    fig1.add_trace(go.Scatter(**kw,y=thetas,line=dict(color="#f38ba8",width=2)),2,2)

    for row in [1,2]:
        for col in [1,2]:
            fig1.add_vline(x=S,line=dict(color="white",dash="dash",width=1),row=row,col=col)
            fig1.add_vline(x=K,line=dict(color="yellow",dash="dot",width=1),row=row,col=col)

    fig1.update_layout(
        title="Greeks vs Spot (white=spot, yellow=strike)",
        paper_bgcolor="#13131f",plot_bgcolor="#1e1e2e",
        font=dict(color="#cdd6f4"),height=440)
    fig1.update_xaxes(gridcolor="#333")
    fig1.update_yaxes(gridcolor="#333")

    # ── P&L decomposition ─────────────────────────────────────────────────
    S_pnl    = np.linspace(S*0.75, S*1.25, 200)
    pnl_act  = [bs_greeks(s,K,T,r,q,sigma,is_call)["price"]-g["price"] for s in S_pnl]
    pnl_d    = [g["delta"]*(s-S) for s in S_pnl]
    pnl_dg   = [g["delta"]*(s-S)+0.5*g["gamma"]*(s-S)**2 for s in S_pnl]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=S_pnl,y=pnl_act,
        name="Actual P&L",line=dict(color="#89b4fa",width=2)))
    fig2.add_trace(go.Scatter(x=S_pnl,y=pnl_d,
        name="Delta approx",line=dict(color="#a6e3a1",width=2,dash="dash")))
    fig2.add_trace(go.Scatter(x=S_pnl,y=pnl_dg,
        name="Delta+Gamma approx",line=dict(color="#fab387",width=2,dash="dot")))
    fig2.add_hline(y=0,line=dict(color="white",width=1))
    fig2.add_vline(x=S,line=dict(color="white",dash="dash",width=1))
    fig2.update_layout(
        title="P&L: Actual vs Delta vs Delta-Gamma Approximation",
        xaxis_title="Spot Price",yaxis_title="P&L ($)",
        paper_bgcolor="#13131f",plot_bgcolor="#1e1e2e",
        font=dict(color="#cdd6f4"),height=370,
        legend=dict(bgcolor="#1e1e2e"))
    fig2.update_xaxes(gridcolor="#333")
    fig2.update_yaxes(gridcolor="#333")

    # ── Price surface ─────────────────────────────────────────────────────
    k_grid  = np.linspace(-0.25, 0.25, 30)
    T_grid  = np.linspace(0.05,  2.0,  25)
    Z = np.array([
        [bs_greeks(S, S*np.exp(k), t, r, q, sigma, is_call)["price"]
         for k in k_grid]
        for t in T_grid
    ])

    fig3 = go.Figure(data=[go.Surface(
        x=k_grid, y=T_grid, z=Z,
        colorscale="Viridis",
        colorbar=dict(title="Price ($)")
    )])
    fig3.update_layout(
        title="Option Price Surface",
        scene=dict(
            xaxis_title="Log-Moneyness",
            yaxis_title="Maturity (years)",
            zaxis_title="Price ($)",
            bgcolor="#13131f"),
        paper_bgcolor="#13131f",
        font=dict(color="#cdd6f4"),height=440)

    return cards, fig1, fig2, fig3

if __name__ == "__main__":
    app.run(debug=True, port=8050)