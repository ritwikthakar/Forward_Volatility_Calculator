# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 17:54:06 2025

@author: ritwi
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from math import log, sqrt, exp
from scipy.stats import norm
import altair as alt

# --- Utility Functions -----------------------------------------------------

def year_frac_from_date(expiry_date):
    exp = pd.to_datetime(expiry_date)
    now = datetime.now()
    if exp.tzinfo is not None:
        exp = exp.replace(tzinfo=None)
    delta_days = max(0.0, (exp - now).total_seconds() / 86400.0)
    return delta_days / 365.0

def implied_vol(option_type, price, spot, strike, r, T, tol=1e-6, max_iter=100):
    if price <= 0 or T <= 0:
        return np.nan
    sigma = 0.2
    for _ in range(max_iter):
        d1 = (log(spot / strike) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        if option_type == "call":
            model = spot * norm.cdf(d1) - strike * exp(-r * T) * norm.cdf(d2)
        else:
            model = strike * exp(-r * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        vega = spot * norm.pdf(d1) * sqrt(T)
        diff = price - model
        if abs(diff) < tol:
            return sigma
        sigma += diff / max(vega, 1e-8)
        sigma = max(sigma, 1e-6)
    return sigma

# --- Main Analysis Function ------------------------------------------------

def analyze_ticker(ticker, front_exp, back_exp, r=0.02, multiplier=100):
    tk = yf.Ticker(ticker)
    spot = tk.history(period="1d")["Close"].iloc[-1]

    T1, T2 = year_frac_from_date(front_exp), year_frac_from_date(back_exp)
    oc1, oc2 = tk.option_chain(front_exp), tk.option_chain(back_exp)

    def atm_iv(chain, T):
        df = pd.concat([chain.calls.assign(type='call'), chain.puts.assign(type='put')]).reset_index(drop=True)
        df['strike'] = df['strike'].astype(float)
        df['diff'] = np.abs(df['strike'] - spot)
        row = df.loc[df['diff'].idxmin()]
        iv = float(row.get('impliedVolatility', np.nan))
        if pd.isna(iv) or iv == 0:
            mid = np.nanmean([row.get('bid', np.nan), row.get('ask', np.nan)])
            iv = implied_vol(row['type'], mid, spot, row['strike'], r, T)
        return iv

    iv1, iv2 = atm_iv(oc1, T1), atm_iv(oc2, T2)
    var1, var2 = iv1**2 * T1, iv2**2 * T2
    fwd_var = max((var2 - var1) / (T2 - T1), 0)
    fwd_vol = sqrt(fwd_var)
    fwd_factor = (iv1 - fwd_vol) / fwd_vol if iv1 > 0 else np.nan

    def calc_gamma(chain, T):
        df = pd.concat([chain.calls.assign(type="call"), chain.puts.assign(type="put")]).reset_index(drop=True)
        df["gamma"] = np.nan
        for i, row in df.iterrows():
            sigma = row.get("impliedVolatility", np.nan)
            if pd.isna(sigma):
                continue
            strike = row["strike"]
            d1 = (log(spot / strike) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
            df.loc[i, "gamma"] = np.exp(-0.5 * d1**2) / (spot * sigma * sqrt(2 * np.pi * T))
        return df

    def gamma_exposure(df):
        return np.nansum(df["gamma"] * df["openInterest"] * multiplier * spot * 0.01)

    def gamma_flip(df):
        df = df.copy()
        df["gamma_contrib"] = df["gamma"] * df["openInterest"] * np.sign(df["type"].map({"call": 1, "put": -1}))
        df.dropna(subset=["gamma_contrib"], inplace=True)
        if df.empty:
            return np.nan
        return np.average(df["strike"], weights=df["gamma_contrib"].abs())

    # --- Expected Move Calculation ---
    def expected_move(spot, atm_iv, T):
        em = spot * atm_iv * sqrt(T)
        return em, spot + em, spot - em

    df1, df2 = calc_gamma(oc1, T1), calc_gamma(oc2, T2)
    gex1, gex2 = gamma_exposure(df1), gamma_exposure(df2)
    flip1, flip2 = gamma_flip(df1), gamma_flip(df2)

    em1, upper1, lower1 = expected_move(spot, iv1, T1)
    em2, upper2, lower2 = expected_move(spot, iv2, T2)

    df1_plot = df1.reset_index()
    df2_plot = df2.reset_index()

    return {
        "spot": spot,
        "iv1": iv1,
        "iv2": iv2,
        "forward_vol": fwd_vol,
        "forward_factor": fwd_factor,
        "gamma_exp_front": gex1,
        "gamma_exp_back": gex2,
        "gamma_flip_front": flip1,
        "gamma_flip_back": flip2,
        "expected_move_front": em1,
        "upper_front": upper1,
        "lower_front": lower1,
        "expected_move_back": em2,
        "upper_back": upper2,
        "lower_back": lower2,
        "df1": df1_plot,
        "df2": df2_plot
    }

# --- Streamlit App ---------------------------------------------------------

st.title("📈 Options Dashboard with Forward Vol, Gamma & Expected Move")

ticker_input = st.text_input("Enter ticker symbol:", value="COIN").upper()

if ticker_input:
    tk = yf.Ticker(ticker_input)
    try:
        expiries = tk.options
        if not expiries:
            st.warning("No option expiries found.")
        else:
            st.subheader("Select Expiries")
            front_exp = st.selectbox("Front expiry", expiries)
            back_exp = st.selectbox("Back expiry", expiries, index=min(1, len(expiries)-1))

            if st.button("Analyze"):
                with st.spinner("Fetching data and computing metrics..."):
                    result = analyze_ticker(ticker_input, front_exp, back_exp)
                    st.success("Analysis Complete!")

                    # --- Key Metrics ---
                    st.subheader("📌 Key Metrics")
                    st.metric("Spot Price", result["spot"])
                    st.metric("Front ATM IV", round(result["iv1"],4))
                    st.metric("Back ATM IV", round(result["iv2"],4))
                    st.metric("Forward Volatility", round(result["forward_vol"],4))
                    st.metric("Forward Factor", round(result["forward_factor"],4))
                    st.metric("Gamma Exposure Front", round(result["gamma_exp_front"],2))
                    st.metric("Gamma Exposure Back", round(result["gamma_exp_back"],2))
                    st.metric("Gamma Flip Front", round(result["gamma_flip_front"],2))
                    st.metric("Gamma Flip Back", round(result["gamma_flip_back"],2))

                    # --- Expected Move Metrics ---
                    st.subheader("📌 Expected Move")
                    st.metric("Front EM", round(result["expected_move_front"],2))
                    st.metric("Front Upper Price", round(result["upper_front"],2))
                    st.metric("Front Lower Price", round(result["lower_front"],2))
                    st.metric("Back EM", round(result["expected_move_back"],2))
                    st.metric("Back Upper Price", round(result["upper_back"],2))
                    st.metric("Back Lower Price", round(result["lower_back"],2))

                    # --- Gamma Exposure Tables ---
                    st.subheader("📊 Gamma Exposure Table (Front Expiry)")
                    df1_plot = result["df1"][["index","strike","gamma","openInterest"]].copy()
                    df1_plot["gamma_exposure"] = df1_plot["gamma"]*df1_plot["openInterest"]*100*result["spot"]*0.01
                    st.dataframe(df1_plot.style.background_gradient(subset=["gamma_exposure"], cmap="RdYlGn"))

                    st.subheader("📊 Gamma Exposure Table (Back Expiry)")
                    df2_plot = result["df2"][["index","strike","gamma","openInterest"]].copy()
                    df2_plot["gamma_exposure"] = df2_plot["gamma"]*df2_plot["openInterest"]*100*result["spot"]*0.01
                    st.dataframe(df2_plot.style.background_gradient(subset=["gamma_exposure"], cmap="RdYlGn"))

                    # --- IV vs Strike Chart with Tooltips ---
                    st.subheader("📈 ATM IV vs Strike with Gamma Flip & Expected Move")
                    df1_iv = result["df1"][["strike","impliedVolatility"]].assign(Expiry="Front")
                    df2_iv = result["df2"][["strike","impliedVolatility"]].assign(Expiry="Back")
                    iv_df = pd.concat([df1_iv, df2_iv])
                    iv_chart = alt.Chart(iv_df).mark_line().encode(
                        x="strike",
                        y="impliedVolatility",
                        color="Expiry",
                        tooltip=["strike","impliedVolatility","Expiry"]
                    )

                    # Gamma flip vertical lines
                    flip_lines = pd.DataFrame({
                        "strike":[result["gamma_flip_front"],result["gamma_flip_back"]],
                        "label":["Front Flip","Back Flip"]
                    })
                    flip_chart = alt.Chart(flip_lines).mark_rule(color='red').encode(
                        x='strike',
                        tooltip=['label','strike']
                    )
                    
                    # Expected move lines
                    em_lines = pd.DataFrame({
                        "strike":[result["upper_front"],result["lower_front"],result["upper_back"],result["lower_back"]],
                        "label":["Front Upper","Front Lower","Back Upper","Back Lower"]
                    })
                    em_chart = alt.Chart(em_lines).mark_rule(color='blue', strokeDash=[5,5]).encode(
                        x='strike',
                        tooltip=['label','strike']
                    )

                    st.altair_chart((iv_chart + flip_chart + em_chart).interactive(), use_container_width=True)

                    # --- Gamma Exposure vs Strike Chart with Tooltips ---
                    st.subheader("📈 Gamma Exposure vs Strike with Gamma Flip & Expected Move")
                    df1_chart = result["df1"].assign(Expiry="Front")
                    df2_chart = result["df2"].assign(Expiry="Back")
                    gex_df = pd.concat([df1_chart, df2_chart])
                    gex_df["gamma_exposure"] = gex_df["gamma"]*gex_df["openInterest"]*100*result["spot"]*0.01

                    gex_chart = alt.Chart(gex_df).mark_line().encode(
                        x="strike",
                        y="gamma_exposure",
                        color="Expiry",
                        tooltip=["strike","gamma","openInterest","gamma_exposure","Expiry"]
                    )

                    # Gamma flip and expected move lines with tooltips
                    gex_chart_final = gex_chart + \
                                     alt.Chart(flip_lines).mark_rule(color='red').encode(
                                         x='strike', tooltip=['label','strike']) + \
                                     alt.Chart(em_lines).mark_rule(color='blue', strokeDash=[5,5]).encode(
                                         x='strike', tooltip=['label','strike'])

                    st.altair_chart(gex_chart_final.interactive(), use_container_width=True)

    except Exception as e:
        st.error(f"Error fetching data: {e}")



