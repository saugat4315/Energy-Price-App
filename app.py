"""
⚡ Norway Energy Analyzer
Power prices in the Norwegian bidding zones vs reservoirs, weather, demand/supply and
cross-border flows, with a 7-day price forecast.

Run locally:  streamlit run app.py
Deploy:       Streamlit Community Cloud (see README.md)
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import data_sources as ds
import model

st.set_page_config(page_title="Norway Energy Analyzer", page_icon="⚡", layout="wide")

TEMPLATE = "plotly_white"
RED, GREEN, BLUE, ORANGE, GREY = "#e53e3e", "#38a169", "#2b6cb0", "#dd6b20", "#a0aec0"
TODAY = pd.Timestamp.now(tz=ds.OSLO).date()
TOMORROW = TODAY + dt.timedelta(days=1)


def fig_layout(fig, height=420, **kw):
    opts = dict(template=TEMPLATE, height=height, margin=dict(t=30, r=20, b=40, l=60),
                legend=dict(orientation="h", y=-0.18), hovermode="x unified")
    opts.update(kw)   # caller's settings win
    fig.update_layout(**opts)
    return fig


def corr_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 5 or x.std() == 0:
        return np.nan, np.nan, len(x)
    r = np.corrcoef(x, y)[0, 1]
    return r, r * y.std() / x.std(), len(x)


def strength(r):
    a = abs(r)
    return "strong" if a >= .5 else "moderate" if a >= .3 else "weak" if a >= .1 else "none"


def scatter_trend(x, y, labels, xt, yt, color=BLUE):
    fig = go.Figure(go.Scatter(x=x, y=y, mode="markers", text=labels, marker=dict(size=8, color=color, opacity=.7),
                               hovertemplate="%{text}<br>%{x:.1f}<br>%{y:" + DP + "} " + U + "<extra></extra>"))
    r, slope, n = corr_slope(x, y)
    if not np.isnan(r):
        xs = np.array([np.nanmin(x), np.nanmax(x)])
        fig.add_trace(go.Scatter(x=xs, y=np.nanmean(y) + slope * (xs - np.nanmean(x)), mode="lines",
                                 line=dict(color="#4a5568", dash="dash"), hoverinfo="skip"))
    fig_layout(fig, 380, showlegend=False, xaxis_title=xt, yaxis_title=yt, hovermode="closest")
    return fig, r, slope, n


# ======================================================================= header
st.title("⚡ Norway Energy Analyzer")
st.caption("Day-ahead power prices in the Norwegian bidding zones, explained by reservoirs, weather, demand/supply and "
           "cross-border flows, with a statistical 7-day price forecast.")

tab_in, tab_res, tab_fc, tab_meth = st.tabs(["⚙️ Inputs", "📊 Results", "🔮 Forecast", "📐 Methodology (Notes & Assumptions)"])

# ======================================================================= INPUTS
with tab_in:
    st.header("1 · Price area")
    icons = {"NO1": "🏙️", "NO2": "🌊", "NO3": "⛰️", "NO4": "❄️", "NO5": "🌧️"}
    blurbs = {"NO1": "Eastern Norway, largest consumption. Linked to Sweden (SE3).",
              "NO2": "Southern Norway, cables to Germany, UK, Netherlands, Denmark.",
              "NO3": "Central Norway, linked to Sweden (SE2) and the north.",
              "NO4": "Northern Norway, large surplus, often the lowest prices.",
              "NO5": "Western Norway, big hydro reservoirs, few external links."}
    cols = st.columns(5)
    for c, (a, info) in zip(cols, ds.AREAS.items()):
        with c:
            st.markdown(f"<div style='font-size:3rem;line-height:1'>{icons[a]}</div>", unsafe_allow_html=True)
            st.markdown(f"**{a}** · {info['name']}")
            st.caption(blurbs[a])
    area = st.radio("Select the price area", list(ds.AREAS), horizontal=True,
                    format_func=lambda a: f"{icons[a]} {a}", key="area")
    A = ds.AREAS[area]

    st.divider()
    st.header("2 · Period")
    c1, c2 = st.columns([1, 2])
    with c1:
        preset = st.radio("Quick choice", ["Since 1 Aug 2026", "Last 30 days", "Last 3 months", "This year", "Custom"], key="preset")
    with c2:
        defaults = {"Since 1 Aug 2026": dt.date(2026, 8, 1), "Last 30 days": TODAY - dt.timedelta(days=30),
                    "Last 3 months": TODAY - dt.timedelta(days=91), "This year": dt.date(TODAY.year, 1, 1)}
        if preset == "Custom":
            rng = st.date_input("Select a period", (dt.date(2026, 8, 1), TOMORROW), min_value=dt.date(2015, 1, 5),
                                max_value=TOMORROW, format="DD.MM.YYYY")
            start, end = (rng if isinstance(rng, tuple) and len(rng) == 2 else (rng[0], rng[0]))
        else:
            start, end = defaults[preset], TOMORROW
            st.info(f"**{start:%d.%m.%Y} → {end:%d.%m.%Y}** ({(end - start).days + 1} days). Tomorrow's day-ahead prices "
                    "are published around 13:00 and appear automatically once available.")

    st.divider()
    st.header("3 · Parameters to analyse")
    p1, p2, p3 = st.columns(3)
    with p1:
        use_res = st.checkbox("💧 Reservoir levels (NVE)", True)
        use_weather = st.checkbox("🌦️ Weather (MET Norway)", True, disabled=not ds.secret("FROST_ID"))
        stations = {A["station"]: "Main station for " + area, "SN18700": "Oslo – Blindern", "SN4780": "Gardermoen",
                    "SN39040": "Kristiansand – Kjevik", "SN69100": "Trondheim – Værnes", "SN90450": "Tromsø", "SN50540": "Bergen – Florida"}
        station = st.selectbox("Weather station", list(dict.fromkeys(stations)), format_func=lambda s: f"{stations[s]} ({s})",
                               disabled=not use_weather)
        elements = st.multiselect("Weather parameters", list(ds.WEATHER), default=list(ds.WEATHER)[:3],
                                  format_func=lambda e: f"{ds.WEATHER[e][0]} ({ds.WEATHER[e][1]})", disabled=not use_weather)
    with p2:
        has_entsoe = bool(ds.secret("ENTSOE_TOKEN"))
        use_ds = st.checkbox("⚖️ Demand & supply (ENTSO-E)", has_entsoe, disabled=not has_entsoe)
        use_tr = st.checkbox("🔌 Cross-border flows (ENTSO-E)", has_entsoe, disabled=not has_entsoe)
        st.caption("Consumption, production by type and physical flows to neighbouring zones: " + ", ".join(ds.NEIGHBOURS[area]) + ".")
    with p3:
        has_hyd = bool(ds.secret("HYDAPI_KEY"))
        use_hyd = st.checkbox("🏞️ River discharge (NVE HydAPI)", False, disabled=not has_hyd)
        hyd_ids = []
        if use_hyd:
            try:
                hs = ds.hyd_stations(("Oslo", "Akershus", "Østfold", "Buskerud", "Innlandet", "Viken"))
                pick = st.multiselect("River stations (biggest catchments first)", hs.stationId.tolist(),
                                      default=hs.stationId.tolist()[:3],
                                      format_func=lambda i: f"{hs.set_index('stationId').at[i, 'stationName']} – {hs.set_index('stationId').at[i, 'riverName']}")
                hyd_ids = pick
            except Exception as e:
                st.warning(f"HydAPI: {e}")

    st.divider()
    st.header("4 · Forecast settings")
    f1, f2 = st.columns(2)
    with f1:
        train_years = st.slider("Training history (years)", 1, 3, 2)
    with f2:
        horizon = st.slider("Forecast horizon (days)", 3, 9, 7, help="MET's weather forecast covers about 9 days")

    st.divider()
    st.header("5 · Price unit")
    u1, u2 = st.columns([1, 2])
    with u1:
        unit_choice = st.radio("Show prices in", ["EUR/MWh (market unit)", "NOK/kWh (like your power bill)"], key="unit")
    with u2:
        vat = st.checkbox("Include 25 % VAT", False, help="VAT is not charged in Northern Norway (NO4). Grid tariffs and supplier markups are never included.")
        st.caption("NOK/kWh uses Norges Bank's official EUR→NOK rate for each day (weekends use the last business day). "
                   "1 EUR/MWh = rate ÷ 1000 NOK/kWh.")

    st.divider()
    st.header("6 · Data sources")
    status = pd.DataFrame([
        ["Day-ahead prices", "ENTSO-E (key) → falls back to hvakosterstrommen.no (free)", "✅" if has_entsoe else "⚠️ free source only"],
        ["Reservoirs", "NVE magasinstatistikk (free)", "✅"],
        ["Weather history", "MET Frost (client ID)", "✅" if ds.secret("FROST_ID") else "❌ add FROST_ID"],
        ["Weather forecast & nowcast", "MET Locationforecast (free)", "✅"],
        ["Demand, supply, flows", "ENTSO-E (key)", "✅" if has_entsoe else "❌ add ENTSOE_TOKEN"],
        ["River discharge", "NVE HydAPI (key)", "✅" if has_hyd else "❌ add HYDAPI_KEY"],
        ["EUR→NOK exchange rate", "Norges Bank (free)", "✅"],
    ], columns=["Data", "Source", "Status"])
    st.dataframe(status, hide_index=True, width="stretch")


# ======================================================================= units
NOK = unit_choice.startswith("NOK")
VAT = 1.25 if (vat and area != "NO4") else 1.0
U = ("NOK/kWh" if NOK else "€/MWh") + (" incl. VAT" if VAT > 1 else "")
DP = ".3f" if NOK else ".1f"      # decimals for prices
DS = "+.3f" if NOK else "+.1f"   # signed
DSL = "+.4f" if NOK else "+.2f"  # slopes


def conv(s):
    """Convert an EUR/MWh price series (hourly or daily index) to the chosen unit."""
    if s is None or len(s) == 0:
        return s
    if not NOK:
        return s * VAT
    idx = s.index.tz_localize(None) if getattr(s.index, "tz", None) is not None else s.index
    days = idx.normalize()
    try:
        rate = ds.eur_nok(days.min().date(), days.max().date())
        r = rate.reindex(days).ffill().bfill().values
    except Exception as e:
        st.toast(f"Norges Bank rate unavailable ({e}); using 11.7 NOK/EUR")
        r = 11.7
    return s * r / 1000 * VAT


# ======================================================================= load data
def load_all():
    out = {"area": area, "start": start, "end": end}
    out["price"], out["price_src"] = ds.prices_any(area, start, end)
    if use_res:
        try:
            out["res"] = ds.reservoirs(int(area[-1]))
        except Exception as e:
            st.warning(f"NVE reservoirs: {e}")
    if use_weather and elements:
        try:
            out["wx"] = ds.weather_history(station, tuple(elements), start, min(end, TODAY))
            out["wx_info"] = ds.station_info(station)
        except Exception as e:
            st.warning(f"Frost: {e}")
    if use_ds:
        try:
            out["load"] = ds.load(area, start, min(end, TODAY))
            out["gen"] = ds.generation(area, start, min(end, TODAY))
        except Exception as e:
            st.warning(f"ENTSO-E demand/supply: {e}")
    if use_tr:
        try:
            out["flows"] = ds.flows(area, start, min(end, TODAY))
        except Exception as e:
            st.warning(f"ENTSO-E flows: {e}")
    if use_hyd and hyd_ids:
        try:
            out["hyd"] = ds.discharge(tuple(hyd_ids), start, min(end, TODAY))
        except Exception as e:
            st.warning(f"HydAPI: {e}")
    return out


# ======================================================================= RESULTS
with tab_res:
    st.markdown(f"**{icons[area]} {area} · {A['name']}** · {start:%d.%m.%Y} – {end:%d.%m.%Y}")
    if st.button("▶ Run analysis", type="primary"):
        st.session_state["data"] = load_all()
    D = st.session_state.get("data")
    if not D:
        st.info("Choose your settings in **⚙️ Inputs**, then click **Run analysis**.")
    elif D["area"] != area or D["start"] != start or D["end"] != end:
        st.warning("Inputs changed. Click **Run analysis** to update.")
    if D:
        P = conv(D["price"])
        if P is None or P.empty:
            st.error("No price data returned for this period.")
            st.stop()
        Pd = ds.daily(P)
        r_pr, r_res, r_wx, r_ds, r_tr, r_hyd = st.tabs(["💶 Prices", "💧 Reservoirs", "🌦️ Weather", "⚖️ Demand & supply", "🔌 Transmission", "🏞️ Rivers"])

        # ---------------- prices
        with r_pr:
            last7 = P[P.index >= P.index.max() - pd.Timedelta(days=7)]
            has_tm = (P.index.date == TOMORROW).any()
            m = st.columns(4)
            m[0].metric("Average price", f"{P.mean():{DP}} {U}", help="Mean of hourly day-ahead prices, excl. VAT")
            m[1].metric("Last 7 days", f"{last7.mean():{DP}} {U}", f"{last7.mean() - P.mean():{DS}} vs period", delta_color="inverse")
            m[2].metric("Lowest hour", f"{P.min():{DP}} {U}", P.idxmin().strftime("%d.%m %H:00"), delta_color="off")
            m[3].metric("Highest hour", f"{P.max():{DP}} {U}", P.idxmax().strftime("%d.%m %H:00"), delta_color="off")
            m = st.columns(4)
            m[0].metric("Volatility (std)", f"{P.std():{DP}} {U}")
            m[1].metric("Negative-price hours", int((P < 0).sum()))
            m[2].metric("Daily spread (avg max−min)", f"{(P.resample('D').max() - P.resample('D').min()).mean():{DP}} {U}")
            m[3].metric("Tomorrow average", f"{P[P.index.date == TOMORROW].mean():{DP}} {U}" if has_tm else "after ~13:00")
            st.caption(f"Source: {D['price_src']}. 15-minute prices (since Oct 2025) are averaged to hourly.")

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Average day: when is power cheap or expensive?")
                h = P.groupby(P.index.hour)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(24)), y=h.quantile(.9), line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=list(range(24)), y=h.quantile(.1), fill="tonexty", fillcolor="rgba(43,108,176,.12)",
                                         line=dict(width=0), name="10–90 % range"))
                dev = h.mean() - P.mean()
                fig.add_trace(go.Bar(x=list(range(24)), y=dev, name="vs period average", yaxis="y2",
                                     marker_color=[RED if v > 0 else GREEN for v in dev], opacity=.55))
                fig.add_trace(go.Scatter(x=list(range(24)), y=h.mean(), name="Average price", line=dict(color="#2d3748", width=2.5), mode="lines+markers"))
                fig_layout(fig, 420, xaxis=dict(title="Hour", tickmode="array", tickvals=list(range(0, 24, 2)), ticktext=[f"{i:02d}:00" for i in range(0, 24, 2)]),
                           yaxis_title=f"{U}", yaxis2=dict(overlaying="y", side="right", title=f"{U} vs average", showgrid=False))
                st.plotly_chart(fig, width="stretch")
            with c2:
                st.subheader("Hourly price (red = expensive, green = cheap)")
                hm = pd.DataFrame({"p": P.values}, index=P.index.tz_localize(None))
                hm = hm.pivot_table(index=hm.index.normalize(), columns=hm.index.hour, values="p")
                fig = go.Figure(go.Heatmap(z=hm.values, x=[f"{h:02d}:00" for h in hm.columns], y=[d.strftime("%d.%m.%Y") for d in hm.index],
                                           colorscale="RdYlGn_r", colorbar=dict(title=f"{U}")))
                fig_layout(fig, max(420, 14 * len(hm)), yaxis=dict(autorange="reversed"), hovermode="closest")
                st.plotly_chart(fig, width="stretch")
            st.subheader("Daily average and 7-day trend")
            fig = go.Figure([go.Bar(x=Pd.index, y=Pd.values, name="Daily average", marker_color="#90cdf4"),
                             go.Scatter(x=Pd.index, y=Pd.rolling(7, min_periods=3).mean(), name="7-day average", line=dict(color=BLUE, width=3))])
            st.plotly_chart(fig_layout(fig, 360, yaxis_title=f"{U}"), width="stretch")

        # ---------------- reservoirs
        with r_res:
            R = D.get("res")
            if R is None:
                st.info("Turn on reservoir levels in **Inputs**.")
            else:
                Rp = R[(R.index >= pd.Timestamp(start) - pd.Timedelta(days=7)) & (R.index <= pd.Timestamp(end))]
                latest = R.dropna(subset=["fyllingsgrad"]).iloc[-1]
                m = st.columns(4)
                m[0].metric(f"Filling, week {int(latest.iso_uke)}", f"{latest.fyllingsgrad:.1f} %", f"{latest.deviation:+.1f} pp vs median")
                m[1].metric("Change since last week", f"{latest.endring_fyllingsgrad * 100:+.1f} pp" if "endring_fyllingsgrad" in latest else "–")
                m[2].metric("Stored energy", f"{latest.fylling_TWh:.1f} TWh", f"of {latest.kapasitet_TWh:.1f} TWh", delta_color="off")
                m[3].metric("20-year range this week", f"{latest['min']:.0f}–{latest['max']:.0f} %")
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=Rp.index, y=Rp["max"], line=dict(width=0), showlegend=False, hoverinfo="skip"), secondary_y=True)
                fig.add_trace(go.Scatter(x=Rp.index, y=Rp["min"], fill="tonexty", fillcolor="rgba(229,62,62,.07)", line=dict(width=0), name="20-year min–max"), secondary_y=True)
                fig.add_trace(go.Scatter(x=Rp.index, y=Rp["median"], name="Median", line=dict(color=ORANGE, dash="dash")), secondary_y=True)
                fig.add_trace(go.Scatter(x=Rp.index, y=Rp.fyllingsgrad, name="Filling", line=dict(color=GREEN, width=3)), secondary_y=True)
                fig.add_trace(go.Scatter(x=Pd.index, y=Pd.values, name="Daily price", line=dict(color=BLUE, width=1.5)), secondary_y=False)
                fig.update_yaxes(title_text=f"{U}", secondary_y=False)
                fig.update_yaxes(title_text="Reservoir filling (%)", range=[0, 100], secondary_y=True, showgrid=False)
                st.plotly_chart(fig_layout(fig, 460), width="stretch")
                wk = Pd.resample("W-MON", label="left", closed="left").mean()
                j = Rp[["deviation"]].join(wk.rename("price"), how="inner").dropna()
                c1, c2 = st.columns(2)
                with c1:
                    fig, r, slope, n = scatter_trend(j.deviation, j.price, j.index.strftime("Week of %d.%m"), "Filling minus median (pp)", f"Weekly avg price ({U})", GREEN)
                    st.plotly_chart(fig, width="stretch")
                    if not np.isnan(r):
                        st.caption(f"Correlation **{r:.2f}** ({strength(r)}): {slope:{DSL}} {U} per percentage point, {n} weeks. Negative = fuller reservoirs, lower prices.")
                with c2:
                    fig = go.Figure(go.Bar(x=Rp.index, y=Rp.deviation, marker_color=[GREEN if v >= 0 else RED for v in Rp.deviation]))
                    st.plotly_chart(fig_layout(fig, 380, yaxis_title="pp vs median", title="Deviation from the 20-year median"), width="stretch")

        # ---------------- weather
        with r_wx:
            W = D.get("wx")
            if W is None or W.empty:
                st.info("Turn on weather in **Inputs** (needs FROST_ID).")
            else:
                info = D.get("wx_info", {})
                st.markdown(f"📍 **{info.get('name', station)}** · altitude {info.get('masl', '–')} m · "
                            f"{info.get('municipality', '')}")
                try:
                    nc = ds._get("https://api.met.no/weatherapi/nowcast/2.0/complete", {"lat": A["lat"], "lon": A["lon"]}).json()
                    det = nc["properties"]["timeseries"][0]["data"]["instant"]["details"]
                    st.caption(f"Right now near {A['name'].split('(')[-1].rstrip(')')}: {det.get('air_temperature', float('nan')):.1f} °C · "
                               f"rain {det.get('precipitation_rate', 0):.1f} mm/h · wind {det.get('wind_speed', float('nan')):.1f} m/s (MET Nowcast)")
                except Exception:
                    pass
                j = W.join(Pd.rename("price"), how="inner")
                rows = []
                for el in W.columns:
                    r, slope, n = corr_slope(j[el], j.price - j.price.mean())
                    rows.append({"Parameter": ds.WEATHER[el][0], "Correlation": r, "Strength": strength(r) if not np.isnan(r) else "–",
                                 "Price change per unit": f"{slope:{DSL}} {U} per {ds.WEATHER[el][1]}" if not np.isnan(slope) else "–", "Days": n})
                st.subheader("Which weather parameter moves with the price?")
                st.dataframe(pd.DataFrame(rows).sort_values("Correlation", key=lambda s: -s.abs()), hide_index=True, width="stretch",
                             column_config={"Correlation": st.column_config.ProgressColumn(min_value=-1, max_value=1, format="%.2f")})
                st.caption("Correlation shows days moving together, not cause and effect (cold days are often also calm days).")
                el = st.selectbox("Show parameter", list(W.columns), format_func=lambda e: ds.WEATHER[e][0])
                lab, unit = ds.WEATHER[el]
                c1, c2 = st.columns(2)
                with c1:
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    tr = go.Bar if "precip" in el else go.Scatter
                    fig.add_trace(tr(x=W.index, y=W[el], name=f"{lab} ({unit})", marker_color="#319795", **({} if tr is go.Bar else {"line": dict(color="#319795")})), secondary_y=False)
                    fig.add_trace(go.Scatter(x=Pd.index, y=Pd.values, name="Price", line=dict(color=BLUE)), secondary_y=True)
                    fig.update_yaxes(title_text=f"{lab} ({unit})", secondary_y=False)
                    fig.update_yaxes(title_text=f"{U}", secondary_y=True, showgrid=False)
                    st.plotly_chart(fig_layout(fig, 400), width="stretch")
                with c2:
                    dev = j.price - j.price.mean()
                    fig, r, slope, n = scatter_trend(j[el], dev, j.index.strftime("%d.%m.%Y"), f"{lab} ({unit})", f"Price − period average ({U})", "#805ad5")
                    st.plotly_chart(fig, width="stretch")

        # ---------------- demand & supply
        with r_ds:
            L, G = D.get("load"), D.get("gen")
            if L is None or G is None or (L.empty and G.empty):
                st.info("Turn on demand & supply in **Inputs** (needs ENTSO-E token).")
            else:
                Ld, Gd = ds.daily(L), ds.daily(G)
                gsum = Gd.sum(axis=1)
                m = st.columns(4)
                m[0].metric("Avg consumption", f"{Ld.mean():,.0f} MW")
                m[1].metric("Avg production", f"{gsum.mean():,.0f} MW",
                            f"{'net importer' if Ld.mean() > gsum.mean() else 'net exporter'} {abs(Ld.mean() - gsum.mean()):,.0f} MW", delta_color="off")
                hydro = Gd[[c for c in Gd if c in ("B10", "B11", "B12")]].sum(axis=1).sum() / gsum.sum() * 100
                m[2].metric("Hydro share", f"{hydro:.0f} %")
                m[3].metric("Peak hour consumption", f"{L.max():,.0f} MW", L.idxmax().strftime("%d.%m %H:00"), delta_color="off")
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                for c in sorted(Gd, key=lambda c: list(ds.PSR).index(c) if c in ds.PSR else 99):
                    lab, col = ds.PSR.get(c, (c, GREY))
                    fig.add_trace(go.Scatter(x=Gd.index, y=Gd[c], name=lab, stackgroup="g", line=dict(width=0), fillcolor=col), secondary_y=False)
                fig.add_trace(go.Scatter(x=Ld.index, y=Ld.values, name="Consumption", line=dict(color="#1a202c", width=2.5)), secondary_y=False)
                fig.add_trace(go.Scatter(x=Pd.index, y=Pd.values, name="Price", line=dict(color=RED, width=2)), secondary_y=True)
                fig.update_yaxes(title_text="MW (daily average)", secondary_y=False)
                fig.update_yaxes(title_text=f"{U}", secondary_y=True, showgrid=False)
                st.plotly_chart(fig_layout(fig, 460, title="Production by type vs consumption"), width="stretch")
                c1, c2 = st.columns(2)
                with c1:
                    gap = (Ld - gsum).rename("gap")
                    j = gap.to_frame().join(Pd.rename("price"), how="inner").dropna()
                    fig, r, slope, n = scatter_trend(j.gap, j.price, j.index.strftime("%d.%m.%Y"), "Consumption − production (MW)", f"{U}")
                    st.plotly_chart(fig, width="stretch")
                    if not np.isnan(r):
                        st.caption(f"Correlation **{r:.2f}** ({strength(r)}): +{slope * 1000:{DP}} {U} per 1,000 MW more import need.")
                with c2:
                    tot = G.mean()
                    fig = go.Figure(go.Pie(labels=[ds.PSR.get(c, (c,))[0] for c in tot.index], values=tot.values, hole=.45,
                                           marker_colors=[ds.PSR.get(c, (c, GREY))[1] for c in tot.index], sort=False))
                    st.plotly_chart(fig_layout(fig, 380, title="Production mix", showlegend=False), width="stretch")

        # ---------------- transmission
        with r_tr:
            Fl = D.get("flows")
            if Fl is None or Fl.empty:
                st.info("Turn on cross-border flows in **Inputs** (needs ENTSO-E token).")
            else:
                Fd = ds.daily(Fl)
                net = Fd.sum(axis=1)
                m = st.columns(3)
                m[0].metric(f"Avg net import into {area}", f"{net.mean():+,.0f} MW", "importing" if net.mean() > 0 else "exporting", delta_color="off")
                busiest = Fl.mean().abs().idxmax()
                m[1].metric("Busiest border (net)", f"{busiest} ⇄ {area}", f"{Fl[busiest].mean():+,.0f} MW avg", delta_color="off")
                m[2].metric("Hours with net import", f"{(Fl.sum(axis=1) > 0).mean() * 100:.0f} %")
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                for i, c in enumerate(Fd):
                    fig.add_trace(go.Bar(x=Fd.index, y=Fd[c], name=f"{c} ⇄ {area}"), secondary_y=False)
                fig.add_trace(go.Scatter(x=net.index, y=net, name="Net import", line=dict(color="#1a202c", width=2.5)), secondary_y=False)
                fig.add_trace(go.Scatter(x=Pd.index, y=Pd.values, name="Price", line=dict(color=RED, width=2)), secondary_y=True)
                fig.update_layout(barmode="relative")
                fig.update_yaxes(title_text=f"MW (+ into {area})", secondary_y=False)
                fig.update_yaxes(title_text=f"{U}", secondary_y=True, showgrid=False)
                st.plotly_chart(fig_layout(fig, 460), width="stretch")
                c1, c2 = st.columns(2)
                with c1:
                    j = net.rename("net").to_frame().join(Pd.rename("price"), how="inner").dropna()
                    fig, r, slope, n = scatter_trend(j.net, j.price, j.index.strftime("%d.%m.%Y"), f"Net import into {area} (MW)", f"{U}", "#805ad5")
                    st.plotly_chart(fig, width="stretch")
                with c2:
                    tab = pd.DataFrame({"Avg in (MW)": Fl.clip(lower=0).mean(), "Avg out (MW)": -Fl.clip(upper=0).mean(), "Net (MW)": Fl.mean(),
                                        "Peak in (MW)": Fl.max(), "Peak out (MW)": -Fl.min()}).round(0)
                    tab.index = [f"{c} ⇄ {area}" for c in tab.index]
                    st.dataframe(tab, width="stretch")
                    st.caption("Physical flows. A border running near its usual peak is a sign of congestion (Nordic flow-based market since Oct 2024).")

        # ---------------- rivers
        with r_hyd:
            H = D.get("hyd")
            if H is None or H.empty:
                st.info("Turn on river discharge in **Inputs** (needs HYDAPI_KEY).")
            else:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                for c in H:
                    fig.add_trace(go.Scatter(x=H.index, y=H[c], name=c), secondary_y=False)
                fig.add_trace(go.Scatter(x=Pd.index, y=Pd.values, name="Price", line=dict(color=GREY, dash="dot")), secondary_y=True)
                fig.update_yaxes(title_text="Discharge (m³/s)", secondary_y=False)
                fig.update_yaxes(title_text=f"{U}", secondary_y=True, showgrid=False)
                st.plotly_chart(fig_layout(fig, 440), width="stretch")

# ======================================================================= FORECAST
with tab_fc:
    st.markdown(f"**{icons[area]} {area}** · trained on {train_years} year(s) · horizon {horizon} days · weather from {A['station']}")
    if st.button("🔮 Run forecast", type="primary"):
        with st.status("Building the forecast…", expanded=True) as stat:
            t0 = TODAY - dt.timedelta(days=365 * train_years + 35)
            st.write("Prices…")
            hp, src = ds.prices_any(area, t0, TOMORROW)
            hp = conv(hp)   # train directly in the chosen unit
            d = ds.daily(hp).rename("price").to_frame()
            if ds.secret("FROST_ID"):
                st.write("Weather history (Frost) and forecast (MET)…")
                try:
                    wh = ds.weather_history(A["station"], ("mean(air_temperature P1D)", "mean(wind_speed P1D)", "sum(precipitation_amount P1D)"), t0, TODAY)
                    wh = wh.rename(columns={"mean(air_temperature P1D)": "temp", "mean(wind_speed P1D)": "wind", "sum(precipitation_amount P1D)": "precip"})
                    d = d.join(wh, how="outer")
                except Exception as e:
                    st.write(f"⚠️ Frost: {e}")
            try:
                wf = ds.weather_forecast(A["lat"], A["lon"], A["alt"])
                d = d.reindex(d.index.union(wf.index))
                for c in ["temp", "wind", "precip"]:
                    if c not in d:
                        d[c] = np.nan
                    # past days: keep observations, fill gaps from the forecast; today onwards: forecast
                    d[c] = d[c].combine_first(wf[c])
                    fut = wf.index[wf.index >= pd.Timestamp(TODAY)]
                    d.loc[fut, c] = wf.loc[fut, c]
            except Exception as e:
                st.write(f"⚠️ MET forecast: {e}")
            st.write("Reservoirs (NVE)…")
            try:
                R = ds.reservoirs(int(area[-1]))
                d["reservoir"] = R["deviation"].reindex(d.index.union(R.index)).ffill().reindex(d.index)
            except Exception as e:
                st.write(f"⚠️ NVE: {e}")
            if ds.secret("ENTSOE_TOKEN"):
                st.write("Week-ahead consumption forecast (ENTSO-E)…")
                try:
                    wl = ds.daily(ds.load(area, t0, TODAY + dt.timedelta(days=horizon + 2), "weekahead"))
                    d["load"] = wl.reindex(d.index)
                except Exception as e:
                    st.write(f"⚠️ ENTSO-E load forecast: {e}")
            st.write("Training…")
            try:
                st.session_state["fc"] = {"res": model.forecast(d, hp, horizon), "area": area, "src": src, "hist": d["price"], "unit": U}
                stat.update(label="Forecast ready", state="complete", expanded=False)
            except Exception as e:
                stat.update(label=f"Forecast failed: {e}", state="error")

    FC = st.session_state.get("fc")
    if not FC:
        st.info("Click **Run forecast**. It downloads up to 3 years of history, so the first run takes about a minute.")
    else:
        res, hist = FC["res"], FC["hist"]
        T, acc = res["table"], res["accuracy"]
        if FC["area"] != area or FC.get("unit") != U:
            st.warning(f"This forecast is for {FC['area']} in {FC.get('unit')}. Click **Run forecast** to update it for {area} in {U}.")
        last7 = hist.dropna().iloc[-7:].mean()
        m = st.columns(4)
        m[0].metric(f"Forecast average, next {len(T)} days", f"{T.forecast.mean():{DP}} {U}",
                    f"{(T.forecast.mean() / last7 - 1) * 100:+.1f} % vs last 7 days", delta_color="inverse")
        m[1].metric("Highest day", f"{T.forecast.max():{DP}} {U}", T.forecast.idxmax().strftime("%a %d.%m"), delta_color="off")
        m[2].metric("Lowest day", f"{T.forecast.min():{DP}} {U}", T.forecast.idxmin().strftime("%a %d.%m"), delta_color="off")
        m[3].metric("Typical error 1 / 7 days ahead", f"±{acc.mae.iloc[0]:{DP}} / ±{acc.mae.iloc[-1]:{DP}}",
                    help="Average absolute error on the last 20 % of history, which the model did not see while fitting")

        c1, c2 = st.columns([3, 2])
        with c1:
            st.subheader("Daily average: actual and forecast")
            h = hist.dropna().iloc[-45:]
            lk = res["last_known"]
            fig = go.Figure([
                go.Scatter(x=h.index, y=h.values, name="Actual", line=dict(color=BLUE, width=2.5)),
                go.Scatter(x=list(T.index) + list(T.index[::-1]), y=list(T.high) + list(T.low[::-1]), fill="toself",
                           fillcolor="rgba(221,107,32,.15)", line=dict(width=0), name="80 % range", hoverinfo="skip"),
                go.Scatter(x=[lk] + list(T.index), y=[hist.get(lk)] + list(T.forecast), name="Forecast", mode="lines+markers",
                           line=dict(color=ORANGE, width=3, dash="dot"))])
            fig.add_vline(x=lk, line_dash="dash", line_color=GREY)
            st.plotly_chart(fig_layout(fig, 420, yaxis_title=f"{U}"), width="stretch")
        with c2:
            st.subheader("Hourly forecast")
            Hh = res["hourly"]
            hm = Hh.pivot_table(index=Hh.index.normalize(), columns=Hh.index.hour, values="price")
            fig = go.Figure(go.Heatmap(z=hm.values, x=[f"{h:02d}:00" for h in hm.columns], y=[d.strftime("%a %d.%m") for d in hm.index],
                                       colorscale="RdYlGn_r", colorbar=dict(title=f"{U}")))
            st.plotly_chart(fig_layout(fig, 420, yaxis=dict(autorange="reversed"), hovermode="closest"), width="stretch")
            st.caption("Daily forecast + the typical hourly pattern of the last 4 weeks (weekday/weekend).")

        show = T.copy()
        show.index = show.index.strftime("%a %d.%m.%Y")
        show = show.rename(columns={"forecast": f"Forecast {U}", "low": "Low (80 %)", "high": "High (80 %)", "temp": "Temp °C",
                                    "wind": "Wind m/s", "precip": "Rain mm", "load": "Consumption fc MW", "reservoir": "Reservoir vs median pp"})
        price_cols = [c for c in show.columns if U in c or c.startswith(("Low", "High"))]
        st.dataframe(show.round(1).assign(**{c: show[c].round(3 if NOK else 1) for c in price_cols}), width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("What drives the forecast")
            I = res["importance"].copy()
            I.index = [model.FEATURE_LABELS.get(i, i) for i in I.index]
            I = I.reindex(I.abs().sum(axis=1).sort_values().index)
            fig = go.Figure([go.Bar(y=I.index, x=I[h], orientation="h", name=f"{h} day{'s' if h > 1 else ''} ahead") for h in I.columns])
            st.plotly_chart(fig_layout(fig, 420, barmode="group", xaxis_title=f"{U} per typical change", hovermode="closest",
                                       margin=dict(l=170, t=20, r=20, b=40)), width="stretch")
            if res["dropped"]:
                st.caption("Not used (too little data): " + ", ".join(model.FEATURE_LABELS.get(c, c) for c in res["dropped"]))
        with c2:
            st.subheader("How accurate is it?")
            fig = go.Figure([go.Bar(x=[f"{h} d" for h in acc.index], y=acc.naive, name="Simple guess: same as last price", marker_color="#cbd5e0"),
                             go.Bar(x=[f"{h} d" for h in acc.index], y=acc.mae, name="This model", marker_color=ORANGE)])
            st.plotly_chart(fig_layout(fig, 420, barmode="group", yaxis_title=f"Average error ({U})", hovermode="closest"), width="stretch")
            st.caption(f"Beats the simple guess on {(acc.mae < acc.naive).sum()} of {len(acc)} horizons · trained on {acc.n_train.iloc[0]} days · prices from {FC['src']}.")
        if res["backtest"] is not None:
            st.subheader("Back-test: 3 days ahead vs what happened")
            b = res["backtest"]
            fig = go.Figure([go.Scatter(x=b.index, y=b.actual, name="Actual", line=dict(color=BLUE)),
                             go.Scatter(x=b.index, y=b.model, name="Model (3 days ahead)", line=dict(color=ORANGE, dash="dot"))])
            st.plotly_chart(fig_layout(fig, 340, yaxis_title=f"{U}"), width="stretch")

# ======================================================================= METHODOLOGY
with tab_meth:
    st.markdown("""
### Data
| Topic | Source | Resolution |
|---|---|---|
| Day-ahead prices | ENTSO-E Transparency Platform (A44); fallback hvakosterstrommen.no | 15 min → averaged to hourly (Oslo time) |
| Reservoir filling | NVE magasinstatistikk, incl. 20-year min/median/max per week | weekly |
| Weather history | MET Norway Frost: daily mean temperature, wind, pressure, cloud cover; daily precipitation sum; max gust; snow depth | daily |
| Weather forecast / nowcast | MET Locationforecast 2.0 / Nowcast 2.0 | hourly → daily |
| Consumption, production by type, flows | ENTSO-E (A65, A75, A11) | hourly → daily |
| River discharge | NVE HydAPI, parameter 1001 | daily |

Prices are excluding VAT and grid tariffs. Weather is measured at one station per area, which represents the area only roughly.

### Analysis
* **Correlation** (Pearson *r*) compares daily values; the *slope* converts it to €/MWh per unit.
  Correlation is not causation: cold, calm, dry periods tend to coincide and all raise prices.
* **Reservoir deviation** = filling minus the 20-year median for the same week, in percentage points.
* **Net import** = sum of physical flows into the area minus flows out of it.

### Forecast model
* One **ridge regression** per horizon *h* = 1…H days (a "direct" multi-step forecast). Target: daily average day-ahead price.
* Inputs known at forecast time: last known price, 7- and 28-day average price, reservoir deviation, rain over the last 14 days.
  Inputs for the target day: MET weather forecast (temperature, wind, precipitation), ENTSO-E week-ahead consumption forecast, weekday/weekend, season.
* Inputs are standardised; the ridge penalty (λ ∈ {0.3, 3, 30, 100}) is chosen on a time-ordered hold-out (last 20 % of days), then the model is refitted on all days.
* **Uncertainty band**: ±1.28 × RMSE on the hold-out → the range that contained the actual price about 80 % of the time.
* **Hourly profile**: daily forecast + average hourly deviation from the daily mean over the last 28 days, separately for weekdays and weekends.
* **Benchmark**: "same as the last known price" (persistence). A useful model must beat it.

### Limitations
* No information about outages, cable failures, fuel and CO₂ prices in neighbouring markets, or hydropower producers' water values.
* Weather forecasts lose skill after 3–5 days, so errors grow with the horizon.
* During the training period, observed weather stands in for the weather forecast, which makes the model look slightly better than it is in real use.
""")
