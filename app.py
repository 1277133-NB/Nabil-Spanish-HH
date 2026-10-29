import io
import json
import math
import re
import textwrap
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="HHM | Historical Data Polynomial Explorer", page_icon="📈", layout="wide")
API = "https://api.worldbank.org/v2"
UNITS = {
    "SP.POP.TOTL": "people",
    "NY.GDP.MKTP.CD": "current US dollars",
    "NY.GDP.PCAP.CD": "current US dollars per person",
    "EN.ATM.CO2E.PC": "metric tons of CO₂ per person",
    "SP.DYN.LE00.IN": "years",
    "FP.CPI.TOTL.ZG": "annual percent",
    "SL.UEM.TOTL.ZS": "percent of labor force",
    "EG.USE.PCAP.KG.OE": "kg of oil equivalent per person",
}
INDICATORS = {
    "Population, total": "SP.POP.TOTL",
    "GDP (current US$)": "NY.GDP.MKTP.CD",
    "GDP per capita (current US$)": "NY.GDP.PCAP.CD",
    "CO₂ emissions per capita": "EN.ATM.CO2E.PC",
    "Life expectancy at birth": "SP.DYN.LE00.IN",
    "Inflation, consumer prices (annual %)": "FP.CPI.TOTL.ZG",
    "Unemployment (% of labor force)": "SL.UEM.TOTL.ZS",
    "Energy use per capita": "EG.USE.PCAP.KG.OE",
}
LEVELS = {
    "Algebra 2": "Polynomial basics, domain/range, end behavior, trends, average rates of change.",
    "Precalculus": "Adds degree/coefficient effects, turning points, and concavity from changing average rates.",
    "Calculus": "Adds first/second derivatives, tangent/secant lines, critical points, extrema, and inflection checks.",
}

@st.cache_data(ttl=3600, show_spinner=False)
def get_countries():
    r = requests.get(f"{API}/country", params={"format": "json", "per_page": 400}, timeout=20)
    r.raise_for_status()
    data = r.json()[1]
    return {x["name"]: x["id"] for x in data if x.get("region", {}).get("value") != "Aggregates"}

@st.cache_data(ttl=3600, show_spinner=False)
def get_indicator_data(country_code, indicator_code):
    r = requests.get(
        f"{API}/country/{country_code}/indicator/{indicator_code}",
        params={"format": "json", "per_page": 20000, "date": "1900:2100"},
        timeout=30,
    )
    r.raise_for_status()
    payload = r.json()
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        return pd.DataFrame(columns=["year", "value"])
    rows = []
    for item in payload[1]:
        try:
            year = int(item["date"])
            value = item.get("value")
            if value is not None and np.isfinite(float(value)):
                rows.append({"year": year, "value": float(value)})
        except (TypeError, ValueError, KeyError):
            pass
    return pd.DataFrame(rows).drop_duplicates("year").sort_values("year").reset_index(drop=True)

def fit_poly(data, degree):
    clean = data[["year", "value"]].dropna().drop_duplicates("year").sort_values("year")
    if len(clean) < degree + 1:
        raise ValueError(f"Degree {degree} requires at least {degree + 1} distinct years; only {len(clean)} are available.")
    base_year = int(round(float(clean["year"].mean())))
    t = clean["year"].to_numpy(dtype=float) - base_year
    y = clean["value"].to_numpy(dtype=float)
    coeff = np.polyfit(t, y, degree)
    model = np.poly1d(coeff)
    fitted = model(t)
    residual = y - fitted
    mae = float(np.mean(np.abs(residual)))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    result = clean.copy()
    result["t"] = t
    result["fitted"] = fitted
    result["residual"] = residual
    return {"base": base_year, "coeff": coeff, "model": model, "mae": mae, "r2": r2,
            "residuals": result, "n": len(clean), "min_year": int(clean.year.min()),
            "max_year": int(clean.year.max()), "mean": float(clean.value.mean()),
            "units": None}

def poly_equation(coeff, base, unit):
    terms = []
    degree = len(coeff) - 1
    for i, c in enumerate(coeff):
        power = degree - i
        if power == 0:
            term = f"{c:.10g}"
        elif power == 1:
            term = f"{c:.10g}·t"
        else:
            term = f"{c:.10g}·t^{power}"
        terms.append(term)
    return "y = " + " + ".join(terms) + f", where t = year − {base}"

def pred(model_info, years):
    years = np.asarray(years, dtype=float)
    # The same centered-time transform used in fitting is applied exactly once.
    return model_info["model"](years - model_info["base"])

def safe_float(value):
    return "—" if value is None or not np.isfinite(value) else f"{value:,.6g}"

def parse_years(raw, min_year=1950, max_year=2100):
    requested, invalid = [], []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if not re.fullmatch(r"\d{4}", part):
            invalid.append(part)
            continue
        year = int(part)
        if not min_year <= year <= max_year:
            invalid.append(part)
        else:
            requested.append(year)
    return sorted(set(requested)), invalid

def make_plot(data, info, single_year=None, batch_years=None, sample_step=5, title="Historical data polynomial fit"):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = data.year.to_numpy(dtype=float)
    y = data.value.to_numpy(dtype=float)
    ax.scatter(x, y, label="Observed data", s=22, alpha=.8)
    smooth_x = np.linspace(info["min_year"], info["max_year"], 500)
    ax.plot(smooth_x, pred(info, smooth_x), label="Fitted curve (historical range)", linewidth=2)
    ax.axvline(info["min_year"], linestyle=":", linewidth=1, label="Historical boundary")
    ax.axvline(info["max_year"], linestyle=":", linewidth=1)
    # Sample markers make the sample increment control visible without replacing the smooth curve.
    sampled = data.iloc[::max(1, int(sample_step))]
    ax.scatter(sampled.year, pred(info, sampled.year), marker="x", s=45, label=f"Model sample markers (every {sample_step} row(s))")
    if single_year is not None:
        ax.scatter([single_year], pred(info, [single_year]), marker="*", s=150, label=f"Single-year estimate ({single_year})")
    if batch_years:
        ax.scatter(batch_years, pred(info, batch_years), marker="D", s=40, label="Batch estimates")
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(info.get("units") or "Indicator value")
    ax.grid(True, alpha=.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig

def excel_bytes(summary, data, results):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        data.to_excel(writer, sheet_name="Working data", index=False)
        results.to_excel(writer, sheet_name="Calculated years", index=False)
    out.seek(0)
    return out.getvalue()

def pdf_bytes(summary_text, fig, preview, results):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=landscape(letter), rightMargin=0.45*inch, leftMargin=0.45*inch,
                            topMargin=0.45*inch, bottomMargin=0.45*inch)
    styles = getSampleStyleSheet()
    story = [Paragraph("HHM Historical Data Polynomial Explorer", styles["Title"]), Spacer(1, 8)]
    for line in summary_text.split("\n"):
        if line.strip():
            story.append(Paragraph(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), styles["BodyText"]))
            story.append(Spacer(1, 4))
    img = io.BytesIO()
    fig.savefig(img, format="png", dpi=150, bbox_inches="tight")
    img.seek(0)
    story += [Spacer(1, 8), Image(img, width=9.2*inch, height=4.4*inch), PageBreak()]
    story.append(Paragraph("Data preview", styles["Heading2"]))
    p = preview.head(18).copy()
    table_data = [list(p.columns)] + p.astype(str).values.tolist()
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),.25,colors.grey),
                               ("FONTSIZE",(0,0),(-1,-1),7),("VALIGN",(0,0),(-1,-1),"TOP")]))
    story.append(table)
    if len(results):
        story += [Spacer(1, 12), Paragraph("Calculated year results", styles["Heading2"])]
        table_data = [list(results.columns)] + results.astype(str).values.tolist()
        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),.25,colors.grey),
                                   ("FONTSIZE",(0,0),(-1,-1),7),("VALIGN",(0,0),(-1,-1),"TOP")]))
        story.append(table)
    doc.build(story)
    out.seek(0)
    return out.getvalue()

st.title("📈 HHM Historical Data Polynomial Explorer")
st.write("Explore historical country indicators with centered-time polynomial models. Source: World Bank API.")
st.info("A polynomial is a descriptive curve, not proof of causation. Extrapolations beyond observed years can become implausible quickly.")

with st.sidebar:
    st.header("1. Data and model")
    level = st.selectbox("Math level", list(LEVELS))
    st.caption(LEVELS[level])
    try:
        countries = get_countries()
        country_name = st.selectbox("Country", sorted(countries), index=(sorted(countries).index("United States") if "United States" in countries else 0))
        indicator_label = st.selectbox("Indicator", list(INDICATORS))
        indicator_code = INDICATORS[indicator_label]
        degree = st.slider("Polynomial degree", 1, 6, 2)
        sample_step = st.slider("Model sample increment (rows)", 1, 10, 5)
        refresh = st.button("Refresh source data")
        if refresh:
            get_indicator_data.clear()
    except Exception as exc:
        st.error(f"Could not load country list: {exc}")
        st.stop()

try:
    raw_data = get_indicator_data(countries[country_name], indicator_code)
except Exception as exc:
    st.error(f"World Bank API request failed: {exc}. Check your internet connection and the API response, then retry.")
    st.stop()

if raw_data.empty:
    st.error("The source returned no non-missing observations for this country and indicator.")
    st.stop()

default_units = UNITS.get(indicator_code, indicator_label)
st.caption(f"Source coverage returned: {raw_data.year.min()}–{raw_data.year.max()} | {len(raw_data)} non-missing observations | Units: {default_units}")

with st.expander("Edit or remove data before fitting", expanded=False):
    st.write("Edits apply to this session's working copy only. Reset restores the source data. Duplicate years are reduced to the last edited row when fitting.")
    edited = st.data_editor(raw_data, num_rows="dynamic", use_container_width=True, key="data_editor")
    col_a, col_b = st.columns(2)
    with col_a:
        use_edits = st.checkbox("Use edited working data", value=False)
    with col_b:
        if st.button("Reset edited data"):
            st.session_state.pop("data_editor", None)
            st.session_state["use_edits_reset"] = True
            st.rerun()
else:
    edited = raw_data.copy()
    use_edits = False

if st.session_state.pop("use_edits_reset", False):
    use_edits = False
working = edited.copy() if use_edits else raw_data.copy()
working["year"] = pd.to_numeric(working["year"], errors="coerce")
working["value"] = pd.to_numeric(working["value"], errors="coerce")
working = working.dropna(subset=["year", "value"])
working = working[working["year"].between(1900, 2100)]
working["year"] = working["year"].astype(int)
working = working.groupby("year", as_index=False, sort=True)["value"].mean()
working = working.sort_values("year").reset_index(drop=True)

signature = (country_name, indicator_code, degree, tuple(working[["year","value"]].round(12).itertuples(index=False, name=None)))
if st.session_state.get("model_signature") != signature:
    st.session_state.pop("saved_calculation", None)
    st.session_state["model_signature"] = signature

try:
    info = fit_poly(working, degree)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
info["units"] = default_units

# Optional chronological holdout: train on earlier 80%, evaluate on later 20%.
holdout = {"n": 0, "mae": None, "r2": None}
if len(working) >= max(degree + 3, 10):
    split = max(degree + 1, int(math.floor(len(working) * .8)))
    train, test = working.iloc[:split], working.iloc[split:]
    if len(test) >= 2:
        try:
            hinfo = fit_poly(train, degree)
            yh = pred(hinfo, test.year.to_numpy())
            holdout["n"] = len(test)
            holdout["mae"] = float(np.mean(np.abs(test.value.to_numpy() - yh)))
            ss_total = float(np.sum((test.value.to_numpy() - test.value.mean())**2))
            holdout["r2"] = float(1 - np.sum((test.value.to_numpy() - yh)**2) / ss_total) if ss_total > 0 else float("nan")
        except Exception:
            pass

st.header("2. Fit and interpret")
st.write(f"**{country_name} · {indicator_label} · degree {degree}**")
st.code(poly_equation(info["coeff"], info["base"], default_units))
metric_cols = st.columns(5)
metric_cols[0].metric("R² (in-sample)", safe_float(info["r2"]))
metric_cols[1].metric("MAE (in-sample)", safe_float(info["mae"]))
metric_cols[2].metric("Observations", info["n"])
metric_cols[3].metric("Historical range", f'{info["min_year"]}–{info["max_year"]}')
metric_cols[4].metric("Mean observed value", safe_float(info["mean"]))
st.caption(f"Base year: {info['base']}. Residual = observed − fitted. Duplicate years were combined by mean in the working data.")
if holdout["n"]:
    st.write(f"**Chronological holdout (separate from in-sample metrics):** trained on earlier years, tested on the latest {holdout['n']} observations. Holdout MAE: {safe_float(holdout['mae'])}; holdout R²: {safe_float(holdout['r2'])}.")
else:
    st.write("Chronological holdout comparison unavailable: not enough observations for a meaningful split at this degree.")

with st.expander("Residuals and observed-year fitted values"):
    residual_view = info["residuals"].copy()
    residual_view["fitted_at_observed_year"] = residual_view["fitted"]
    st.dataframe(residual_view[["year","value","fitted_at_observed_year","residual"]], use_container_width=True)
    st.caption("Fitted values at observed years are model estimates. They are not the same as the actual observed values.")

# Model-specific interpretation, avoiding causal stories.
ymin, ymax = float(working.value.min()), float(working.value.max())
first_fit = float(pred(info, [info["min_year"]])[0])
last_fit = float(pred(info, [info["max_year"]])[0])
trend = "increases" if last_fit > first_fit else ("decreases" if last_fit < first_fit else "is roughly level")
st.subheader("Interpretation")
st.write(f"- **Observed domain:** years {info['min_year']}–{info['max_year']}. The fitted polynomial is mathematically defined for all real years, but the data-supported range is narrower.")
st.write(f"- **Observed range:** {safe_float(ymin)} to {safe_float(ymax)} {default_units}. The fitted curve's endpoint values {trend} from {safe_float(first_fit)} to {safe_float(last_fit)} over the observed range.")
st.write(f"- **Average rate across the historical interval:** {(last_fit-first_fit)/(info['max_year']-info['min_year']):,.6g} {default_units}/year, using fitted endpoints.")
st.write(f"- **Uncertainty:** in-sample MAE is the average absolute residual size ({safe_float(info['mae'])} {default_units}); individual residuals show misses at observed years. This is not a validated 95% prediction interval. Data quality, structural changes, missing observations, and polynomial shape limit reliability.")

if level in ("Precalculus", "Calculus"):
    degree = len(info["coeff"]) - 1
    st.write(f"- **Degree and coefficients:** degree {degree} controls the maximum number of turning points (at most {max(0, degree-1)}). Higher-degree terms can create sharper bends and unstable tails; coefficients describe the fitted curve in centered time t.")
    derivative = np.polyder(info["model"])
    second = np.polyder(info["model"], 2)
    st.write("- **Changing average rates / concavity:** compare fitted values over equal year intervals. Increasing secant slopes suggest the rate is rising; decreasing slopes suggest the rate is falling. Curvature should be interpreted cautiously, especially near sparse observations.")
    if level == "Calculus":
        st.write(f"- **First derivative:** f′(t) = `{np.poly1d(derivative)}`. **Second derivative:** f″(t) = `{np.poly1d(second)}`.")
        crit_t = np.roots(derivative) if len(derivative) > 1 else np.array([])
        real_crit = sorted([float(z.real + info["base"]) for z in crit_t if abs(z.imag) < 1e-7 and info["min_year"] <= z.real + info["base"] <= info["max_year"]])
        if real_crit:
            points = []
            for yr in real_crit:
                leftv = float(pred(info, [yr-0.01])[0]); rightv = float(pred(info, [yr+0.01])[0])
                kind = "local maximum" if leftv < float(pred(info,[yr])[0]) > rightv else ("local minimum" if leftv > float(pred(info,[yr])[0]) < rightv else "stationary point (not classified)")
                points.append(f"{yr:.3f} ({kind})")
            st.write("Critical points inside observed range: " + "; ".join(points) + ".")
        else:
            st.write("No real critical points were found inside the observed range.")
        endpoints = [(info["min_year"], float(pred(info,[info["min_year"]])[0])), (info["max_year"], float(pred(info,[info["max_year"]])[0]))]
        st.write("Interval extrema are checked against critical points and both endpoints; see the derivative and critical-point notes above. Inflection points require a real root of f″ and a verified concavity change; treat candidate roots cautiously if the sign does not change.")

st.header("3. Calculate values")
single_col, single_action = st.columns([2,1])
with single_col:
    single_year = st.number_input("Single year (1950–2100)", min_value=1950, max_value=2100, value=2025, step=1, key="single_year")
with single_action:
    st.write("")
    st.write("")
    calculate = st.button("Calculate Value", type="primary")
if calculate:
    val = float(pred(info, [single_year])[0])
    classification = "within fitted historical range" if info["min_year"] <= single_year <= info["max_year"] else "outside fitted historical range (extrapolation)"
    actual = working.loc[working.year == int(single_year), "value"]
    st.session_state["saved_calculation"] = {"signature": signature, "single_year": int(single_year), "value": val}
    st.write(f"**Year:** {int(single_year)}")
    st.write(f"**Estimated value:** {val:,.8g} {default_units}")
    st.write(f"**Classification:** {classification}")
    if len(actual):
        st.write(f"**Actual observed value at this year:** {float(actual.iloc[0]):,.8g} {default_units}. This is distinct from the fitted estimate.")
    else:
        st.write("**Actual observed value:** no observation exists for this exact year.")
    if single_year < info["min_year"] or single_year > info["max_year"]:
        st.warning("This is an extrapolation. Polynomial tails can grow, reverse direction, or become physically implausible. The value is not clipped; check its scale and context.")
    if abs(val) > max(1, abs(info["mean"])) * 100:
        st.warning("The estimate is unusually large relative to the historical mean. Inspect the plot and treat it as a likely unstable extrapolation.")

st.subheader("Batch calculation")
batch_mode = st.radio("Batch input method", ["Year range", "Custom years"], horizontal=True)
batch_years = []
invalid = []
if batch_mode == "Year range":
    c1, c2, c3 = st.columns(3)
    start_y = c1.number_input("Start year", 1950, 2100, 2000, 1, key="batch_start")
    end_y = c2.number_input("End year", 1950, 2100, 2030, 1, key="batch_end")
    step_y = c3.slider("Step (1–10)", 1, 10, 5)
    if start_y > end_y:
        st.error("Start year must be less than or equal to end year.")
    else:
        batch_years = list(range(int(start_y), int(end_y)+1, int(step_y)))
else:
    custom = st.text_input("Comma-separated years", "2000, 2010, 2020, 2030")
    batch_years, invalid = parse_years(custom)
    if invalid:
        st.warning("Ignored invalid year entries: " + ", ".join(invalid) + ". Enter whole years from 1950 through 2100, separated by commas.")
    st.caption("Repeated years are deduplicated. Blank entries are ignored.")
if batch_years:
    batch_vals = pred(info, batch_years)
    batch_df = pd.DataFrame({"year": batch_years, "estimate": batch_vals})
    batch_df["units"] = default_units
    batch_df["classification"] = batch_df.year.apply(lambda y: "within fitted historical range" if info["min_year"] <= y <= info["max_year"] else "outside fitted historical range (extrapolation)")
    actual_map = working.set_index("year").value.to_dict()
    batch_df["actual_observation"] = batch_df.year.map(actual_map)
    st.dataframe(batch_df, use_container_width=True)
    if (batch_df.classification.str.startswith("outside")).any():
        st.warning("Some batch years are outside the fitted historical range. Treat these as extrapolations, not reliable forecasts.")
else:
    batch_df = pd.DataFrame(columns=["year","estimate","units","classification","actual_observation"])
    if batch_mode == "Custom years":
        st.info("Enter at least one valid year to calculate a batch.")

st.header("4. Visualize")
fig = make_plot(working, info, single_year=(int(single_year) if st.session_state.get("saved_calculation", {}).get("signature") == signature else None),
                batch_years=batch_years, sample_step=sample_step, title=f"{country_name}: {indicator_label} (degree {degree})")
st.pyplot(fig, use_container_width=True)
st.caption("Observed points show source data; the smooth line is the fitted curve; boundary lines mark the observed range; estimates outside the boundaries are extrapolations.")
st.checkbox("Printer mode (simplified report)", key="printer_mode")
if st.session_state.get("printer_mode"):
    st.markdown("### Printable summary")
    st.write(f"{country_name} — {indicator_label} | Degree {degree} | Base year {info['base']} | Observations {info['n']}")
    st.write(f"Equation: {poly_equation(info['coeff'], info['base'], default_units)}")
    st.write(f"In-sample R²: {safe_float(info['r2'])} | In-sample MAE: {safe_float(info['mae'])} {default_units}")
    st.write(f"Observed range: {info['min_year']}–{info['max_year']}. This descriptive model does not establish causes and extrapolation is uncertain.")

st.header("5. Export current model")
summary_rows = [
    ("Country", country_name), ("Indicator", indicator_label), ("Indicator code", indicator_code),
    ("Units", default_units), ("Degree", degree), ("Base year", info["base"]),
    ("Equation", poly_equation(info["coeff"], info["base"], default_units)),
    ("R-squared (in-sample)", info["r2"]), ("MAE (in-sample)", info["mae"]),
    ("Observation count", info["n"]), ("Historical start", info["min_year"]),
    ("Historical end", info["max_year"]), ("Observed mean", info["mean"]),
    ("Chronological holdout count", holdout["n"]), ("Chronological holdout MAE", holdout["mae"]),
    ("Chronological holdout R-squared", holdout["r2"]), ("Working data edited", bool(use_edits)),
    ("Export generated (UTC)", datetime.utcnow().isoformat(timespec="seconds")),
]
summary_df = pd.DataFrame(summary_rows, columns=["field","value"])
residual_export = info["residuals"][["year","value","fitted","residual"]].copy()
export_results = batch_df.copy()
if st.session_state.get("saved_calculation", {}).get("signature") == signature:
    s = st.session_state["saved_calculation"]
    one = pd.DataFrame([{"year": s["single_year"], "estimate": s["value"], "units": default_units,
                         "classification": "within fitted historical range" if info["min_year"] <= s["single_year"] <= info["max_year"] else "outside fitted historical range (extrapolation)",
                         "actual_observation": actual_map.get(s["single_year"]) if 'actual_map' in locals() else (float(working.loc[working.year == s["single_year"], "value"].iloc[0]) if (working.year == s["single_year"]).any() else np.nan)}])
    export_results = pd.concat([one, export_results], ignore_index=True).drop_duplicates(subset=["year"], keep="first")
csv_bytes = residual_export.to_csv(index=False).encode("utf-8")
xlsx = excel_bytes(summary_df, residual_export, export_results)
pdf_summary = "\n".join([f"{k}: {v}" for k,v in summary_rows if k not in ("Export generated (UTC)",)])
pdf_summary += "\nFindings/limitations: Residuals and MAE describe in-sample error. These are not 95% prediction intervals. Extrapolations may be implausible; no causal interpretation is inferred."
pdf = pdf_bytes(pdf_summary, fig, residual_export, export_results)
d1, d2, d3 = st.columns(3)
d1.download_button("Download CSV", csv_bytes, file_name="hhm_working_data.csv", mime="text/csv")
d2.download_button("Download Excel (.xlsx)", xlsx, file_name="hhm_model_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
d3.download_button("Download PDF report", pdf, file_name="hhm_model_report.pdf", mime="application/pdf")

with st.expander("6. Data source, deployment, and verification checklist"):
    st.markdown("""
**Data source and assumptions**
- Historical observations come from the World Bank Indicators API. Country/indicator coverage varies; missing years are omitted.
- The model uses centered time: `t = year − base year`. Coefficients are fitted against `t` and predictions apply this transform once.
- R² and MAE are in-sample scores. A chronological holdout score is shown separately when there are enough observations.
- Polynomial fits describe patterns; they do not prove historical causes. No default 95% prediction interval is claimed.

**Local run**
1. Install Python 3.10 or newer.
2. In the project folder, run `pip install -r requirements.txt`.
3. Run `python -m streamlit run app.py`.

**GitHub + Streamlit Community Cloud**
1. Create a GitHub repository and upload `app.py` and `requirements.txt`.
2. In Streamlit Community Cloud, connect GitHub, choose the repository and `app.py`, then deploy.

**Render**
- Create a **Python Web Service** from the GitHub repository.
- Build command: `pip install -r requirements.txt`
- Start command: `python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT`

**Verification checklist**
- Confirm source country/indicator and units against the World Bank page/API.
- Edit data, enable the edited working copy, reset it, and confirm the fit changes/restores.
- Try duplicate years, missing values, and fewer than degree + 1 distinct years.
- Check centered equation, residuals, R², MAE, and observation count.
- Verify a year entered in single-year mode matches the same year in batch mode.
- Try invalid custom years, reversed range endpoints, API failure/offline behavior, and degrees 1–6.
- Test each chart control, printer mode, and all three exports. Open the `.xlsx` workbook and PDF to confirm they are genuine formats.
- After deployment, inspect the service logs if startup or API requests fail.

**Common errors**
- `ModuleNotFoundError`: install dependencies from `requirements.txt`.
- API/network timeout or HTTP errors: retry and inspect World Bank API availability and deployment logs.
- Render port errors: verify the exact start command uses `$PORT`.
- Too few observations: lower the degree or choose an indicator/country with more distinct years.
- Wild predictions: reduce the degree, stay within observed coverage, and do not clip the result silently.
""")
st.caption("Team explanation: data flow is source API → clean working data → centered-time fit → residual/metric calculation → single/batch estimates → chart and exports. Higher-degree polynomials can overfit and extrapolate poorly; MAE summarizes typical in-sample absolute error but does not guarantee future error.")
