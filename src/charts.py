import io
from src import data
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests



def data_preview(df):
    # --- 0) Dataset introduction ---
    st.markdown(
        """
## Dataset presentation

The dataset used in this dashboard comes from the [**Santé Publique France**](https://www.santepubliquefrance.fr/) open data portal,  
available on [data.gouv.fr](https://www.data.gouv.fr/datasets/synthese-des-indicateurs-de-suivi-de-lepidemie-covid-19/).

It gathers daily hospital and testing indicators related to the COVID-19 epidemic across metropolitan France.  
Each row corresponds to a specific department and date, with indicators describing the hospital situation  
such as hospitalizations, intensive care occupancy, recoveries, deaths, and test positivity rates.

The goal of this dataset is to provide a clear view of how the epidemic evolved both temporally and geographically.

**Main dimensions:
- **Temporal:** from the first recorded cases in 2020 to the most recent updates.  
- **Geographical:** includes all departments and regions of metropolitan France.  
- **Indicators:** hospitalizations, ICU, new admissions, deaths, recoveries, and testing data.

Below, you’ll find a technical overview of the dataset, including its structure, types of variables, and main characteristics.
"""
    )

    st.divider()

    # 1) Basic counts and date range
    n_rows, n_cols = df.shape

    dates = pd.to_datetime(df["date"], errors="coerce") if "date" in df.columns else pd.Series(dtype="datetime64[ns]")
    dmin, dmax = dates.min(), dates.max()
    duration_days = (dmax - dmin).days if pd.notna(dmin) and pd.notna(dmax) else np.nan

    n_reg = df["lib_reg"].nunique(dropna=True) if "lib_reg" in df.columns else np.nan
    n_dep = df["lib_dep"].nunique(dropna=True) if "lib_dep" in df.columns else np.nan

    # 2) Column type distribution
    n_numeric = df.select_dtypes(include=["number"]).shape[1]
    n_dates = df.select_dtypes(include=["datetime", "datetimetz"]).shape[1]
    n_string = df.select_dtypes(include=["string", "object"]).shape[1]
    by_dtype = (
        pd.Series({"Numeric": n_numeric, "Datetime": n_dates, "String/Object": n_string})
        .reset_index()
        .rename(columns={"index": "Type", 0: "Count"})
    )

    # 3) KPI summary header
    with st.container(border=True):
        st.subheader("Overview")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{n_rows:,}".replace(",", " "))
        c2.metric("Columns", f"{n_cols}")
        c3.metric("Regions (unique)", f"{n_reg}")
        c4.metric("Departments (unique)", f"{n_dep}")

        c5, c6, c7 = st.columns(3)
        c5.metric("Min date", dmin.date().isoformat() if pd.notna(dmin) else "—")
        c6.metric("Max date", dmax.date().isoformat() if pd.notna(dmax) else "—")
        c7.metric("Span (days)", f"{duration_days}" if pd.notna(duration_days) else "—")

    st.divider()

    # 4) Column glossary
    st.subheader("Column glossary")
    st.markdown(
        """
dep — Department code (e.g., 01, 2A, 75)  
date — Observation date  
reg — INSEE region code (e.g., 84)  
lib_dep — Department name  
lib_reg — Region name  

tx_pos — Test positivity rate (fraction: 0–1 or %)  
tx_incid — Incidence (often per 100k inhabitants over 7 days)  
TO — ICU bed occupancy rate (fraction)  

hosp — Hospitalized COVID patients (stock)  
rea — Patients in ICU/critical care (stock)  
rad — Cumulative discharges (returned home)  
dchosp — Cumulative in-hospital deaths  

incid_hosp — New hospital admissions (24h)  
incid_rea — New ICU admissions (24h)  
incid_rad — New discharges (24h)  
incid_dchosp — New in-hospital deaths (24h)  

reg_rea — Regional total ICU patients (repeated per department row)  
reg_incid_rea — Regional new ICU admissions (repeated)  

pos — Number of positive tests (often 7-day smoothed)  
pos_7j — Positive tests accumulated over 7 days
"""
    )
    st.divider()

    # 5) Data previews in tabs
    st.subheader("Information preview")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Head (5)", "Random (5)", "Info dump", "Describe dump", "Columns by type"]
    )

    with tab1:
        st.dataframe(df.head(5))

    with tab2:
        st.dataframe(df.sample(5, random_state=42) if len(df) >= 5 else df)

    with tab3:
        buf = io.StringIO()
        df.info(buf=buf)
        st.code(buf.getvalue(), language="text")

    with tab4:
        st.dataframe(df.describe(), use_container_width=True)

    with tab5:
        fig = px.bar(by_dtype, x="Type", y="Count", title="Column types")
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
























def national_view(df: pd.DataFrame) -> None:


    # 1) Intro / context
    st.markdown(
        """
# National aggregation
This section aggregates departmental records by date to obtain national-level indicators.
"""
    )

    # 2) Metric dictionaries (labels and units)
    ALL_METRICS = [
        "hosp", "rea",
        "incid_hosp", "incid_rea", "incid_rad", "incid_dchosp",
        "rad", "dchosp",
        "pos", "pos_7j",
    ]
    LABELS = {
        "hosp": "Hospitalized (stock)",
        "rea": "ICU (stock)",
        "incid_hosp": "New hospital admissions (24h)",
        "incid_rea": "New ICU admissions (24h)",
        "incid_rad": "New discharges (24h)",
        "incid_dchosp": "New in-hospital deaths (24h)",
        "rad": "Discharged (cumulative)",
        "dchosp": "In-hospital deaths (cumulative)",
        "pos": "Positive tests (often 7d smoothed)",
        "pos_7j": "Positive tests (7-day sum)",
    }
    UNITS = {
        "hosp": "patients",
        "rea": "patients",
        "rad": "patients (cum.)",
        "dchosp": "deaths (cum.)",
        "incid_hosp": "patients/day",
        "incid_rea": "patients/day",
        "incid_rad": "patients/day",
        "incid_dchosp": "deaths/day",
        "pos": "tests",
        "pos_7j": "tests (7d)",
    }

    # 3) Normalize schema
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")

    st.caption("National aggregated time series (France)")
    st.dataframe(df.head(), use_container_width=True)
    st.divider()

    # 4) Column glossary
    st.subheader("Column glossary")
    st.markdown(
        """
hosp – hospitalized (stock) • rea – ICU (stock) • rad – cumulative discharges • dchosp – cumulative in-hospital deaths  
incid_hosp – new hospital admissions (24h) • incid_rea – new ICU admissions (24h) • incid_rad – new discharges (24h) • incid_dchosp – new in-hospital deaths (24h)  
pos – number of positive tests • pos_7j – 7-day sum of positives
"""
    )
    st.divider()

    # 5) KPI header (latest values and deltas vs 7 days ago)
    c1, c2, c3 = st.columns(3)
    latest_date = df["date"].max()
    week_ago_date = latest_date - pd.Timedelta(days=7)

    def kpi_for(col: str):
        cur = df.loc[df["date"] == latest_date, col].sum() if col in df.columns else None
        prev = df.loc[df["date"] == week_ago_date, col].sum() if col in df.columns else None
        delta = None if (cur is None or prev is None or pd.isna(prev)) else cur - prev
        return cur, delta

    cur_hosp, d_hosp = kpi_for("hosp")
    cur_rea, d_rea = kpi_for("rea")
    cur_incid_hosp, d_incid_hosp = kpi_for("incid_hosp")

    c1.metric(
        "Hospitalized (latest)",
        f"{int(cur_hosp):,}".replace(",", " "),
        f"{'+' if d_hosp and d_hosp >= 0 else ''}{int(d_hosp) if d_hosp is not None else '—'} vs 7d",
    )
    c2.metric(
        "ICU (latest)",
        f"{int(cur_rea):,}".replace(",", " "),
        f"{'+' if d_rea and d_rea >= 0 else ''}{int(d_rea) if d_rea is not None else '—'} vs 7d",
    )
    c3.metric(
        "New hospital admissions (24h)",
        f"{int(cur_incid_hosp):,}".replace(",", " "),
        f"{'+' if d_incid_hosp and d_incid_hosp >= 0 else ''}{int(d_incid_hosp) if d_incid_hosp is not None else '—'} vs 7d",
    )
    st.divider()

    # 6) Main time series (controls + line chart)
    left, right = st.columns([2, 1])
    with left:
        metric = st.selectbox(
            "Metric",
            ALL_METRICS,
            index=ALL_METRICS.index("hosp"),
            format_func=lambda c: LABELS.get(c, c),
            key="nat_metric",
        )
    with right:
        smooth = st.checkbox("7-day rolling average", value=True)
        weekly = st.checkbox("Aggregate by week (Mon–Sun)", value=False)

    ts = df[["date", metric]].dropna()

    # Optional weekly resample
    if weekly:
        ts = (
            ts.set_index("date")[metric]
            .resample("W-MON")
            .sum()
            .reset_index()
            .rename(columns={metric: metric})
        )
        unit_label = f"{UNITS.get(metric, '')} per week"
    else:
        unit_label = UNITS.get(metric, "")

    # Optional smoothing (skip for cumulatives)
    use_smooth = smooth and metric not in ["rad", "dchosp"]
    if use_smooth:
        ts["rolling"] = ts[metric].rolling(7, min_periods=1).mean()
        ycol = "rolling"
    else:
        ycol = metric

    fig = px.line(
        ts, x="date", y=ycol,
        labels={"date": "Date", ycol: unit_label},
        title=f"{LABELS.get(metric, metric)}",
    )
    # Show raw markers when smoothing is on (visual context)
    if use_smooth:
        fig.add_scatter(x=ts["date"], y=ts[metric], mode="markers", name="daily/raw", opacity=0.3)

    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=420)
    st.plotly_chart(fig, use_container_width=True, key=f"nat_ts_{metric}_{use_smooth}_{weekly}")

    st.markdown(
        """
The curve reveals clear phases in the data, alternating between periods of growth and decline.
These cycles indicate that the phenomenon evolves over time through successive waves rather than a steady trend.
Each phase likely corresponds to distinct underlying dynamics or external factors influencing the overall behavior of the system.
"""
    )

    # 7) Stocks vs daily flows (overlay)
    st.subheader("Stocks vs daily flows")
    stock_col = st.selectbox("Stock", ["hosp", "rea"], index=0, format_func=lambda c: LABELS[c], key="stock_col")
    flow_col = st.selectbox(
        "Daily flow",
        ["incid_hosp", "incid_rea", "incid_rad", "incid_dchosp"],
        index=0,
        format_func=lambda c: LABELS[c],
        key="flow_col",
    )
    dd = df[["date", stock_col, flow_col]].dropna().sort_values("date")
    dd["flow_7d"] = dd[flow_col].rolling(7, min_periods=1).mean()

    fig2 = px.line(
        dd,
        x="date",
        y=stock_col,
        labels={"date": "Date", stock_col: UNITS.get(stock_col, "")},
        title=f"{LABELS[stock_col]} (line) + {LABELS[flow_col]} 7-day avg (bars)",
    )
    fig2.add_bar(x=dd["date"], y=dd["flow_7d"], name=f"{LABELS[flow_col]} (7d avg)", opacity=0.45)
    fig2.update_layout(barmode="overlay", height=420, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig2, use_container_width=True, key="stocks_vs_flows")
    st.divider()

    # 8) Small multiples (key outcomes)
    st.subheader("Small multiples: key outcomes")
    subset = ["hosp", "rea", "incid_hosp", "incid_dchosp"]
    long = df.melt(id_vars="date", value_vars=subset, var_name="metric", value_name="value").dropna()
    long["label"] = long["metric"].map(LABELS)
    if smooth:
        long["value"] = long.groupby("metric")["value"].transform(lambda s: s.rolling(7, min_periods=1).mean())

    fig3 = px.line(
        long,
        x="date",
        y="value",
        facet_col="label",
        facet_col_wrap=2,
        height=500,
        labels={"date": "Date", "value": "Value", "label": ""},
        category_orders={"label": [LABELS[m] for m in subset]},
    )
    # Clean facet titles
    fig3.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig3.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig3, use_container_width=True, key="small_multiples")

    st.markdown(
        """
These small multiples show core indicators side by side: hospitalizations, ICU occupancy, new admissions, and in-hospital deaths over time.
Aligning timelines makes it easier to compare dynamics: peaks and declines occur in similar periods, with different magnitudes and lags.
The view helps relate severity (ICU and deaths) to overall hospital load across successive waves.
"""
    )
    st.divider()

    # 9) ICU share among hospitalized
    if {"hosp", "rea"}.issubset(df.columns):
        st.subheader("ICU share among hospitalized")
        share = df[["date", "hosp", "rea"]].dropna().copy()
        share = share[share["hosp"] > 0]
        share["icu_share"] = share["rea"] / share["hosp"]
        fig4 = px.line(
            share,
            x="date",
            y="icu_share",
            labels={"date": "Date", "icu_share": "ICU / Hospitalized"},
            title="ICU share (rea / hosp)",
        )
        fig4.add_hline(y=0.25, line_dash="dot", annotation_text="25% reference", annotation_position="top left")
        fig4.update_layout(height=360, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig4, use_container_width=True, key="icu_share")

        st.markdown(
            """
This chart shows the share of ICU patients among hospitalized individuals.
The proportion fluctuates across phases, reaching high levels in some waves before declining.
After mid-2022, the share stabilizes at a lower level, suggesting reduced severity relative to total hospitalizations.
"""
        )




















def data_processing(df_ini: pd.DataFrame):
    # 1) Pipeline overview (raw dataset stats)
    st.subheader("Pipeline overview")
    with st.container(border=True):
        r0, c0, m0 = data.df_info(df_ini)
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows (raw)", f"{r0:,}".replace(",", " "))
        col2.metric("Columns (raw)", f"{c0}")
        col3.metric("Missing values (raw)", f"{m0:,}".replace(",", " "))

    st.divider()
    st.caption("The steps below reproduce src.data.clean_data.")

    # 2) Tabs scaffold
    tab_step, tab_summary, tab_previews, tab_export = st.tabs(
        ["Step-by-step", "Final summary", "Previews", "Export"]
    )

    # 3) Step-by-step walkthrough
    with tab_step:
        st.markdown("### 1) Drop columns ['cv_dose1', 'R']")
        cols_to_drop = ["cv_dose1", "R"]
        step1 = df_ini.drop(columns=cols_to_drop, errors="ignore")
        dropped_effective = [c for c in cols_to_drop if c in df_ini.columns]
        st.write(f"Dropped columns: {dropped_effective if dropped_effective else 'none (absent)'}")
        st.dataframe(step1.head(10), use_container_width=True)

        st.markdown("### 2) Remove duplicates (keep='first')")
        before_dups = len(step1)
        step2 = step1.drop_duplicates(keep="first")
        st.write(f"Removed duplicates: {before_dups - len(step2)}")
        st.dataframe(step2.head(10), use_container_width=True)

        st.markdown("### 3) Drop rows containing at least one NaN")
        before_na = len(step2)
        step3 = step2.dropna(axis=0, how="any")
        st.write(f"Dropped rows with NaN: {before_na - len(step3)}")
        with st.expander("Missing values per column (before this step)"):
            st.dataframe(
                step2.isna().sum().to_frame("NaN").sort_values("NaN", ascending=False),
                use_container_width=True,
            )
        st.dataframe(step3.head(10), use_container_width=True)

        st.markdown("### 4) Reset index and parse 'date' column")
        step4 = step3.reset_index(drop=True).copy()
        if "date" in step4.columns:
            step4["date"] = pd.to_datetime(step4["date"], errors="coerce")
            bad_dates = int(step4["date"].isna().sum())
            st.write(f"Converted to datetime. Non-parsable values -> NaT: {bad_dates}")
        else:
            st.info("Column 'date' not found: no conversion applied.")
        st.dataframe(step4.head(10), use_container_width=True)

        st.markdown("---")
        st.subheader("Pipeline diagram")
        st.graphviz_chart(
            """
digraph G {
  rankdir=LR;
  node [shape=box, fontsize=10];
  A [label="Raw CSV"];
  B [label="Drop ['cv_dose1','R']"];
  C [label="Drop duplicates"];
  D [label="Drop rows with NaN"];
  E [label="Reset index + to_datetime('date')"];
  F [label="Clean dataset"];
  A -> B -> C -> D -> E -> F;
}
"""
        )

        # Keep the cleaned frame for other tabs
        st.session_state["cleaned_df"] = step4

    # 4) Final summary (KPIs + preview)
    with tab_summary:
        st.subheader("Summary after cleaning")
        cleaned_df = st.session_state.get("cleaned_df", data.clean_data(df_ini))
        r1, c1, m1 = data.df_info(cleaned_df)

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Rows (raw)", f"{r0:,}".replace(",", " "))
        k2.metric("Rows (clean)", f"{r1:,}".replace(",", " "))
        k3.metric("Columns (raw)", f"{c0}")
        k4.metric("Columns (clean)", f"{c1}")
        k5.metric("Missing (raw)", f"{m0:,}".replace(",", " "))
        k6.metric("Missing (clean)", f"{m1:,}".replace(",", " "))

        st.markdown("Clean preview")
        st.dataframe(cleaned_df.head(20), use_container_width=True)

    # 5) Previews (raw vs clean + dtype counts)
    with tab_previews:
        st.subheader("Previews (raw vs clean)")
        cleaned_df = st.session_state.get("cleaned_df", data.clean_data(df_ini))

        t_raw, t_clean = st.tabs(["Raw (head)", "Clean (head)"])
        with t_raw:
            st.dataframe(df_ini.head(20), use_container_width=True)
        with t_clean:
            st.dataframe(cleaned_df.head(20), use_container_width=True)

        st.markdown("Columns by dtype (clean)")
        by_dtype = pd.Series(
            {
                "Numeric": cleaned_df.select_dtypes(include=["number"]).shape[1],
                "Datetime": cleaned_df.select_dtypes(include=["datetime", "datetimetz"]).shape[1],
                "String/Object": cleaned_df.select_dtypes(include=["string", "object"]).shape[1],
            }
        ).reset_index().rename(columns={"index": "Type", 0: "Count"})
        fig = px.bar(by_dtype, x="Type", y="Count", title="Column types (clean)")
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # 6) Export (clean CSV)
    with tab_export:
        st.subheader("Export cleaned dataset")
        cleaned_df = st.session_state.get("cleaned_df", data.clean_data(df_ini))
        st.download_button(
            "Download cleaned CSV",
            data.to_csv_bytes(cleaned_df),
            file_name="data_clean.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # 7) Data quality and limitations (narrative)
    st.markdown("### Data quality and limitations")
    st.info(
        """
Data quality assessment:

The dataset went through a four-step cleaning process to improve reliability and usability:
1) Removal of unnecessary or empty columns.  
2) Deduplication to avoid redundant records.  
3) Row-wise removal of missing values to keep metrics aligned.  
4) Standardization of the 'date' column to a proper datetime format.  

Limitations to keep in mind:
- Removing all rows with missing values may bias the dataset toward better-reported departments or periods.  
- Some dates may have incomplete reporting, creating temporal gaps.  
- Reported indicators (e.g., hospitalizations, ICU admissions) depend on regional reporting systems, which may vary in accuracy or timing.  

Overall, the cleaned dataset is suitable for exploratory analysis and visualization, but results should be interpreted with awareness of these potential reporting biases.
"""
    )





















def regional_view(df: pd.DataFrame) -> None:

    # 1) Validate input
    df = df.copy()
    if "date" not in df.columns or "lib_reg" not in df.columns:
        st.error("This view expects columns: 'date' and 'lib_reg'.")
        return

    # 2) Normalize schema
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")

    # 3) Identify numeric indicators available
    num_candidates = [
        "hosp", "rea",
        "incid_hosp", "incid_rea", "incid_rad", "incid_dchosp",
        "rad", "dchosp", "pos", "pos_7j",
    ]
    num_cols = [c for c in num_candidates if c in df.columns]
    if not num_cols:
        st.error("No numeric COVID indicators found in the dataset.")
        return

    # 4) Page header and context
    latest_date = df["date"].max()
    st.title("Regional View")
    st.caption(f"Latest available date in data: {latest_date.date()}")
    st.divider()

    # 5) Latest snapshot by region (aggregated on numeric columns)
    latest_df = (
        df.loc[df["date"] == latest_date, ["lib_reg", *num_cols]]
          .groupby("lib_reg", as_index=False)
          .sum(numeric_only=True)
    )

    # 6) KPI row (select a region and compare)
    st.subheader("Regional KPIs")
    left, right = st.columns([2, 1])
    with right:
        regions_all = sorted(latest_df["lib_reg"].unique().tolist())
        if not regions_all:
            st.info("No regions found at the latest date.")
            return
        sel_reg = st.selectbox("Region", options=regions_all)

    with left:
        # Derive kpis with safeguards
        hosp_val = latest_df.loc[latest_df["lib_reg"] == sel_reg, "hosp"].sum() if "hosp" in latest_df else None
        rea_val  = latest_df.loc[latest_df["lib_reg"] == sel_reg, "rea"].sum()  if "rea"  in latest_df else None
        icu_share = (rea_val / hosp_val) if (rea_val is not None and hosp_val and hosp_val > 0) else None

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Hospitalized (selected region)",
            f"{int(hosp_val):,}".replace(",", " ") if hosp_val is not None else "—",
            help="Current stock of hospitalized patients."
        )
        c2.metric(
            "ICU (selected region)",
            f"{int(rea_val):,}".replace(",", " ") if rea_val is not None else "—",
            help="Current ICU/critical care stock."
        )
        c3.metric(
            "ICU share (rea / hosp)",
            f"{icu_share:.2%}" if icu_share is not None else "—",
            help="Proportion of ICU among hospitalized."
        )

        st.caption("Question answered: How does one region compare on key indicators right now?")
    st.divider()

    # 7) Top regions by hospitalizations (latest date)
    if "hosp" in latest_df.columns:
        st.subheader("Top regions by hospitalizations (latest)")
        st.caption("Question answered: Which regions have the highest and lowest hospitalizations right now?")

        top_sorted = latest_df.sort_values("hosp", ascending=False)
        fig_bar = px.bar(
            top_sorted,
            x="lib_reg",
            y="hosp",
            color="hosp",
            color_continuous_scale="Reds",
            labels={"lib_reg": "", "hosp": "Hospitalized"},
        )
        fig_bar.update_layout(xaxis_tickangle=-30, height=420, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_bar, use_container_width=True, key="bar_hosp")
    else:
        st.info("Column 'hosp' not available for the bar chart.")
    st.divider()

    # 8) Multi-region time series (hospitalizations)
    st.subheader("Hospitalization trends by region")
    st.caption("Question answered: How do hospitalization trends evolve across regions over time?")

    regions = sorted(df["lib_reg"].dropna().unique().tolist())
    default_regs = regions[:3] if len(regions) >= 3 else regions
    pick = st.multiselect("Select regions to compare:", options=regions, default=default_regs, key="regions_ts")
    smooth = st.checkbox("7-day rolling average", value=True, key="smooth_ts")

    if pick and "hosp" in df.columns:
        sub = df.loc[df["lib_reg"].isin(pick), ["date", "lib_reg", "hosp"]].dropna()
        if smooth:
            sub["hosp_smooth"] = sub.groupby("lib_reg")["hosp"].transform(lambda s: s.rolling(7, min_periods=1).mean())
            ycol = "hosp_smooth"
        else:
            ycol = "hosp"

        fig_line = px.line(
            sub.sort_values("date"),
            x="date", y=ycol, color="lib_reg",
            labels={"date": "Date", ycol: "Hospitalized", "lib_reg": "Region"},
        )
        fig_line.update_layout(height=450, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_line, use_container_width=True, key="ts_hosp_regions")
    else:
        st.info("Select at least one region and ensure 'hosp' exists.")
    st.divider()

    # 9) Daily flows for a chosen region (discharges vs deaths)
    st.subheader("Daily flows in a selected region")
    st.caption("Question answered: Are recoveries and deaths evolving differently across regions?")

    flows_ok = {"incid_rad", "incid_dchosp"}.issubset(df.columns)
    reg2 = st.selectbox("Region (flows)", options=regions, index=0 if regions else None)

    if reg2 and flows_ok:
        flows = (
            df.loc[df["lib_reg"] == reg2, ["date", "incid_rad", "incid_dchosp"]]
              .dropna()
              .sort_values("date")
        )
        flows["rad_7d"] = flows["incid_rad"].rolling(7, min_periods=1).mean()
        flows["dchosp_7d"] = flows["incid_dchosp"].rolling(7, min_periods=1).mean()

        long = flows.melt(
            id_vars="date",
            value_vars=["rad_7d", "dchosp_7d"],
            var_name="metric",
            value_name="value"
        )
        long["metric"] = long["metric"].map(
            {"rad_7d": "Discharges (7d avg)", "dchosp_7d": "Deaths (7d avg)"}
        )

        fig_flows = px.line(
            long,
            x="date", y="value", color="metric",
            labels={"date": "Date", "value": "Patients/day", "metric": ""},
            title=f"Daily outcomes in {reg2}",
        )
        fig_flows.update_layout(height=420, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_flows, use_container_width=True, key="flows_region")
    else:
        st.info("This plot requires columns 'incid_rad' and 'incid_dchosp'.")
























def map_view(df: pd.DataFrame) -> None:


    # 1) Validate input
    required = {"date", "dep"}
    if not required.issubset(df.columns):
        st.error("Map View expects at least 'date' and 'dep' columns.")
        return

    # 2) Normalize schema
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["dep"] = df["dep"].astype(str)
    df = df.sort_values("date")

    # 3) Identify numeric metrics available
    id_like = {"date", "dep", "reg", "lib_dep", "lib_reg"}
    num_cols = [c for c in df.columns if c not in id_like and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        st.error("No numeric indicators found to map.")
        return

    st.title("Map View — Metropolitan France (Departments)")

    # 4) Sidebar controls (metric, date, quantiles)
    with st.sidebar:
        st.header("Map controls")

        metric = st.selectbox(
            "Metric to map",
            options=num_cols,
            index=num_cols.index("hosp") if "hosp" in num_cols else 0,
        )

        available_dates = sorted(df["date"].dropna().unique())
        if len(available_dates) == 0:
            st.error("No valid dates found in the dataset.")
            return

        formatted_dates = [d.strftime("%Y-%m-%d") for d in available_dates]
        selected_date_str = st.selectbox("Select date", options=formatted_dates, index=len(formatted_dates) - 1)
        selected_date = pd.to_datetime(selected_date_str)

        q_low = st.slider("Lower clip quantile", 0.0, 0.2, 0.02, step=0.01)
        q_high = st.slider("Upper clip quantile", 0.8, 1.0, 0.98, step=0.01)

    # 5) Compute snapshot for the selected date
    OVERSEAS_DEPS = {"971", "972", "973", "974", "976", "975", "977", "978", "984", "986", "987", "988", "989"}

    cols_needed = ["dep", "lib_dep", metric] if "lib_dep" in df.columns else ["dep", metric]
    snap = (
        df.loc[df["date"] == selected_date, cols_needed]
        .groupby([c for c in cols_needed if c != metric], as_index=False)
        .sum(numeric_only=True)
    )
    snap = snap[~snap["dep"].isin(OVERSEAS_DEPS)]

    # 6) Load GeoJSON (cached)
    @st.cache_data(show_spinner=False)
    def _get_departements_geojson() -> dict:
        url = "https://france-geojson.gregoiredavid.fr/repo/departements.geojson"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return r.json()

    geo = _get_departements_geojson()

    # 7) Color scaling (quantile clipping)
    if len(snap) == 0 or metric not in snap.columns:
        st.info("No data available for the selected date/metric.")
        return
    vmin = float(snap[metric].quantile(q_low))
    vmax = float(snap[metric].quantile(q_high))
    if vmin == vmax:
        vmin, vmax = None, None  # fall back to auto-scaling

    # 8) Render choropleth
    st.caption(f"Question answered: Which departments show the highest levels for {metric}?")
    st.write(f"Date selected: {selected_date.date()}")

    fig = px.choropleth(
        snap,
        geojson=geo,
        featureidkey="properties.code",
        locations="dep",
        color=metric,
        color_continuous_scale="Reds",
        range_color=(vmin, vmax) if (vmin is not None and vmax is not None) else None,
        hover_name="lib_dep" if "lib_dep" in snap.columns else None,
        hover_data={metric: True, "dep": True},
        labels={metric: metric.replace("_", " ").title()},
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(height=650, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # 9) Help / how to read
    with st.expander("How to read this map"):
        st.markdown(
            f"""
- Metric: The color encodes the selected indicator (currently {metric}).
- Date: {selected_date.date()} (chosen from the sidebar).
- Quantile clipping: Colors are restricted between the selected quantiles to improve contrast.
- Scope: Metropolitan France only (overseas territories excluded).
"""
        )

    # 10) Narrative summary
    st.markdown(
        """
Interpretation:

This choropleth map provides a spatial overview of the COVID-19 situation across metropolitan France.
The color intensity highlights the departments most affected according to the selected indicator and date.
Overall, Île-de-France consistently appears as the region with the highest concentration of cases and hospitalizations, reflecting its high population density and mobility.
Other regions (e.g., Provence-Alpes-Côte d’Azur, Auvergne-Rhône-Alpes, Grand Est) also show periodic peaks depending on the phase of the pandemic.
These shifts underline how epidemic waves spread unevenly across time and space, emphasizing the need for localized monitoring and adaptive strategies.
"""
    )







def conclusion_page():
    st.title("Conclusion")

    st.markdown("## Key Takeaways")
    st.markdown("""
- The evolution of COVID-19 indicators across France shows distinct temporal phases corresponding to epidemic waves.  
- Hospitalizations and ICU admissions follow similar patterns, peaking simultaneously with each wave before gradually decreasing.  
- The Île-de-France region consistently shows the highest concentration of hospitalizations and ICU patients, reflecting its high population density and mobility.  
- Some regions, such as Provence-Alpes-Côte d’Azur, Auvergne-Rhône-Alpes, and Grand Est, also show periodic surges depending on the phase of the pandemic.  
- Over time, both the severity and amplitude of epidemic peaks tend to diminish, suggesting the combined effect of vaccination, immunity, and better hospital management.  
- Seasonal or behavioral patterns, such as winter increases and post-holiday rebounds, can also be observed in the data.  
""")

    st.divider()

    st.markdown("## What Drives These Patterns")
    st.markdown("""
- The geographical concentration of severe cases can be linked to population density, urban mobility, and regional healthcare capacity.  
- The Île-de-France area acts as a national hub, both economically and demographically, which explains its persistent overrepresentation in hospital data.  
- The south and coastal regions tend to experience later or milder waves, possibly due to lower population density or better seasonal conditions.  
- The gradual decline in the intensity of waves can be explained by vaccination campaigns, public health measures, and increasing population immunity.  
""")

    st.divider()

    st.markdown("## Caveats and Data Limits")
    st.markdown("""
- The dataset covers only metropolitan France, excluding overseas territories (DROM/COM).  
- There may be reporting delays or missing data on certain dates or regions.  
- No socio-demographic or behavioral variables (such as age, vaccination status, income, or mobility) are included, which limits causal interpretation.  
- Data cleaning involved dropping rows with missing values, which may introduce selection bias toward better-reported periods.  
- Hospital indicators are administrative measures, not direct epidemiological metrics, and may not capture undiagnosed or mild cases.  
""")

    st.divider()

    st.markdown("## Closing Remarks")
    st.markdown("""
This analysis provides a clear picture of how COVID-19 evolved across France from both a temporal and geographical perspective.  
While the Île-de-France region stands out as a consistent epicenter due to its density and connectivity, other regions show episodic surges that reflect regional differences in exposure, timing, and population behavior.  

The visualization highlights how successive waves gradually weakened, a sign of increased resilience through vaccination and adaptation.  
However, the study also underlines the importance of data completeness and contextual interpretation — epidemic dynamics are influenced by multiple external factors not captured in hospital data alone.  

Overall, the dashboard demonstrates how data visualization helps translate complex epidemiological patterns into clear, interpretable insights, supporting more adaptive and regionally nuanced public health responses.
""")


