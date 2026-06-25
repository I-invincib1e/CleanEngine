#!/usr/bin/env python3
"""
CleanEngine — Streamlit GUI
Multi-feature data cleaning, drift monitoring, anomaly explanation, and recipe export.
"""

import io
import json
import textwrap
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dataset_cleaner.analysis.analyzer import DataAnalyzer
from dataset_cleaner.core.cleaner import DatasetCleaner
from dataset_cleaner.interfaces.anomaly_explainer import explain_anomalies
from dataset_cleaner.interfaces.domain_templates import list_templates, get_template
from dataset_cleaner.interfaces.drift_monitor import compare_datasets
from dataset_cleaner.interfaces.recipe_exporter import (
    generate_python_recipe,
    generate_sql_recipe,
    generate_yaml_config,
)


# ─────────────────────────────── Helpers ────────────────────────────────────

def load_uploaded_file(uploaded_file) -> pd.DataFrame | None:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        for enc in ["utf-8", "latin-1", "iso-8859-1", "cp1252"]:
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding=enc)
            except UnicodeDecodeError:
                continue
        st.error("Could not decode CSV file with any supported encoding.")
        return None
    elif name.endswith((".xlsx", ".xls")):
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file)
    elif name.endswith(".json"):
        uploaded_file.seek(0)
        return pd.read_json(uploaded_file)
    elif name.endswith(".parquet"):
        uploaded_file.seek(0)
        return pd.read_parquet(uploaded_file)
    st.error(f"Unsupported file type: {uploaded_file.name}")
    return None


def create_missing_heatmap(df: pd.DataFrame):
    missing_data = df.isnull()
    fig = px.imshow(
        missing_data.values,
        labels=dict(x="Columns", y="Rows", color="Missing"),
        x=df.columns.tolist(),
        color_continuous_scale=["#e8f4f8", "#e74c3c"],
        title="Missing Values Heatmap",
    )
    fig.update_layout(xaxis_tickangle=-45, height=380, margin=dict(t=40, b=0))
    return fig


def create_outliers_boxplot(df: pd.DataFrame, max_cols: int = 6):
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns[:max_cols]
    if len(numeric_cols) == 0:
        return None
    rows_n = (len(numeric_cols) + 2) // 3
    fig = make_subplots(
        rows=rows_n, cols=3,
        subplot_titles=list(numeric_cols),
        vertical_spacing=0.12,
    )
    for i, col in enumerate(numeric_cols):
        fig.add_trace(
            go.Box(y=df[col], name=col, showlegend=False, marker_color="#3498db"),
            row=(i // 3) + 1, col=(i % 3) + 1,
        )
    fig.update_layout(title="Outlier Detection — Boxplots", height=420, margin=dict(t=50, b=0))
    return fig


def create_comparison_charts(original_df: pd.DataFrame, cleaned_df: pd.DataFrame):
    fig1 = go.Figure(data=[
        go.Bar(name="Original", x=["Rows", "Columns"],
               y=[original_df.shape[0], original_df.shape[1]], marker_color="#3498db"),
        go.Bar(name="Cleaned", x=["Rows", "Columns"],
               y=[cleaned_df.shape[0], cleaned_df.shape[1]], marker_color="#2ecc71"),
    ])
    fig1.update_layout(title="Dataset Shape: Before vs After", barmode="group", height=340)

    fig2 = go.Figure(data=[go.Bar(
        x=["Before", "After"],
        y=[original_df.isnull().sum().sum(), cleaned_df.isnull().sum().sum()],
        marker_color=["#e74c3c", "#2ecc71"],
    )])
    fig2.update_layout(title="Missing Values: Before vs After", height=340)
    return fig1, fig2


def severity_badge(severity: str) -> str:
    colors = {"none": "🟢", "low": "🟡", "medium": "🟠", "high": "🔴"}
    return colors.get(severity, "⚪")


# ──────────────────────────── Tab: Clean ────────────────────────────────────

def tab_clean():
    st.markdown("Upload your dataset, choose a domain template or configure manually, then clean and export.")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.header("🗂️ Domain Template")
    template_options = {t.icon + " " + t.name: k for k, t in list_templates()}
    chosen_label = st.sidebar.selectbox("Choose a template", list(template_options.keys()))
    chosen_key = template_options[chosen_label]
    tmpl = get_template(chosen_key)

    if chosen_key != "general":
        with st.sidebar.expander("ℹ️ Template Tips", expanded=False):
            for tip in tmpl.tips:
                st.markdown(f"• {tip}")

    st.sidebar.divider()
    st.sidebar.header("⚙️ Cleaning Config")

    missing_threshold = st.sidebar.slider(
        "Missing Values Threshold",
        min_value=0.05, max_value=1.0,
        value=tmpl.missing_threshold, step=0.05,
        help="Drop columns with more missing values than this fraction",
    )
    outlier_method = st.sidebar.selectbox(
        "Outlier Detection Method",
        options=["iqr", "zscore"],
        index=["iqr", "zscore"].index(tmpl.outlier_method),
    )
    encoding_method = st.sidebar.selectbox(
        "Categorical Encoding",
        options=["label", "onehot"],
        index=["label", "onehot"].index(tmpl.encoding_method),
    )
    normalization_method = st.sidebar.selectbox(
        "Normalization Method",
        options=["minmax", "standard"],
        index=["minmax", "standard"].index(tmpl.normalization_method),
    )

    st.sidebar.divider()
    st.sidebar.header("🔬 Advanced Analysis")
    perform_analysis = st.sidebar.checkbox("Run Advanced Analysis", value=True)

    # ── File upload ───────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "📂 Upload your dataset",
        type=["csv", "xlsx", "xls", "json", "parquet"],
        help="CSV, Excel, JSON, or Parquet",
    )

    if uploaded_file is None:
        st.info("👆 Upload a dataset to get started.")
        _render_landing_features()
        return

    original_df = load_uploaded_file(uploaded_file)
    if original_df is None:
        return

    st.success(f"✅ **{uploaded_file.name}** loaded — {original_df.shape[0]:,} rows × {original_df.shape[1]} columns")

    # ── Domain column hints ───────────────────────────────────────────────────
    if tmpl.column_hints:
        matched = {col: hint for pattern, hint in tmpl.column_hints.items()
                   for col in original_df.columns
                   if pattern.rstrip("*").lower() in col.lower()}
        if matched:
            with st.expander(f"{tmpl.icon} Column hints from {tmpl.name} template"):
                for col, hint in matched.items():
                    st.markdown(f"**`{col}`** — {hint}")

    # ── Data preview ──────────────────────────────────────────────────────────
    with st.expander("📊 Dataset Preview", expanded=True):
        st.dataframe(original_df.head(10), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{original_df.shape[0]:,}")
    c2.metric("Columns", original_df.shape[1])
    c3.metric("Missing Values", f"{original_df.isnull().sum().sum():,}")
    c4.metric("Duplicates", f"{original_df.duplicated().sum():,}")

    # ── Quality visualizations ────────────────────────────────────────────────
    st.subheader("🔍 Data Quality Overview")
    col1, col2 = st.columns(2)
    with col1:
        if original_df.isnull().sum().sum() > 0:
            st.plotly_chart(create_missing_heatmap(original_df), use_container_width=True)
        else:
            st.success("✅ No missing values!")
    with col2:
        fig_box = create_outliers_boxplot(original_df)
        if fig_box:
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("No numeric columns for outlier preview.")

    # ── Clean button ──────────────────────────────────────────────────────────
    if not st.button("🚀 Clean Dataset", type="primary", use_container_width=True):
        return

    with st.spinner("Cleaning dataset…"):
        import os, tempfile
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = tmp.name

        try:
            cleaner = DatasetCleaner()
            dataset_name = Path(uploaded_file.name).stem
            output_folder = cleaner.create_output_folder(temp_path)

            cleaned_df = cleaner.clean_dataset(
                temp_path,
                missing_threshold=missing_threshold,
                outlier_method=outlier_method,
                encoding_method=encoding_method,
                normalization_method=normalization_method,
            )
            cleaner.generate_report(output_folder, dataset_name)
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    st.success("🎉 Dataset cleaned successfully!")

    # ── Results ───────────────────────────────────────────────────────────────
    with st.expander("✨ Cleaned Dataset Preview", expanded=True):
        st.dataframe(cleaned_df.head(10), use_container_width=True)

    st.subheader("📈 Before vs After")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows Removed", f"{original_df.shape[0] - cleaned_df.shape[0]:,}")
    c2.metric("Columns Removed", original_df.shape[1] - cleaned_df.shape[1])
    c3.metric(
        "Missing Cleaned",
        f"{original_df.isnull().sum().sum() - cleaned_df.isnull().sum().sum():,}",
    )
    c4.metric("Duplicates Removed", f"{cleaner.report.get('duplicates_removed', 0):,}")

    f1, f2 = create_comparison_charts(original_df, cleaned_df)
    col1, col2 = st.columns(2)
    col1.plotly_chart(f1, use_container_width=True)
    col2.plotly_chart(f2, use_container_width=True)

    # ── Advanced analysis ─────────────────────────────────────────────────────
    analysis_results = None
    if perform_analysis:
        with st.spinner("Running advanced analysis…"):
            try:
                analysis_results = cleaner.perform_advanced_analysis(output_folder, dataset_name)
                if analysis_results:
                    st.success("🔬 Advanced analysis complete!")
            except Exception as e:
                st.warning(f"⚠️ Advanced analysis failed: {e}")

    if analysis_results:
        _render_analysis_results(analysis_results, dataset_name)

    # ── Cleaning report ───────────────────────────────────────────────────────
    with st.expander("📋 Full Cleaning Report"):
        st.json(cleaner.report)

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.subheader("💾 Download Results")
    _render_download_section(uploaded_file, cleaned_df, cleaner, analysis_results, dataset_name)

    # ── RECIPE EXPORT ─────────────────────────────────────────────────────────
    st.subheader("🧾 Cleaning Recipe Export")
    st.markdown(
        "Turn your cleaning steps into **production-ready code** you can run anywhere — "
        "no GUI needed."
    )
    recipe_tabs = st.tabs(["🐍 Python Script", "🗄️ SQL Query", "📄 YAML Config"])

    with recipe_tabs[0]:
        python_code = generate_python_recipe(cleaner.report, uploaded_file.name)
        st.code(python_code, language="python")
        st.download_button(
            "⬇️ Download Python Script",
            data=python_code,
            file_name=f"clean_{dataset_name}.py",
            mime="text/x-python",
            use_container_width=True,
        )

    with recipe_tabs[1]:
        sql_code = generate_sql_recipe(cleaner.report, dataset_name)
        st.code(sql_code, language="sql")
        st.download_button(
            "⬇️ Download SQL Script",
            data=sql_code,
            file_name=f"clean_{dataset_name}.sql",
            mime="text/plain",
            use_container_width=True,
        )

    with recipe_tabs[2]:
        yaml_str = generate_yaml_config(
            missing_threshold, outlier_method, encoding_method, normalization_method
        )
        st.code(yaml_str, language="yaml")
        st.info(
            "💡 Share this `.yaml` file with your team to apply identical cleaning settings "
            "to any future dataset — just load it into CleanEngine."
        )
        st.download_button(
            "⬇️ Download YAML Config",
            data=yaml_str,
            file_name=f"cleanengine_{dataset_name}.yaml",
            mime="text/yaml",
            use_container_width=True,
        )


def _render_analysis_results(analysis_results: dict, dataset_name: str):
    st.subheader("🔬 Advanced Analysis")
    ar = analysis_results.get("analysis_results", {})

    insights = ar.get("insights", [])
    if insights:
        st.markdown("**💡 Key Insights**")
        for insight in insights[:6]:
            st.info(f"💡 {insight}")

    if "data_quality" in ar:
        q = ar["data_quality"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Quality Score", f"{q['overall_quality_score']:.1f}/100")
        c2.metric("Completeness", f"{q['completeness_percentage']:.1f}%")
        c3.metric("Consistency Issues", len(q["consistency_issues"]))

    corr_data = ar.get("correlation_analysis", {})
    strong_corr = corr_data.get("strong_correlations", [])
    if strong_corr:
        st.markdown("**🔗 Strong Correlations**")
        cols = st.columns(min(3, len(strong_corr)))
        for i, corr in enumerate(strong_corr[:3]):
            cols[i].metric(
                f"{corr['var1']} ↔ {corr['var2']}",
                f"{corr['correlation']:.3f}",
            )

    viz_folder = analysis_results.get("visualizations_folder")
    if viz_folder and viz_folder.exists():
        viz_files = list(viz_folder.glob("*.png"))
        if viz_files:
            st.markdown("**📊 Visualizations**")
            selected_name = st.selectbox(
                "Select chart",
                [f.stem.replace("_", " ").title() for f in viz_files],
            )
            for f in viz_files:
                if f.stem.replace("_", " ").title() == selected_name:
                    st.image(str(f), use_column_width=True)
                    break


def _render_download_section(uploaded_file, cleaned_df, cleaner, analysis_results, dataset_name):
    col1, col2, col3 = st.columns(3)

    with col1:
        if uploaded_file.name.lower().endswith(".csv"):
            st.download_button(
                "📥 Cleaned CSV",
                data=cleaned_df.to_csv(index=False),
                file_name=f"cleaned_{uploaded_file.name}",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                cleaned_df.to_excel(writer, index=False)
            st.download_button(
                "📥 Cleaned Excel",
                data=buf.getvalue(),
                file_name=f"cleaned_{uploaded_file.name}",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    with col2:
        st.download_button(
            "📄 Cleaning Report (JSON)",
            data=json.dumps(cleaner.report, indent=2, default=str),
            file_name=f"report_{dataset_name}.json",
            mime="application/json",
            use_container_width=True,
        )

    with col3:
        if analysis_results:
            viz_folder = analysis_results.get("visualizations_folder")
            if viz_folder and viz_folder.exists():
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w") as zf:
                    for f in viz_folder.glob("*.png"):
                        zf.write(f, f.name)
                st.download_button(
                    "📈 Visualizations (ZIP)",
                    data=zip_buf.getvalue(),
                    file_name=f"viz_{dataset_name}.zip",
                    mime="application/zip",
                    use_container_width=True,
                )


def _render_landing_features():
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("### 🏥 Domain Templates\nHealthcare, Finance, E-commerce, IoT, Survey — pre-tuned cleaning configs.")
    c2.markdown("### 🧾 Recipe Export\nAuto-generate Python scripts, SQL queries, and YAML configs from your cleaning steps.")
    c3.markdown("### 📡 Drift Monitor\nCompare two versions of a dataset and detect statistical drift per column.")
    c4.markdown("### 🔍 Anomaly Explorer\nDetect anomalous rows and get a plain-English explanation of *why* each one is unusual.")


# ──────────────────────────── Tab: Drift Monitor ────────────────────────────

def tab_drift():
    st.markdown(
        "Upload two versions of the same dataset (e.g. last month vs this month) "
        "to detect schema changes, missing-value shifts, and distribution drift per column."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📁 Reference Dataset (baseline)")
        ref_file = st.file_uploader(
            "Upload reference (older) dataset",
            type=["csv", "xlsx", "xls", "json", "parquet"],
            key="drift_ref",
        )
    with col2:
        st.markdown("### 📁 New Dataset (latest)")
        new_file = st.file_uploader(
            "Upload new dataset to compare",
            type=["csv", "xlsx", "xls", "json", "parquet"],
            key="drift_new",
        )

    if ref_file is None or new_file is None:
        st.info("👆 Upload both datasets above to run the drift analysis.")
        with st.expander("ℹ️ How drift detection works"):
            st.markdown(
                """
**Numeric columns** — Kolmogorov-Smirnov test (p < 0.05 = drifted)  
**Categorical columns** — Chi-squared test (p < 0.05 = drifted)  
**Missing values** — Delta > 5 percentage points flagged  
**Schema** — Added / removed columns detected automatically
                """
            )
        return

    df_ref = load_uploaded_file(ref_file)
    df_new = load_uploaded_file(new_file)
    if df_ref is None or df_new is None:
        return

    st.success(
        f"Reference: **{ref_file.name}** ({len(df_ref):,} rows) · "
        f"New: **{new_file.name}** ({len(df_new):,} rows)"
    )

    with st.spinner("Comparing datasets…"):
        report = compare_datasets(df_ref, df_new)

    summary = report["summary"]

    # ── Summary banner ────────────────────────────────────────────────────────
    severity = summary["severity"]
    badge = severity_badge(severity)
    st.markdown(f"## {badge} Drift Severity: **{severity.upper()}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Columns Drifted", f"{summary['columns_with_drift']} / {summary['total_columns_compared']}")
    c2.metric("Drift Rate", f"{summary['drift_rate_pct']:.1f}%")
    c3.metric("Row Count Change", f"{summary['row_count_change']:+,}")
    c4.metric("Row Change %", f"{summary['row_count_change_pct']:+.1f}%")

    # ── Schema changes ────────────────────────────────────────────────────────
    schema = report["schema_changes"]
    if schema["added_columns"] or schema["removed_columns"]:
        st.subheader("🗂️ Schema Changes")
        sc1, sc2 = st.columns(2)
        with sc1:
            if schema["added_columns"]:
                st.error(f"➕ New columns in latest dataset ({len(schema['added_columns'])}):")
                for col in schema["added_columns"]:
                    st.markdown(f"  - `{col}`")
            else:
                st.success("No new columns added.")
        with sc2:
            if schema["removed_columns"]:
                st.error(f"➖ Columns removed from latest dataset ({len(schema['removed_columns'])}):")
                for col in schema["removed_columns"]:
                    st.markdown(f"  - `{col}`")
            else:
                st.success("No columns removed.")

    # ── Drifted columns ───────────────────────────────────────────────────────
    st.subheader("📊 Column-Level Drift Analysis")

    drifted = summary["drifted_column_names"]
    if drifted:
        st.warning(f"⚠️ {len(drifted)} column(s) show statistically significant drift:")
        _render_drift_table(report, drifted, drifted_only=True)
    else:
        st.success("✅ No statistically significant distribution drift detected.")

    with st.expander("📋 All Columns (including stable ones)"):
        all_cols = schema["common_columns"]
        _render_drift_table(report, all_cols, drifted_only=False)

    # ── Missing value drift ───────────────────────────────────────────────────
    missing_changes = summary["columns_with_missing_changes"]
    if missing_changes:
        st.subheader("🕳️ Missing Value Shifts (> 5pp)")
        rows = []
        for col in missing_changes:
            md = report["missing_drift"][col]
            rows.append({
                "Column": col,
                "Reference %": md["reference_pct"],
                "New %": md["new_pct"],
                "Delta pp": md["delta_pct"],
            })
        mdf = pd.DataFrame(rows).sort_values("Delta pp", key=abs, ascending=False)
        st.dataframe(mdf, use_container_width=True, hide_index=True)

    # ── New categories ────────────────────────────────────────────────────────
    new_cats = report.get("new_categories", {})
    if new_cats:
        st.subheader("🆕 New Categories in Latest Dataset")
        for col, cats in new_cats.items():
            st.markdown(f"**`{col}`**: {', '.join(f'`{c}`' for c in cats[:10])}"
                        + (f" (+{len(cats)-10} more)" if len(cats) > 10 else ""))

    # ── Download report ───────────────────────────────────────────────────────
    st.divider()
    st.download_button(
        "⬇️ Download Full Drift Report (JSON)",
        data=json.dumps(report, indent=2, default=str),
        file_name=f"drift_report_{ref_file.name}_vs_{new_file.name}.json",
        mime="application/json",
        use_container_width=True,
    )


def _render_drift_table(report: dict, columns: list, drifted_only: bool):
    rows = []
    drifted_set = set(report["summary"]["drifted_column_names"])
    for col in columns:
        if drifted_only and col not in drifted_set:
            continue
        dist = report["distribution_drift"].get(col, {})
        miss = report["missing_drift"].get(col, {})
        test = dist.get("statistical_test", {})
        p = test.get("p_value")
        drifted = test.get("drifted", False)

        row = {
            "Column": col,
            "Type": dist.get("type", "—"),
            "Drifted": "🔴 Yes" if drifted else "🟢 No",
            "p-value": f"{p:.4f}" if p is not None else "—",
            "Missing Δ": f"{miss.get('delta_pct', 0):+.1f}pp",
        }

        if dist.get("type") == "numeric":
            ref_stats = dist.get("reference_stats", {})
            new_stats = dist.get("new_stats", {})
            mean_delta = dist.get("mean_delta")
            row["Ref Mean"] = f"{ref_stats.get('mean', '—'):.4g}" if ref_stats else "—"
            row["New Mean"] = f"{new_stats.get('mean', '—'):.4g}" if new_stats else "—"
            row["Mean Δ"] = f"{mean_delta:+.4g}" if mean_delta is not None else "—"
        else:
            ref_u = dist.get("reference_stats", {}).get("unique_values", "—")
            new_u = dist.get("new_stats", {}).get("unique_values", "—")
            row["Ref Unique"] = ref_u
            row["New Unique"] = new_u
            row["New Categories"] = len(dist.get("new_categories", []))

        rows.append(row)

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No columns to display.")


# ────────────────────────── Tab: Anomaly Explorer ───────────────────────────

def tab_anomalies():
    st.markdown(
        "Upload a dataset to detect anomalous rows using **Isolation Forest**, "
        "then get a plain-English explanation of *exactly why* each row is unusual."
    )

    uploaded_file = st.file_uploader(
        "📂 Upload dataset for anomaly analysis",
        type=["csv", "xlsx", "xls", "json", "parquet"],
        key="anomaly_upload",
    )

    col1, col2 = st.columns(2)
    with col1:
        contamination = st.slider(
            "Expected anomaly rate (%)",
            min_value=1, max_value=25, value=5, step=1,
            help="What fraction of rows do you expect to be anomalous?",
        ) / 100
    with col2:
        max_display = st.slider(
            "Max anomalies to explain",
            min_value=5, max_value=100, value=20, step=5,
        )

    if uploaded_file is None:
        st.info("👆 Upload a dataset to begin anomaly analysis.")
        with st.expander("ℹ️ How anomaly explanation works"):
            st.markdown(
                """
**Detection:** Isolation Forest identifies rows that are easiest to "isolate" — 
they sit far from clusters of normal data.

**Explanation:** For each anomaly, CleanEngine checks every numeric column and 
identifies which values deviate most from the norm using z-scores and IQR fences.

**Output:** Each row gets a plain-English summary like:
> *"Row 47 is anomalous because: `salary` = 2,400,000 (unusually high, 99th pct); 
> `age` = 22 (unusually low, 4th pct); `tenure_years` = 18 (unusually high, 97th pct)."*
                """
            )
        return

    df = load_uploaded_file(uploaded_file)
    if df is None:
        return

    st.success(f"✅ Loaded {len(df):,} rows × {df.shape[1]} columns")

    with st.spinner(f"Detecting anomalies (contamination={contamination:.0%})…"):
        results = explain_anomalies(df, contamination=contamination, max_anomalies=max_display)

    if "error" in results:
        st.error(results["error"])
        return

    summary = results["summary"]

    # ── Summary metrics ───────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Anomalies", f"{summary['total_anomalies']:,}")
    c2.metric("Anomaly Rate", f"{summary['anomaly_rate_pct']:.1f}%")
    c3.metric("Total Rows", f"{summary['total_rows']:,}")
    c4.metric("Columns Analyzed", len(summary["columns_analyzed"]))

    anomaly_rows = results["anomaly_rows"]
    if not anomaly_rows:
        st.success("✅ No anomalies detected at the current contamination rate.")
        return

    # ── Anomaly score distribution ────────────────────────────────────────────
    all_scores = [r["anomaly_score"] for r in anomaly_rows]
    fig_scores = px.histogram(
        x=all_scores, nbins=20,
        title="Anomaly Score Distribution (more negative = more anomalous)",
        labels={"x": "Anomaly Score", "y": "Count"},
        color_discrete_sequence=["#e74c3c"],
    )
    fig_scores.update_layout(height=300, margin=dict(t=50, b=0))
    st.plotly_chart(fig_scores, use_container_width=True)

    # ── Per-column anomaly frequency ──────────────────────────────────────────
    col_freq: dict[str, int] = {}
    for row in anomaly_rows:
        for dev in row["feature_deviations"]:
            col_freq[dev["column"]] = col_freq.get(dev["column"], 0) + 1

    if col_freq:
        freq_df = pd.DataFrame(
            sorted(col_freq.items(), key=lambda x: -x[1]),
            columns=["Column", "Anomaly Contribution"],
        )
        fig_freq = px.bar(
            freq_df, x="Column", y="Anomaly Contribution",
            title="Which Columns Drive the Most Anomalies",
            color="Anomaly Contribution",
            color_continuous_scale="Reds",
        )
        fig_freq.update_layout(height=300, showlegend=False, margin=dict(t=50, b=0))
        st.plotly_chart(fig_freq, use_container_width=True)

    # ── Individual anomaly cards ──────────────────────────────────────────────
    st.subheader(f"🔍 Top {len(anomaly_rows)} Anomalies — Explained")

    for i, row_info in enumerate(anomaly_rows):
        with st.expander(
            f"**Row {row_info['row_index']}** — Score: {row_info['anomaly_score']:.3f}  "
            f"| {len(row_info['feature_deviations'])} deviating features",
            expanded=(i < 3),
        ):
            st.markdown(f"**📌 Summary:** {row_info['summary']}")
            st.divider()

            if row_info["feature_deviations"]:
                st.markdown("**Feature Deviations (worst first):**")
                dev_rows = []
                for dev in row_info["feature_deviations"]:
                    dev_rows.append({
                        "Column": dev["column"],
                        "Value": dev["value"],
                        "Z-Score": dev["z_score"],
                        "Percentile": f"{dev['percentile']:.1f}th",
                        "Direction": "⬆️ High" if dev["direction"] == "high" else "⬇️ Low",
                        "Outside IQR": "⚠️ Yes" if (dev["below_iqr_lower"] or dev["above_iqr_upper"]) else "No",
                        "Detail": dev["explanation"],
                    })
                st.dataframe(pd.DataFrame(dev_rows), use_container_width=True, hide_index=True)
            else:
                st.info("This row is anomalous due to an unusual *combination* of values rather than a single extreme feature.")

            with st.expander("📋 Full row data"):
                st.json(row_info["row_data"])

    # ── Download ──────────────────────────────────────────────────────────────
    st.divider()
    anomaly_export = {
        "summary": summary,
        "anomalies": [
            {
                "row_index": r["row_index"],
                "anomaly_score": r["anomaly_score"],
                "summary": r["summary"],
                "feature_deviations": r["feature_deviations"],
            }
            for r in anomaly_rows
        ],
    }

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download Anomaly Report (JSON)",
            data=json.dumps(anomaly_export, indent=2, default=str),
            file_name=f"anomalies_{uploaded_file.name}.json",
            mime="application/json",
            use_container_width=True,
        )
    with col2:
        anomaly_indices = [r["row_index"] for r in anomaly_rows]
        anomaly_df = df.iloc[[i for i in range(len(df)) if df.index[i] in anomaly_indices]]
        st.download_button(
            "⬇️ Download Anomalous Rows (CSV)",
            data=anomaly_df.to_csv(index=True),
            file_name=f"anomalous_rows_{uploaded_file.name}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ────────────────────────────── Main app ────────────────────────────────────

def main():
    st.set_page_config(
        page_title="CleanEngine",
        page_icon="🧹",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .stTabs [data-baseweb="tab"] { font-size: 1rem; padding: 0.6rem 1.2rem; }
        .stMetric { background: #f8f9fa; border-radius: 8px; padding: 0.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🧹 CleanEngine")
    st.caption("The Ultimate Data Cleaning & Analysis Toolkit")

    tab1, tab2, tab3 = st.tabs([
        "🧹 Clean & Export",
        "📡 Drift Monitor",
        "🔍 Anomaly Explorer",
    ])

    with tab1:
        tab_clean()
    with tab2:
        tab_drift()
    with tab3:
        tab_anomalies()


if __name__ == "__main__":
    main()
