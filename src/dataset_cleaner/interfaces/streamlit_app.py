#!/usr/bin/env python3
"""
Streamlit GUI for Dataset Cleaner
Interactive web interface for dataset cleaning with visualizations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import json
from dataset_cleaner import DatasetCleaner
from data_analyzer import DataAnalyzer
import zipfile


def create_missing_values_heatmap(df):
    """Create heatmap showing missing values"""
    missing_data = df.isnull()
    
    fig = px.imshow(
        missing_data.values,
        labels=dict(x="Columns", y="Rows", color="Missing"),
        x=df.columns,
        color_continuous_scale=['lightblue', 'red'],
        title="Missing Values Heatmap"
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=400
    )
    
    return fig


def create_outliers_boxplot(df, max_cols=6):
    """Create boxplots for numeric columns to show outliers"""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns[:max_cols]
    
    if len(numeric_cols) == 0:
        return None
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=numeric_cols,
        vertical_spacing=0.1
    )
    
    for i, col in enumerate(numeric_cols):
        row = (i // 3) + 1
        col_pos = (i % 3) + 1
        
        fig.add_trace(
            go.Box(y=df[col], name=col, showlegend=False),
            row=row, col=col_pos
        )
    
    fig.update_layout(
        title="Outliers Detection (Boxplots)",
        height=500
    )
    
    return fig


def create_data_summary_charts(original_df, cleaned_df):
    """Create before/after comparison charts"""
    # Shape comparison
    fig1 = go.Figure(data=[
        go.Bar(name='Original', x=['Rows', 'Columns'], 
               y=[original_df.shape[0], original_df.shape[1]]),
        go.Bar(name='Cleaned', x=['Rows', 'Columns'], 
               y=[cleaned_df.shape[0], cleaned_df.shape[1]])
    ])
    fig1.update_layout(title='Dataset Shape: Before vs After', barmode='group')
    
    # Missing values comparison
    missing_before = original_df.isnull().sum().sum()
    missing_after = cleaned_df.isnull().sum().sum()
    
    fig2 = go.Figure(data=[
        go.Bar(name='Missing Values', x=['Before', 'After'], 
               y=[missing_before, missing_after])
    ])
    fig2.update_layout(title='Missing Values: Before vs After')
    
    return fig1, fig2


def main():
    st.set_page_config(
        page_title="Dataset Cleaner",
        page_icon="🧹",
        layout="wide"
    )
    
    st.title("🧹 Automated Dataset Cleaner")
    st.markdown("Upload your CSV or Excel file and get a cleaned dataset with detailed reports!")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Cleaning Configuration")
    
    missing_threshold = st.sidebar.slider(
        "Missing Values Threshold", 
        min_value=0.1, max_value=1.0, value=0.5, step=0.1,
        help="Drop columns with missing values above this threshold"
    )
    
    outlier_method = st.sidebar.selectbox(
        "Outlier Detection Method",
        options=['iqr', 'zscore'],
        help="Method for detecting outliers"
    )
    
    encoding_method = st.sidebar.selectbox(
        "Categorical Encoding",
        options=['label', 'onehot'],
        help="Method for encoding categorical variables"
    )
    
    normalization_method = st.sidebar.selectbox(
        "Normalization Method",
        options=['minmax', 'standard'],
        help="Method for normalizing numeric data"
    )
    
    # Advanced analysis option
    st.sidebar.header("🔬 Advanced Analysis")
    perform_analysis = st.sidebar.checkbox(
        "Perform Advanced Analysis",
        value=True,
        help="Generate comprehensive data analysis with insights and visualizations"
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your dataset file"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                # Try different encodings for CSV files
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                original_df = None
                for encoding in encodings:
                    try:
                        original_df = pd.read_csv(uploaded_file, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                if original_df is None:
                    st.error(f"Could not decode CSV file with any encoding")
                    return
            else:
                original_df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File loaded successfully! Shape: {original_df.shape}")
            
            # Show original data preview
            st.subheader("📊 Original Dataset Preview")
            st.dataframe(original_df.head(), use_container_width=True)
            
            # Data quality overview
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{original_df.shape[0]:,}")
            with col2:
                st.metric("Columns", original_df.shape[1])
            with col3:
                st.metric("Missing Values", f"{original_df.isnull().sum().sum():,}")
            with col4:
                st.metric("Duplicates", f"{original_df.duplicated().sum():,}")
            
            # Visualizations for original data
            st.subheader("🔍 Data Quality Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Missing values heatmap
                if original_df.isnull().sum().sum() > 0:
                    fig_missing = create_missing_values_heatmap(original_df)
                    st.plotly_chart(fig_missing, use_container_width=True)
                else:
                    st.info("No missing values found in the dataset!")
            
            with col2:
                # Outliers boxplot
                fig_outliers = create_outliers_boxplot(original_df)
                if fig_outliers:
                    st.plotly_chart(fig_outliers, use_container_width=True)
                else:
                    st.info("No numeric columns found for outlier analysis!")
            
            # Clean button
            if st.button("🚀 Clean Dataset", type="primary"):
                with st.spinner("Cleaning dataset... This may take a moment."):
                    # Save uploaded file temporarily
                    temp_file = f"temp_{uploaded_file.name}"
                    with open(temp_file, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    try:
                        # Initialize cleaner and clean data
                        cleaner = DatasetCleaner()
                        
                        # Create organized output folder
                        from pathlib import Path
                        dataset_name = Path(uploaded_file.name).stem
                        output_folder = cleaner.create_output_folder(temp_file)
                        
                        cleaned_df = cleaner.clean_dataset(
                            temp_file,
                            missing_threshold=missing_threshold,
                            outlier_method=outlier_method,
                            encoding_method=encoding_method,
                            normalization_method=normalization_method
                        )
                        
                        # Save cleaned dataset in organized folder
                        cleaned_filename = f"cleaned_{uploaded_file.name}"
                        output_file_path = output_folder / cleaned_filename
                        
                        if output_file_path.suffix == '.csv':
                            cleaned_df.to_csv(output_file_path, index=False)
                        else:
                            cleaned_df.to_excel(output_file_path, index=False)
                        
                        # Generate reports in organized folder
                        cleaner.generate_report(output_folder, dataset_name)
                        
                        # Perform advanced analysis if requested
                        analysis_results = None
                        if perform_analysis:
                            with st.spinner("Performing advanced data analysis..."):
                                try:
                                    analysis_results = cleaner.perform_advanced_analysis(output_folder, dataset_name)
                                    if analysis_results:
                                        st.success("🔬 Advanced analysis completed!")
                                except Exception as e:
                                    st.warning(f"⚠️ Advanced analysis failed: {str(e)}")
                        
                        # Remove temp file
                        import os
                        # Clean up temp file if it exists
                        try:
                            os.remove(temp_file)
                        except (OSError, FileNotFoundError) as e:
                            # File may already be deleted or not accessible
                            pass
                        
                        st.info(f"📂 All files saved in organized folder: {output_folder.name}")
                        
                        st.success("🎉 Dataset cleaned successfully!")
                        
                        # Show cleaned data preview
                        st.subheader("✨ Cleaned Dataset Preview")
                        st.dataframe(cleaned_df.head(), use_container_width=True)
                        
                        # Comparison metrics
                        st.subheader("📈 Before vs After Comparison")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            rows_removed = original_df.shape[0] - cleaned_df.shape[0]
                            st.metric("Rows Removed", f"{rows_removed:,}")
                        with col2:
                            cols_removed = original_df.shape[1] - cleaned_df.shape[1]
                            st.metric("Columns Removed", cols_removed)
                        with col3:
                            missing_cleaned = original_df.isnull().sum().sum() - cleaned_df.isnull().sum().sum()
                            st.metric("Missing Values Cleaned", f"{missing_cleaned:,}")
                        with col4:
                            duplicates_removed = cleaner.report.get('duplicates_removed', 0)
                            st.metric("Duplicates Removed", f"{duplicates_removed:,}")
                        
                        # Comparison charts
                        fig1, fig2 = create_data_summary_charts(original_df, cleaned_df)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(fig1, use_container_width=True)
                        with col2:
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        # Advanced Analysis Results
                        if analysis_results and perform_analysis:
                            st.subheader("🔬 Advanced Data Analysis")
                            
                            # Display key insights
                            insights = analysis_results['analysis_results'].get('insights', [])
                            if insights:
                                st.subheader("💡 Key Insights")
                                for insight in insights[:5]:  # Show top 5 insights
                                    st.info(f"💡 {insight}")
                            
                            # Display quality score
                            if 'data_quality' in analysis_results['analysis_results']:
                                quality_data = analysis_results['analysis_results']['data_quality']
                                quality_score = quality_data['overall_quality_score']
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Data Quality Score", f"{quality_score:.1f}/100")
                                with col2:
                                    completeness = quality_data['completeness_percentage']
                                    st.metric("Data Completeness", f"{completeness:.1f}%")
                                with col3:
                                    consistency_issues = len(quality_data['consistency_issues'])
                                    st.metric("Consistency Issues", consistency_issues)
                            
                            # Show correlation insights
                            if 'correlation_analysis' in analysis_results['analysis_results']:
                                corr_data = analysis_results['analysis_results']['correlation_analysis']
                                strong_corr = corr_data.get('strong_correlations', [])
                                if strong_corr:
                                    st.subheader("🔗 Strong Correlations Found")
                                    for corr in strong_corr[:3]:  # Top 3
                                        st.write(f"**{corr['var1']}** ↔ **{corr['var2']}**: {corr['correlation']:.3f}")
                            
                            # Display visualizations if available
                            viz_folder = analysis_results.get('visualizations_folder')
                            if viz_folder and viz_folder.exists():
                                st.subheader("📊 Analysis Visualizations")
                                
                                # Show available visualizations
                                viz_files = list(viz_folder.glob("*.png"))
                                if viz_files:
                                    selected_viz = st.selectbox(
                                        "Select Visualization",
                                        options=[f.stem.replace('_', ' ').title() for f in viz_files],
                                        key="viz_selector"
                                    )
                                    
                                    # Find corresponding file
                                    selected_file = None
                                    for f in viz_files:
                                        if f.stem.replace('_', ' ').title() == selected_viz:
                                            selected_file = f
                                            break
                                    
                                    if selected_file:
                                        st.image(str(selected_file), use_column_width=True)
                        
                        # Detailed report
                        st.subheader("📋 Detailed Cleaning Report")
                        
                        with st.expander("View Full Cleaning Report"):
                            st.json(cleaner.report)
                        
                        if analysis_results and perform_analysis:
                            with st.expander("View Full Analysis Report"):
                                st.json(analysis_results['analysis_results'])
                        
                        # Download buttons
                        st.subheader("💾 Download Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Download cleaned dataset
                            if uploaded_file.name.endswith('.csv'):
                                csv_data = cleaned_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Cleaned CSV",
                                    data=csv_data,
                                    file_name=f"cleaned_{uploaded_file.name}",
                                    mime="text/csv"
                                )
                            else:
                                # For Excel files
                                buffer = io.BytesIO()
                                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                    cleaned_df.to_excel(writer, index=False)
                                
                                st.download_button(
                                    label="📥 Download Cleaned Excel",
                                    data=buffer.getvalue(),
                                    file_name=f"cleaned_{uploaded_file.name}",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                        
                        with col2:
                            # Download report
                            report_json = json.dumps(cleaner.report, indent=2, default=str)
                            st.download_button(
                                label="📄 Download Cleaning Report",
                                data=report_json,
                                file_name=f"cleaning_report_{uploaded_file.name}.json",
                                mime="application/json"
                            )
                        
                        # Additional download options for analysis
                        if analysis_results and perform_analysis:
                            st.subheader("📊 Download Analysis Results")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                # Download analysis report (Markdown)
                                if analysis_results['report_file'].exists():
                                    with open(analysis_results['report_file'], 'r') as f:
                                        analysis_md = f.read()
                                    st.download_button(
                                        label="📊 Download Analysis Report",
                                        data=analysis_md,
                                        file_name=f"analysis_report_{dataset_name}.md",
                                        mime="text/markdown"
                                    )
                            
                            with col2:
                                # Download analysis JSON
                                if analysis_results['analysis_json'].exists():
                                    with open(analysis_results['analysis_json'], 'r') as f:
                                        analysis_json_data = f.read()
                                    st.download_button(
                                        label="📋 Download Analysis Data",
                                        data=analysis_json_data,
                                        file_name=f"analysis_data_{dataset_name}.json",
                                        mime="application/json"
                                    )
                            
                            with col3:
                                # Create and download visualization package
                                viz_folder = analysis_results.get('visualizations_folder')
                                if viz_folder and viz_folder.exists():
                                    # Create zip file with all visualizations
                                    zip_buffer = io.BytesIO()
                                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                                        for viz_file in viz_folder.glob("*.png"):
                                            zip_file.write(viz_file, viz_file.name)
                                    
                                    st.download_button(
                                        label="📈 Download Visualizations",
                                        data=zip_buffer.getvalue(),
                                        file_name=f"visualizations_{dataset_name}.zip",
                                        mime="application/zip"
                                    )
                    
                    except Exception as e:
                        st.error(f"❌ Error during cleaning: {str(e)}")
                        # Clean up temp file if it exists
                        try:
                            os.remove(temp_file)
                        except (OSError, FileNotFoundError) as e:
                            # File may already be deleted or not accessible
                            pass
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    else:
        st.info("👆 Please upload a CSV or Excel file to get started!")
        
        # Show example usage
        st.subheader("🚀 How to Use")
        st.markdown("""
        1. **Upload** your CSV or Excel file using the file uploader above
        2. **Configure** cleaning parameters in the sidebar
        3. **Preview** your data and quality analysis
        4. **Click** the "Clean Dataset" button to start the cleaning process
        5. **Download** your cleaned dataset and detailed report
        
        ### Features:
        - 🧹 **Automatic cleaning**: Missing values, duplicates, outliers
        - 🏷️ **Smart encoding**: Categorical variables handling
        - ⚖️ **Normalization**: Scale your numeric data
        - 📊 **Visualizations**: Before/after comparisons
        - 📋 **Detailed reports**: Complete cleaning summary
        """)


if __name__ == "__main__":
    main()