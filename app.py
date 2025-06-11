import streamlit as st
import pandas as pd
from datetime import datetime
import base64

# Import the cleaner class from your original script
from main import MarketResearchDataCleaner

# Page configuration
st.set_page_config(
    page_title="MESH Experience Data Cleaner", page_icon="🧹", layout="wide"
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap');
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #F7F4EB !important;
        font-family: 'Montserrat', sans-serif !important;
    }
    [data-testid="stSidebar"] {
        background-color: #00F5D4 !important;
    }
    *, .stText, .stMarkdown, .stButton>button, .stMetric, .stDataFrame, .stDownloadButton, .stHeader, .stSubheader {
        font-family: 'Montserrat', sans-serif !important;
    }
    [data-testid="stAlert"] {
        color: #111 !important;                /* Black text */
    }
    [data-testid="stAlert"] * {
        color: #111 !important;                /* Ensure all text inside is black */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
if "report" not in st.session_state:
    st.session_state.report = None
if "api_key" not in st.session_state:
    st.session_state.api_key = None


# Helper functions
def get_download_link(df, filename, text):
    """Generate a download link for a dataframe."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode("utf-8")).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'


def display_quality_metrics(scored_df):
    """Display quality metrics in a nice format."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        high_quality = len(scored_df[scored_df["overall_score"] >= 70])
        total = len(scored_df)
        st.metric(
            "High Quality",
            f"{high_quality}/{total}",
            f"{high_quality / total * 100:.1f}%",
        )

    with col2:
        medium_quality = len(
            scored_df[
                (scored_df["overall_score"] >= 30) & (scored_df["overall_score"] < 70)
            ]
        )
        st.metric(
            "Medium Quality",
            f"{medium_quality}/{total}",
            f"{medium_quality / total * 100:.1f}%",
        )

    with col3:
        low_quality = len(scored_df[scored_df["overall_score"] < 30])
        st.metric(
            "Low Quality", f"{low_quality}/{total}", f"{low_quality / total * 100:.1f}%"
        )

    with col4:
        avg_score = scored_df["overall_score"].mean()
        st.metric("Average Score", f"{avg_score:.1f}/100")


# Main app
def main():
    # Header
    st.title("🧹 Experience Data Cleaner")
    st.markdown("AI-powered data cleaning for Experience Data")

    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")

        # API Key input
        api_key_input = st.text_input(
            "Anthropic API Key",
            type="password",
            value=st.session_state.api_key if st.session_state.api_key else "",
            help="Enter your Anthropic API key. Get one at https://console.anthropic.com/",
        )

        if api_key_input:
            st.session_state.api_key = api_key_input

        # Processing options
        st.subheader("Processing Options")

        use_claude = st.checkbox(
            "Use Claude AI Analysis",
            value=True,
            help="Uncheck for faster processing using only basic checks",
        )

        sample_size = st.number_input(
            "Sample Size (0 = all rows)",
            min_value=0,
            max_value=10000,
            value=0,
            help="Process a sample of rows for testing. Set to 0 to process all rows.",
        )

        st.markdown("---")

        # Info box
        st.info("""
        **What this tool checks:**
        - Brand & touchpoint matching
        - Gibberish detection
        - No experience detection
        - Overall response quality
        """)

    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📤 Upload & Process",
            "📊 Results Overview",
            "🔍 Detailed Results",
            "📋 Report",
        ]
    )

    with tab1:
        st.header("Upload Your Data")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload your market research data in CSV format",
        )

        if uploaded_file is not None:
            # Display file info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.2f} KB",
                "File type": uploaded_file.type,
            }

            col1, col2 = st.columns([2, 3])
            with col1:
                st.write("**File Details:**")
                for key, value in file_details.items():
                    st.write(f"- {key}: {value}")

            # Load and preview data
            try:
                df = pd.read_csv(uploaded_file)

                with col2:
                    st.write("**Data Preview:**")
                    st.write(f"- Rows: {len(df)}")
                    st.write(f"- Columns: {len(df.columns)}")

                    # Check for required columns
                    required_cols = ["brand", "touchpoint", "text"]
                    missing_cols = [
                        col for col in required_cols if col not in df.columns
                    ]

                    if missing_cols:
                        st.error(
                            f"❌ Missing required columns: {', '.join(missing_cols)}"
                        )
                    else:
                        st.success("✅ All required columns found!")

                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(df.head(), use_container_width=True)

                # Process button
                if not missing_cols:
                    st.markdown("---")

                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("🚀 Start Processing", use_container_width=True):
                            if not st.session_state.api_key and use_claude:
                                st.error(
                                    "⚠️ Please enter your Anthropic API key in the sidebar to use Claude AI analysis."
                                )
                            else:
                                # Process the data
                                with st.spinner(
                                    "Processing your data... This may take a few minutes. Please do not refresh the page."
                                ):
                                    try:
                                        # Initialize cleaner
                                        cleaner = MarketResearchDataCleaner(
                                            st.session_state.api_key
                                            if st.session_state.api_key
                                            else "dummy-key"
                                        )

                                        # Create progress bar
                                        progress_bar = st.progress(0)
                                        status_text = st.empty()

                                        # Process dataset
                                        actual_sample_size = (
                                            sample_size if sample_size > 0 else None
                                        )

                                        # For progress tracking, we'll need to modify the process_dataset method
                                        # For now, we'll just show a simple progress
                                        status_text.text("Starting processing...")
                                        progress_bar.progress(10)

                                        scored_df = cleaner.process_dataset(
                                            df,
                                            use_claude=use_claude
                                            and bool(st.session_state.api_key),
                                            sample_size=actual_sample_size,
                                            debug=False,
                                        )

                                        progress_bar.progress(80)
                                        status_text.text("Generating report...")

                                        # Merge with original data
                                        columns_to_drop = [
                                            "brand",
                                            "touchpoint",
                                            "story_preview",
                                        ]
                                        scored_df_clean = scored_df.drop(
                                            columns=[
                                                col
                                                for col in columns_to_drop
                                                if col in scored_df.columns
                                            ],
                                            errors="ignore",
                                        )

                                        final_df = df.merge(
                                            scored_df_clean,
                                            left_index=True,
                                            right_on="row_index",
                                            how="inner",
                                            suffixes=("", "_score"),
                                        )

                                        # Sort by score
                                        final_df = final_df.sort_values(
                                            "overall_score", ascending=True
                                        )

                                        # Generate report
                                        report = cleaner.generate_report(scored_df)

                                        # Store in session state
                                        st.session_state.processed_data = final_df
                                        st.session_state.report = report
                                        st.session_state.scored_df = scored_df

                                        progress_bar.progress(100)
                                        status_text.text("Processing complete!")

                                        st.success(
                                            "✅ Processing complete! Check the other tabs for results."
                                        )

                                        # Quick summary
                                        st.markdown("### Quick Summary")
                                        display_quality_metrics(scored_df)

                                        # After processing is complete and summary is shown
                                        if st.session_state.processed_data is not None:
                                            full_csv = (
                                                st.session_state.processed_data.to_csv(
                                                    index=False, encoding="utf-8-sig"
                                                )
                                            )
                                            st.download_button(
                                                label="📥 Download All Results (CSV)",
                                                data=full_csv,
                                                file_name=f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                mime="text/csv",
                                                key="download_all_results_tab1",
                                            )

                                    except Exception as e:
                                        st.error(f"❌ Error processing data: {str(e)}")
                                        st.exception(e)

            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")

    with tab2:
        st.header("Results Overview")

        if st.session_state.processed_data is not None:
            scored_df = st.session_state.scored_df

            # Quality metrics
            st.subheader("📊 Quality Distribution")
            display_quality_metrics(scored_df)

            # Charts
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🎯 Brand Matching")
                total = len(scored_df)
                correct_brand = (scored_df["brand_match_score"] == 100).sum()
                percent_brand = (correct_brand / total * 100) if total > 0 else 0
                st.metric(
                    "Correct Brand", f"{correct_brand}/{total}", f"{percent_brand:.1f}%"
                )

            with col2:
                st.subheader("📍 Touchpoint Matching")
                correct_tp = (scored_df["touchpoint_match_score"] == 100).sum()
                percent_tp = (correct_tp / total * 100) if total > 0 else 0
                st.metric(
                    "Correct Touchpoint", f"{correct_tp}/{total}", f"{percent_tp:.1f}%"
                )

            # Issues summary
            st.subheader("⚠️ Issues Found")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                profanity_count = scored_df["has_profanity"].sum()
                st.metric(
                    "Profanity",
                    profanity_count,
                    "rows" if profanity_count != 1 else "row",
                )

            with col2:
                gibberish_count = scored_df["is_gibberish"].sum()
                st.metric(
                    "Gibberish",
                    gibberish_count,
                    "rows" if gibberish_count != 1 else "row",
                )

            with col3:
                no_exp_count = scored_df["is_no_experience"].sum()
                st.metric(
                    "No Experience",
                    no_exp_count,
                    "rows" if no_exp_count != 1 else "row",
                )

            with col4:
                short_count = len(scored_df[scored_df["story_length"] < 20])
                st.metric(
                    "Very Short", short_count, "rows" if short_count != 1 else "row"
                )

            # Recommendations
            st.subheader("💡 Recommendations")
            rec_counts = scored_df["recommendation"].value_counts()

            for rec, count in rec_counts.items():
                if "Remove" in rec:
                    st.error(f"🗑️ {rec}: {count} rows")
                elif "Review" in rec:
                    st.warning(f"👀 {rec}: {count} rows")
                else:
                    st.success(f"✅ {rec}: {count} rows")

        else:
            st.info("📤 Please upload and process a file first.")

    with tab3:
        st.header("Detailed Results")

        if st.session_state.processed_data is not None:
            df = st.session_state.processed_data

            # Filters
            st.subheader("🔧 Filters")
            col1, col2, col3 = st.columns(3)

            with col1:
                score_filter = st.slider(
                    "Overall Score Range",
                    0,
                    100,
                    (0, 100),
                    help="Filter by overall quality score",
                )

            with col2:
                rec_filter = st.multiselect(
                    "Recommendations",
                    options=df["recommendation"].unique(),
                    default=df["recommendation"].unique(),
                )

            with col3:
                issue_filter = st.multiselect(
                    "Specific Issues",
                    options=[
                        "Profanity",
                        "Gibberish",
                        "No Experience",
                        "Brand Mismatch",
                        "Touchpoint Mismatch",
                    ],
                    default=[],
                )

            # Apply filters
            filtered_df = df[
                (df["overall_score"] >= score_filter[0])
                & (df["overall_score"] <= score_filter[1])
                & (df["recommendation"].isin(rec_filter))
            ]

            # Additional issue filters
            if "Profanity" in issue_filter:
                filtered_df = filtered_df[filtered_df["has_profanity"]]
            if "Gibberish" in issue_filter:
                filtered_df = filtered_df[filtered_df["is_gibberish"]]
            if "No Experience" in issue_filter:
                filtered_df = filtered_df[filtered_df["is_no_experience"]]
            if "Brand Mismatch" in issue_filter:
                filtered_df = filtered_df[filtered_df["brand_match_score"] == -100]
            if "Touchpoint Mismatch" in issue_filter:
                filtered_df = filtered_df[filtered_df["touchpoint_match_score"] == -100]

            st.write(f"Showing {len(filtered_df)} of {len(df)} rows")

            # Display options
            display_cols = st.multiselect(
                "Select columns to display",
                options=df.columns.tolist(),
                default=[
                    "brand",
                    "touchpoint",
                    "text",
                    "overall_score",
                    "recommendation",
                    "issues",
                ],
            )

            # Show filtered data
            if display_cols:
                st.dataframe(
                    filtered_df[display_cols], use_container_width=True, height=600
                )

            # Download filtered results
            st.subheader("💾 Download Results")
            col1, col2 = st.columns(2)

            with col1:
                csv = filtered_df.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    label="📥 Download Filtered Results (CSV)",
                    data=csv,
                    file_name=f"filtered_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_filtered_results",
                )

            with col2:
                full_csv = df.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    label="📥 Download All Results (CSV)",
                    data=full_csv,
                    file_name=f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_all_results_tab3",
                )

        else:
            st.info("📤 Please upload and process a file first.")

    with tab4:
        st.header("Quality Report")

        if st.session_state.report is not None:
            # Display report
            st.text(st.session_state.report)

            # Download report
            st.download_button(
                label="📥 Download Report (TXT)",
                data=st.session_state.report,
                file_name=f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )

            # Show worst examples
            if st.session_state.scored_df is not None:
                st.subheader("🚨 Worst Quality Examples")

                worst_rows = st.session_state.scored_df.sort_values(
                    "overall_score", ascending=True
                ).head(10)
                for idx, row in worst_rows.iterrows():
                    with st.expander(
                        f"Score: {row['overall_score']:.1f} - {row.get('brand', 'N/A')} / {row.get('touchpoint', 'N/A')}"
                    ):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.write("**Scores:**")
                            st.write(f"- Brand: {row.get('brand_match_score', 'N/A')}")
                            st.write(
                                f"- Touchpoint: {row.get('touchpoint_match_score', 'N/A')}"
                            )
                            st.write(f"- Quality: {row.get('quality_score', 'N/A')}")
                        with col2:
                            st.write("**Story:**")
                            st.write(row.get("story_preview", "N/A"))
                            st.write("**Issues:**")
                            st.write(row.get("issues", []))
                            st.write(
                                f"**Recommendation:** {row.get('recommendation', 'N/A')}"
                            )

        else:
            st.info("📤 Please upload and process a file first.")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>MESH Experience Data Cleaner v1.0 | Powered by Claude AI</p>
            <p>Need help? Contact your data team.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
