import streamlit as st
import pandas as pd

from services.preprocessing import analyze_text


st.set_page_config(page_title="Social Media Comment Intelligence", layout="wide")

st.title("Social Media Comment Intelligence")
st.write("Upload an Excel file to explore your social media comments.")

uploaded_file = st.file_uploader("Upload an Excel file (.xlsx)", type=["xlsx"])

if uploaded_file is None:
    st.info("Please upload an Excel file to get started.")
    st.stop()

try:
    df = pd.read_excel(uploaded_file, engine="openpyxl")
except Exception as error:
    st.error(f"Could not read the Excel file: {error}")
    st.stop()

if df.empty:
    st.warning("The uploaded file has no rows.")
    st.stop()

st.subheader("Uploaded Dataset")
st.dataframe(df, use_container_width=True)

st.subheader("Available Columns")
st.write(list(df.columns))

st.subheader("Select Comment Column")
comment_column = st.selectbox(
    "Which column contains the comments?",
    options=df.columns,
)

if comment_column not in df.columns:
    st.error(f"Column '{comment_column}' was not found in the dataset.")
    st.stop()

comment_series = df[comment_column]

# Convert values to text so we can safely inspect and count them.
text_series = comment_series.astype("string")

# Check that the column looks like text data.
non_null_values = text_series.dropna()
if non_null_values.empty:
    st.warning("The selected column has no values to inspect.")
else:
    sample_values = non_null_values.head(20)
    numeric_like_count = pd.to_numeric(sample_values, errors="coerce").notna().sum()

    if numeric_like_count == len(sample_values):
        st.error(
            f"Column '{comment_column}' appears to contain numeric data, not text comments."
        )
        st.stop()

st.subheader("Comment Statistics")

total_comments = len(text_series)
non_empty_comments = text_series.notna() & text_series.str.strip().ne("")
non_empty_count = int(non_empty_comments.sum())
empty_count = total_comments - non_empty_count

col1, col2, col3 = st.columns(3)
col1.metric("Total comments", total_comments)
col2.metric("Non-empty comments", non_empty_count)
col3.metric("Empty comments", empty_count)

st.subheader("Comment Preview")
preview_df = pd.DataFrame({comment_column: text_series[non_empty_comments].head(10)})
st.dataframe(preview_df, use_container_width=True)

# Analysis runs only when the user clicks the button below.
if st.button("Analyze Comments"):
    comments = df[comment_column]
    total_to_process = len(comments)
    results = []

    progress_bar = st.progress(0, text="Processing comments...")

    for index, comment in enumerate(comments):
        results.append(analyze_text(comment))
        progress_bar.progress(
            (index + 1) / total_to_process,
            text=f"Processing comment {index + 1} of {total_to_process}...",
        )

    progress_bar.empty()

    # Turn each result dictionary into its own row, then into separate columns.
    analysis_df = pd.DataFrame(results)
    analysis_df = analysis_df.rename(columns={"original_text": "comment"})

    display_columns = [
        "comment",
        "cleaned_text",
        "emojis",
        "emoji_count",
        "hashtags",
        "hashtag_count",
        "has_url",
        "excessive_repetition",
        "exclamation_count",
        "question_count",
    ]
    analysis_df = analysis_df[display_columns]

    st.subheader("Text Analysis")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Comments processed", total_to_process)
    col2.metric("Comments with emojis", int((analysis_df["emoji_count"] > 0).sum()))
    col3.metric("Comments with hashtags", int((analysis_df["hashtag_count"] > 0).sum()))
    col4.metric("Comments with URLs", int(analysis_df["has_url"].sum()))
    col5.metric(
        "Comments with excessive repetition",
        int(analysis_df["excessive_repetition"].sum()),
    )

    st.dataframe(analysis_df, use_container_width=True)
