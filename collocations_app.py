import streamlit as st
import pickle
import glob
import os
import pandas as pd
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures

# Streamlit UI
st.title("Latin Collocation Dashboard")
# text with my name and date etc.
st.markdown(
    """
    This dashboard displays the top bigrams by Chi-squared, PMI, or Likelihood Ratio in a given Latin text from the [CLTK-Tesserae collection](https://github.com/cltk/lat_text_tesserae). Presented as part of ["All the Collocations You *Really* Need to Know"](https://diyclassics.github.io/collocations/) at CAAS2024 in New Brunswick, NJ.

    Written by P.J. Burns, October 2024. Last updated 10\.18\.2024.
    """
)

# Checkbox for lemma vs token
use_lemma = st.checkbox("Use Lemma", value=False)

# Set the directory based on the checkbox state
data_dir = "data/bigrams/lemma" if use_lemma else "data/bigrams/token"

# Load all pickle files and sort them
pickle_files = sorted(glob.glob(os.path.join(data_dir, "*.p")))

# Extract file names from paths
file_names = [os.path.basename(f) for f in pickle_files]

# Ensure default option is "cicero.brutus.p" if it exists
default_index = (
    file_names.index("cicero.brutus.p") if "cicero.brutus.p" in file_names else 0
)

# File selection with default option
file_choice = st.selectbox("Select a file", file_names, index=default_index)

# Map the selected file name back to the full file path
file_path = pickle_files[file_names.index(file_choice)]

with open(file_path, "rb") as f:
    bigramFinder: BigramCollocationFinder = pickle.load(f)

# Get the maximum frequency of bigrams
max_freq = max(bigramFinder.ngram_fd.values())

# Slider for frequency filter with default value set to 5
N = st.slider("Minimum Frequency", min_value=1, max_value=max_freq, value=5)

# Apply frequency filter
bigramFinder.apply_freq_filter(N)

# Dropdown for metric selection
metric_choice = st.selectbox(
    "Select a metric", ["Chi-squared", "PMI", "Likelihood Ratio"]
)

# Check if there are any bigrams left after applying the frequency filter
if len(bigramFinder.ngram_fd) == 0:
    st.write("No bigrams found after applying the frequency filter.")
else:
    # Initialize BigramAssocMeasures
    bigram_measures = BigramAssocMeasures()

    # Calculate scores based on the selected metric
    if metric_choice == "Chi-squared":
        scored_bigrams = bigramFinder.score_ngrams(bigram_measures.chi_sq)
    elif metric_choice == "PMI":
        scored_bigrams = bigramFinder.score_ngrams(bigram_measures.pmi)
    elif metric_choice == "Likelihood Ratio":
        scored_bigrams = bigramFinder.score_ngrams(bigram_measures.likelihood_ratio)

    # Convert scored bigrams to DataFrame
    top_bigrams_df = pd.DataFrame(scored_bigrams, columns=["Bigram", "Measure"])

    # Add Rank and Frequency columns
    top_bigrams_df["Rank"] = top_bigrams_df.index + 1
    top_bigrams_df["Frequency"] = top_bigrams_df["Bigram"].apply(
        lambda x: bigramFinder.ngram_fd[x]
    )

    # Reorder columns
    top_bigrams_df = top_bigrams_df[["Rank", "Bigram", "Measure", "Frequency"]]

    # Display the DataFrame
    st.dataframe(top_bigrams_df, use_container_width=True, hide_index=True)
