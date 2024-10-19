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

# Load all pickle files that have 'de_domo_sua' in the name
pickle_files = glob.glob(f"{data_dir}/*.p")

# Extract base names for display in the dropdown and remove .tess_bigrams
file_names = [
    os.path.splitext(os.path.basename(file))[0].replace(".tess_bigrams", "")
    for file in pickle_files
]

# File selection
file_choice = st.selectbox("Select a file", file_names)

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
        scores = [
            (bigram, round(score, 2))
            for bigram, score in bigramFinder.score_ngrams(bigram_measures.chi_sq)
        ]
    elif metric_choice == "PMI":
        scores = [
            (bigram, round(score, 2))
            for bigram, score in bigramFinder.score_ngrams(bigram_measures.pmi)
        ]
    elif metric_choice == "Likelihood Ratio":
        scores = [
            (bigram, round(score, 2))
            for bigram, score in bigramFinder.score_ngrams(
                bigram_measures.likelihood_ratio
            )
        ]

    # Sort the bigrams by their scores in descending order
    ranked_bigrams = sorted(scores, key=lambda x: x[1], reverse=True)

    # Select the top 25 bigrams or fewer if less than 25 are available
    top_bigrams = ranked_bigrams[:25]

    # Create a DataFrame for the top bigrams
    top_bigrams_df = pd.DataFrame(top_bigrams, columns=["bigram", metric_choice])

    # Add frequency column
    top_bigrams_df["Frequency"] = top_bigrams_df["bigram"].apply(
        lambda x: bigramFinder.ngram_fd[x]
    )

    # Convert bigrams to strings
    top_bigrams_df["bigram"] = top_bigrams_df["bigram"].apply(
        lambda x: f"({x[0]}, {x[1]})"
    )

    # Reorder columns to have Rank, Bigram, Metric, Frequency
    top_bigrams_df.insert(0, "Rank", range(1, len(top_bigrams_df) + 1))

    # Display DataFrame
    st.write(f"Top {len(top_bigrams_df)} bigrams by {metric_choice} in {file_choice}:")
    st.dataframe(top_bigrams_df, use_container_width=True, hide_index=True)
