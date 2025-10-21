import streamlit as st
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter
import string

# --- Constants ---
MAX_WORDS = 70000             # Maximum words to process
MAX_WORDCLOUD_WORDS = 5000    # Maximum words in WordCloud

# --- Sidebar Note ---
st.sidebar.info(
    """
⚠️ Note for Users:
• The app processes up to 70,000 words for best performance.
• Very large articles or books may be truncated.
• Word clouds display the top 5,000 words only.
• Word frequency excludes common English stopwords and punctuation.
"""
)

# --- App title ---
st.title("Linguistics Analyzer")
st.info("Mobile users: Please click the arrow (top-left) to open the sidebar for instructions and notes.")
# --- File upload OR text input ---
uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

if uploaded_file is not None:
    # Read the uploaded file
    text = uploaded_file.read().decode("utf-8")
else:
    st.markdown("📝 Please paste your text below (up to 70,000 words).")
    text = st.text_area("Paste your text here:", height=250)

# --- Analyze button ---
if st.button("Analyze") and text:

    # --- Preprocess text ---
    words = text.split()
    if len(words) > MAX_WORDS:
        st.warning(f"Input text truncated to {MAX_WORDS} words for performance.")
        words = words[:MAX_WORDS]
        text = " ".join(words)

    # --- Word Frequency (filtered & cleaned) ---
    stopwords = set(STOPWORDS)
    translator = str.maketrans('', '', string.punctuation)  # remove punctuation
    words_cleaned = [word.lower().translate(translator) for word in words]
    words_filtered = [word for word in words_cleaned if word and word not in stopwords]
    word_freq = Counter(words_filtered)
    
    st.subheader("Top 20 Word Frequency (Excluding Stopwords & Punctuation)")
    st.dataframe(dict(word_freq.most_common(20)))

    # --- Sentiment Analysis ---
    st.subheader("Sentiment")
    st.markdown("""
**What this means:**  
Polarity ranges from -1 (very negative) to +1 (very positive).  
Subjectivity ranges from 0 (very objective/factual) to 1 (very subjective/opinionated).  

This analysis uses **TextBlob**, which looks at the words and phrases in your text to determine overall sentiment and subjectivity.
""")
    blob = TextBlob(text)
    st.write(f"Polarity: {blob.sentiment.polarity:.2f}  |  Subjectivity: {blob.sentiment.subjectivity:.2f}")

    # --- Word Cloud ---
    st.subheader("Word Cloud")
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stopwords,
        max_words=MAX_WORDCLOUD_WORDS
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)