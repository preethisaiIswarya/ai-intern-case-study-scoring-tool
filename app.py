import streamlit as st
import pandas as pd
import json
import nltk
import language_tool_python
from nltk.sentiment import SentimentIntensityAnalyzer

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('vader_lexicon')

### Load rubric ###
with open('rubric.json') as f:
    rubric = json.load(f)

tool = language_tool_python.LanguageTool('en-US')
sentiment = SentimentIntensityAnalyzer()

def check_keywords(text, keywords):
    return sum(1 for kw in keywords if kw.lower() in text.lower())

def grammar_score(text):
    matches = tool.check(text)
    errors = len(matches)
    words = len(nltk.word_tokenize(text))
    errors_per_100 = errors / (words / 100) if words > 0 else 0
    score = max(0, 1 - min(errors_per_100/10, 1))
    return score, errors

def vocab_richness(text):
    words = nltk.word_tokenize(text)
    unique = set(words)
    ttr = len(unique)/len(words) if words else 0
    return ttr

def filler_word_score(text):
    filler_words = ['um', 'uh', 'like', 'you know', 'so', 'actually', 'basically', 'right', 'i mean', 
                    'well', 'kinda', 'sort of', 'okay', 'hmm', 'ah']
    total_words = len(nltk.word_tokenize(text))
    count = sum(text.lower().split().count(fw) for fw in filler_words)
    percent = (count/total_words)*100 if total_words else 0
    if percent <= 3:
        pts = 15
    elif percent <= 6:
        pts = 12
    elif percent <= 9:
        pts = 9
    elif percent <= 12:
        pts = 6
    else:
        pts = 3
    return pts, percent

def sentiment_score(text):
    result = sentiment.polarity_scores(text)
    pos = result['pos']
    if pos >= 0.9:
        pts = 15
    elif pos >= 0.7:
        pts = 12
    elif pos >= 0.5:
        pts = 9
    elif pos >= 0.3:
        pts = 6
    else:
        pts = 3
    return pts, pos

def rubric_score(text):
    total = 0
    feedback = []
    # Salutation and keyword matching
    for crit in rubric['criteria']:
        found = check_keywords(text, crit['keywords'])
        pts = min(found, 1) * crit['weight']
        feedback.append(f"{crit['name']}: {found} keywords -> {pts}/{crit['weight']}")
        total += pts
    # Speech rate (words per minute, default 100 WPM)
    words = len(nltk.word_tokenize(text))
    # Assume 1.5 min speech (90s) if duration unknown
    duration = st.number_input("Enter duration (seconds)", min_value=30, max_value=240, value=90)
    wpm = words/(duration/60)
    if wpm > 161:
        s_pts = 2
    elif wpm > 141:
        s_pts = 6
    elif wpm > 111:
        s_pts = 10
    elif wpm > 81:
        s_pts = 6
    else:
        s_pts = 2
    feedback.append(f"Speech Rate: {wpm:.1f} WPM -> {s_pts}/10")
    total += s_pts

    # Grammar
    g_score, g_errors = grammar_score(text)
    g_pts = int(g_score * 10)
    feedback.append(f"Grammar: {g_errors} errors ({g_score:.2f}) -> {g_pts}/10")
    total += g_pts

    # Vocabulary
    ttr = vocab_richness(text)
    if ttr > 0.9:
        v_pts = 10
    elif ttr > 0.7:
        v_pts = 8
    elif ttr > 0.5:
        v_pts = 6
    elif ttr > 0.3:
        v_pts = 4
    else:
        v_pts = 2
    feedback.append(f"Vocabulary: TTR={ttr:.2f} -> {v_pts}/10")
    total += v_pts

    # Filler Words
    f_pts, f_percent = filler_word_score(text)
    feedback.append(f"Filler Words: {f_percent:.2f}% -> {f_pts}/15")
    total += f_pts

    # Engagement/Sentiment
    e_pts, pos_score = sentiment_score(text)
    feedback.append(f"Engagement: Sentiment={pos_score:.2f} -> {e_pts}/15")
    total += e_pts

    overall = min(100, total)
    return overall, feedback

st.title("Student Spoken Introduction Evaluator")

st.write("Paste your transcript below or upload a `.txt` file.")

txt = st.text_area("Transcript Text")

if st.button("Evaluate"):
    if txt:
        score, fb = rubric_score(txt)
        st.markdown(f"## Final Score: **{score}/100**")
        st.write("### Detailed Feedback:")
        for f in fb:
            st.write("- " + f)
    else:
        st.warning("Please enter or upload transcript text.")
