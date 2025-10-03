# Sentiment Analysis System – Lexicon-Driven

This project analyses the sentiment of movie reviews using dictionary-based scoring. It computes sentiment at the sentence, token, and sliding-window levels, and provides both **command-line utilities** and an **interactive Streamlit web app** for visualization.

---

##  Project Structure

- **`src/lib/`**  
  Core modules for:
  - Preprocessing and tokenization  
  - Lexicon loading (AFINN, emoticons, etc.)  
  - Sentiment scoring (sentence, token, sliding-window)  
  - Extrema selection (most positive/negative spans)  
  - Visualization helpers  

- **`src/app/`**  
  - `main.py`: CLI entry point for batch/command-line analysis  

- **`streamlit_app.py`**  
  Interactive GUI for review sentiment analysis (recommended entry point).  

- **`tests/`**  
  Unit tests for preprocessing, scoring, and window analysis.  

---

##  Installation

Make sure you have **Python 3.9+** installed.  

```bash
# Create and activate virtual environment
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

##  Running the Project

### 1. Streamlit GUI (Recommended)

This launches an interactive web interface to paste reviews and view results.

```bash
streamlit run streamlit_app.py
```

Features:
- Sentence-level scoring with highlights  
- Sliding-window (token and sentence) analysis  
- Overall review sentiment score  
- Plots of positive/negative spans  

---

### 2. Command-Line Mode

Run directly on text or files for batch processing.

```bash
python -m src.app.main --text "Good. Bad. Very good!" --k 3
```

Flags:
- `--text` / `--file` → review input (string or file)   
- `--k ` → window size  
- `--dict` → dictionary file for segmentation  

---

##  Example Outputs

1. **Streamlit GUI**:  
   - Review overall score (positive/negative/neutral)  
   - Highlighted sentences with scores  
   - Line chart of sliding-window sentiment  

2. **CLI**:  
   ```
   Sentence scores:
   1. Good. -> +3
   2. Bad. -> -3
   3. Very good! -> +4

   Best positive window: (sent 1–3, score +4)
   Best negative window: (sent 2–2, score -3)
   ```

---

##  Key Functions

- **Sentence scoring** → per-sentence polarity values  
- **Token scoring** → fine-grained window scores  
- **Sliding-window** → fixed-size windows across tokens/sentences  
- **Kadane scans** → best arbitrary-length segment scoring  
- **Visualization** → Streamlit plots and charts  

---

##  Testing

Run unit tests with:

```bash
pytest tests/
```
