# A Lexicon-Driven Approach to Sentiment Analysis (Sentimental Analysis System)

-   This project focuses on creating a system to analyse the sentiment of movie reviews using a predefined dictionary. The system will compute sentiment scores for each sentence, identify the positive & negative sentences, and use a sliding window technique to find the positive & negative paragraph segments. Text preprocessing, sentiment scoring, and visualisation will be performed in the project, which will allow us to practice Python programming fundamentals such as string manipulation, modular design, and data visualisation.


# What it does

- Scores text using lexicons, prints per‑sentence scores, finds most positive/negative fixed‑k windows, and finds best positive/negative arbitrary‑length sentence segments via Kadane scans. 

- Plots sliding‑window sentiment and optionally shades the most positive/negative spans; can also render simple bar charts of class counts when enabled.

- Includes utilities to tokenize/split sentences and a dictionary‑driven word segmentation module for spaceless strings. 

# Folder map

- src/lib: preprocessing, lexicon loader, sentiment scoring, sliding windows, extrema selection, visuals, and word segmentation.

- src/app: CLI entry point main.python that wires everything with argparse flags and optional plotting. 

- tests: example tests for preprocessing, sliding windows, and sentence workflows. 

# Install (Windows/macOS/Linux)

- Create and activate a virtual environment, then install requirements with Python’s launcher. 

# Windows PowerShell:

- python -m venv .venv; .\.venv\Scripts\Activate.ps1; python -m pip install -r requirements.txt. 

# macOS/Linux:

- python3 -m venv .venv; source .venv/bin/activate; python -m pip install -r requirements.txt. 

# Lexicons expected

- Default AFINN and emoticon lexicons are read as token TAB score per line from data/lexicon by the loader, and names can be overridden via CLI flags. 

# Quick start (single review)

Example:

- python -m src.app.main --text "Good. Bad. Very good!" --unit sentence --window_size 2 --debug. =]

# What prints:

- Per‑sentence scores and sign‑aware extrema, fixed‑k window scores and sign‑aware extrema, and best positive/negative sentence segments with filtered lines. 

# Plotting and histograms

- Add --plot to draw a sliding‑window line plot; when saving is requested the figure is written to save_dir. 

- Add --bar_chart to render and optionally save a bar chart of sentence class counts or dataset label counts if save_dir or --bar_chart_file is specified. 

# Dataset mode

- Place .txt files under data/reviews and run without --text/--input_file to process a collection with optional plotting of the first N files using --plot --plot_first_n N. 

- The code tallies dataset‑level class counts and can save a dataset histogram when --bar_chart is used. 

# Key CLI flags

- --text / --input_file: select single‑review input mode from a literal or file. 

- -unit token|sentence and --window_size k: control sliding‑window unit and size. 

- --plot, --save_dir: enable plotting and select output folder for figures. 

- --bar_chart, --bar_chart_file: enable histogram and optionally choose output path. 

- --data_dir/--reviews_subdir/--lexicon_subdir/--afinn_name/--emoticons_name: control data and lexicon locations. 

# How the scoring works

- Preprocessing normalizes input and produces tokens and sentence token lists for downstream scoring.

- Sentiment scoring sums lexicon hits with simple handling and is used for both per‑sentence and window scoring. 

- Sliding windows compute sums over k tokens or k sentences for each start position, returning (start, end, score) triples. 

- Extrema selection chooses most positive and most negative windows with deterministic tie‑breaks on score then span length then earliest start. 

- Arbitrary‑length sentence segments use Kadane scans for max and min subarrays over sentence scores to find the best spans. 

# Optional GUI notes

Visual plotting uses Matplotlib with a headless‑safe backend, so figures can be saved without an interactive display. 

If using a Streamlit UI script in this repo, launch with python -m streamlit run streamlit_app.python from the root and set lexicon file paths in the app’s controls accordingly. 

# Word segmentation utility

- For spaceless strings, word_segmentation provides one best segmentation and all segmentations given a dictionary set. 

- This is exposed via separate functions that can be imported or wired to a CLI if desired. 

# Run tests

- From the repo root with an active venv: python -m pytest -q to run the provided tests. 

# Troubleshooting

- If “streamlit” or “pip” is not recognized, ensure the venv is activated and use python -m pip to install packages. 

- For plotting errors, confirm matplotlib is installed per requirements and rerun with --plot or --bar_chart. 