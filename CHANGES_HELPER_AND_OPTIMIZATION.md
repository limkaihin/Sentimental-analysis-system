# Changes in this package (2025-09-26)

- Added `src/lib/sentences_for_window.py` — helper to list exact sentences inside a fixed (start,end) window.
- Added `return_contrib` option to `src/lib/sentiment_scoring.py::calculate_window_sentiment(...)` to retrieve per-token contributions.
- Optimized `src/lib/sliding_window.py` for `unit="sentence"` using prefix sums via `best_fixed_windows_from_scores(...)`.
