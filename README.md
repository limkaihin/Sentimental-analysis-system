# A Lexicon-Driven Approach to Sentiment Analysis (Sentimental Analysis System)

This project focuses on creating a system to analyse the sentiment of movie reviews using a predefined dictionary. The system will compute sentiment scores for each sentence, identify the positive & negative sentences, and use a sliding window technique to find the positive & negative paragraph segments. Text preprocessing, sentiment scoring, and visualisation will be performed in the project, which will allow us to practice Python programming fundamentals such as string manipulation, modular design, and data visualisation.


## Project Structure
project_directory/
src/: modular source code
   - preprocessing.py — Unicode normalization, tokenization, file loading, and generic window generator.

   - lexicon.py — unified loader for AFINN-en-165.txt and AFINN-emoticon-8.txt.

   - sentiment_scoring.py — window-level sentiment scoring with AFINN words, emoticons, negation, and intensity.
   
   - sliding_window.py — per-review sliding-window generation and scoring, returns (start_index, score).

    extrema_segments.py — finds most positive and most negative k-length segments.

    visualisation.py — plotting utilities to visualize window scores and annotate extrema.

    main.py — CLI entry point to run analysis and optionally save plots.

**Installation**


**Usage** 


**Testing** 


**Dependencies**


**Contributing**


**License**


**Acknowledgements**










### Key Sections of the `README.md`:

1. **Project Structure**: Lists the key files and directories in your project.
2. **Installation**: Details on how to clone the repo, set up a virtual environment, and install dependencies.
3. **Usage**: Instructions on how to run the sentiment analysis and visualize results.
4. **Testing**: Explains how to run unit tests for the project.
5. **Dependencies**: Lists Python libraries your project depends on.
6. **Contributing**: An invitation for others to contribute to your project.
7. **License**: Specifies the project's license (e.g., MIT).
8. **Acknowledgements**: Credits any external resources or datasets used (like AFINN lexicon, IMDB dataset).

### What You Need to Do:

1. **Replace the GitHub URL**: In the **Clone the repository** section, change `https://github.com/yourusername/sentiment-analysis-imdb.git` to your actual repository URL.
2. **License**: Add the `LICENSE` file to your project directory (if not already done), or change the license information if you choose a different license.

This should help others understand your project and how to run it. Let me know if you need further adjustments!
