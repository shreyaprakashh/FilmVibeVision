# FilmVibeVision: Analyzing Emotional Tones in IMDB Reviews
This project performs sentiment analysis on IMDB movie reviews using machine learning techniques. It utilizes the NLTK library for natural language processing and scikit-learn for machine learning algorithms.

## Dataset
The IMDB movie reviews dataset consists of movie reviews labeled as positive or negative sentiments. The dataset is provided by NLTK and contains a collection of movie reviews along with their corresponding sentiment labels.

## Methodology
* Data Preprocessing: The IMDB movie reviews dataset is loaded and preprocessed.Stop words are removed to improve the quality of features.
* Feature Extraction: TF-IDF vectorization is used to convert text data into numerical features.
* Model Training: Support Vector Machine (SVM) classifier with a linear kernel is trained on the TF-IDF features.
* Model Evaluation: The accuracy of the trained model is evaluated using a test set.
* Sentiment Analysis: The trained model is used to perform sentiment analysis on example texts.

* Results
The trained SVM classifier achieves an accuracy of approximately 80% on the test set.
Example text: "This movie is amazing! I loved every moment of it."
Predicted sentiment: Positive
* Future Improvements
** Experiment with different machine learning algorithms and hyperparameters to improve accuracy.
** Fine-tune preprocessing steps such as handling punctuation, stemming, or lemmatization.
** Explore deep learning-based approaches for sentiment analysis.
