Twitter Sentiment Analysis*

This project analyzes sentiments of tweets about airlines using machine learning. The goal is to classify tweets into three categories: *positive, **negative, or **neutral*. 

It uses *text preprocessing, **TF-IDF vectorization, and a **Logistic Regression model* to predict sentiments with high accuracy.

---

## *Project Overview*
- *Dataset*: The project utilizes the [Twitter Airline Sentiment Dataset](https://www.kaggle.com/crowdflower/twitter-airline-sentiment) containing 14,000 tweets.
- *Environment: The code is implemented in **Google Colab*.
- *Techniques Used*:
  - Text cleaning: stopword removal, lemmatization
  - Feature extraction: TF-IDF (Term Frequency-Inverse Document Frequency)
  - Model training: Logistic Regression
- *Accuracy: The model achieves an accuracy of approximately **79%*.

---

## *Steps in the Project*

1. *Dataset Loading*: Loaded the dataset and explored the tweet sentiments (positive, negative, neutral).
2. *Data Preprocessing*: 
   - Removed stopwords and punctuation.
   - Applied lemmatization to normalize the text.
3. *Feature Extraction: Used **TF-IDF* to convert text data into numerical vectors.
4. *Model Training: Trained a **Logistic Regression* model on the TF-IDF vectors.
5. *Evaluation*: Evaluated the model's performance and obtained ~79% accuracy.

---

## *How to Run the Project*

1. Clone this repository:
   bash
   https://github.com/muntahasidd/Sentiment-Analysis-Twitter.git
   
2. Open the Twitter_Sentiment_Analysis.ipynb notebook in *Google Colab*.
3. Install the required libraries:
   bash
   !pip install nltk scikit-learn pandas matplotlib
   
4. Run all the cells in the notebook to preprocess the data, train the model, and view the results.

---

## *Key Features*

- *Data Preprocessing*:
  - Removed noise from tweets to improve the quality of data.
  - Transformed text into a structured format suitable for machine learning.
- *TF-IDF*: Captures important words in each tweet while minimizing the impact of common words.
- *Logistic Regression*: A simple yet effective classification model.

---

## *Future Work*

- Experiment with other machine learning algorithms such as *Naive Bayes* or *SVM*.
- Use advanced NLP techniques like *Word2Vec* or *BERT* for better feature representation.
- Implement real-time sentiment analysis for live tweets.

---

## *Acknowledgments*

- Dataset: [Kaggle](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)
- Libraries: NLTK, Scikit-learn, Pandas, Matplotlib
