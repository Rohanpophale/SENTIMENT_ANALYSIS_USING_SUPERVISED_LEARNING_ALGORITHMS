import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from flask import Flask, render_template, request

# Create a Flask app
app = Flask(__name__)

# Load the trained models
with open("RandomForest_Tweet_10000_model.model", "rb") as f:
    RandomForest_Tweet_model = pickle.load(f)

# with open("reddit_svm_model.model", "rb") as f:
#     reddit_svm_model = pickle.load(f)

# Load the fitted TF-IDF vectorizer used during training
with open("tfidf_10000_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def sentiment_analysis():
    if request.method == "POST":
        # Get the input text from the form
        input_text = [request.form["input_text"]]
        
        # Transform the input text into a TF-IDF vector
        input_text_tfidf = vectorizer.transform(input_text)

        # Make predictions using both models
        prediction_tweet = RandomForest_Tweet_model.predict(input_text_tfidf)
        # prediction_reddit = reddit_svm_model.predict(input_text_tfidf)

        # Define a mapping dictionary
        sentiment_mapping = {'Negative': -1.0, 'Neutral': 0.0, 'Positive': 1.0}

        # Convert the text sentiment predictions to numerical values
        prediction_tweet_numeric = sentiment_mapping.get(prediction_tweet, None)
        
        # Compare predictions and choose the best one
        # if prediction_tweet_numeric == prediction_reddit:
        if prediction_tweet_numeric == -1.0:
            result = 'Negative'
        elif prediction_tweet_numeric == 0.0:
            result = 'Nutral' 
        elif prediction_tweet_numeric == 1.0:
            result = 'Positive'
        sentiment = f"Sentiment: {result}"
        # else:
        #     if prediction_tweet_numeric == -1.0:
        #         result = 'Negative'
        #     elif prediction_tweet_numeric == 0.0:
        #         result = 'Nutral' 
        #     elif prediction_tweet_numeric == 1.0:
        #         result = 'Positive'
        #     sentiment = f"Sentiment: {result}"

        return render_template("index.html", sentiment=sentiment)

    return render_template("index.html", sentiment=None)

if __name__ == "__main__":
    app.run(debug=True)
