import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from ntscraper import Nitter

st.markdown(
    """
    <style>
        body {
            background-color: #f5f5f5;
        }
        .stApp {
            background: url('https://source.unsplash.com/1600x900/?abstract,nature') no-repeat center center fixed;
            background-size: cover;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            border: none;
            font-size: 16px;
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
            padding: 10px;
            font-size: 16px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Downloading Stopwords once using the Streamlit's caching

@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

# Load model and vectorizer
@st.cache_resource
def load_model_and_vectorize():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer


# Define statement prediction function
def predict_sentiment(text, model, vectorizer, stop_words):
    # Preprocess the text
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if not word in stop_words]
    text = ' '.join(text)
    text = [text]
    text = vectorizer.transform(text)

    # Predict statement 
    sentiment = model.predict(text)
    return "Negative" if sentiment == 0 else "Positive"


# Intialize the Nitter Scraper 

@st.cache_resource
def initialize_scraper():
    return Nitter(log_level = 1)


def create_card(tweet_text, sentiment):
    color = "#2ECC71" if sentiment == "Positive" else "#E74C3C"
    card_html = f"""
    <div style = "background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0;">
        <h5 style = "color : white;"> {sentiment} Sentiment </h5>
        <p style = "color : white"> {tweet_text} </p>
    </div>
    """
    return card_html

# Main app logic
def main():
    # Set title
    st.markdown("<h1 style='text-align: center; color: #2C3E50;'>💬 Social Media Sentiment Analysis</h1>", unsafe_allow_html=True)


    # Load stopwords model vectorizer and scraper only once

    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorize()
    scraper = initialize_scraper()

    option = st.selectbox("Choose an option", ["Enter Manually", "Fetch some tweets"], index=0)

    if option == "Enter Manually":
        text_input = st.text_area("Enter the text to Analyze the sentiment")
    
        if st.button("Analyze", key="analyze_btn"):
            sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
            color = "#2ECC71" if sentiment == "Positive" else "#E74C3C"  # Background color
            text_color = "#000000"  # Text color set to black for visibility

            st.markdown(
                f"""
                <div style="background-color: {color}; padding: 15px; border-radius: 10px; 
                text-align: center; color: {text_color}; font-size: 18px;">
                    <b>Sentiment:</b> {sentiment}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write(f"**Sentiment:** {sentiment}")  # Added bold text for better readability

    elif option == "Fetch some tweets":
        username = st.text_input("Enter Twitter username")
        
        if st.button("Fetch tweets"):
            try:
                tweets_data = scraper.get_tweets(username, mode='user', number=5)

                # Check if tweets are available
                if not tweets_data or 'tweets' not in tweets_data or not isinstance(tweets_data['tweets'], list) or len(tweets_data['tweets']) == 0:
                    raise ValueError("No tweets found")  # Raise error if empty

            except (IndexError, ValueError, Exception) as e:
                tweets_data = None
                st.markdown(
                    """
                    <div style="
                        padding: 10px; 
                        background-color: #ffcccc; 
                        color: #990000; 
                        border-radius: 5px; 
                        border-left: 5px solid #cc0000;">
                        ⚠️ Error fetching tweets or no tweets found for this username.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                # Display tweets
                for tweet in tweets_data['tweets']:
                    tweet_text = tweet.get('text', '')
                    if tweet_text:
                        sentiment = predict_sentiment(tweet_text, model, vectorizer, stop_words)
                        card_html = create_card(tweet_text, sentiment)
                        st.markdown(card_html, unsafe_allow_html=True)

            # Check if tweets_data exists and contains tweets
            if tweets_data and 'tweets' in tweets_data and isinstance(tweets_data['tweets'], list) and len(tweets_data['tweets']) > 0:
                for tweet in tweets_data['tweets']:
                    tweet_text = tweet.get('text', '')  # Ensure 'text' key exists
                    if not tweet_text:
                        continue  # Skip empty tweets

                    # Predict Sentiment of the tweet
                    sentiment = predict_sentiment(tweet_text, model, vectorizer, stop_words)


                    # Create and display the colored carc for the predicted tweet
                    card_html = create_card(tweet_text,sentiment)
                    st.markdown(card_html, unsafe_allow_html=True)
    

if __name__ == '__main__':
    main()




