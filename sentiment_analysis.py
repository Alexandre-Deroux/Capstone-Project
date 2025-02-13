import requests
import nltk
import spacy
import gender_guesser.detector as gender
from datetime import datetime, timedelta
from newspaper import Article
from nltk.tokenize import sent_tokenize
from transformers import pipeline
import warnings

# Ignore unnecessary warnings
warnings.simplefilter("ignore", category=FutureWarning)

# Load NLP model for text parsing
nlp = spacy.load("en_core_web_sm")

# Gender detection
gender_detector = gender.Detector()

# Sentiment Analysis Model
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Domains to Exclude
EXCLUDED_DOMAINS = ["consent.yahoo.com"]

# Fraud-Related Keywords
FRAUD_KEYWORDS = [
    "fraud", "money laundering", "corruption", "financial crime", "embezzlement",
    "tax evasion", "bribery", "scam", "misconduct", "Ponzi scheme", "bank fraud",
    "securities fraud", "tax fraud", "offshore accounts", "insider trading", "market manipulation"
]

# Financial Crime Context Keywords
FRAUD_CONTEXT_WORDS = [
    "financial crime", "money laundering", "embezzlement", "tax fraud", "bribery",
    "securities fraud", "bank fraud", "prosecutors", "charged", "indicted", "Ponzi scheme",
    "illegal transactions", "offshore accounts", "criminal", "felony", "trial"
]

def get_pronouns(person_name):
    """ Detects the person's gender and returns relevant pronouns for fraud detection. """
    first_name = person_name.split()[0]
    gender_guess = gender_detector.get_gender(first_name)

    if gender_guess in ["male", "mostly_male"]:
        return ["he", "his", "him"]
    elif gender_guess in ["female", "mostly_female"]:
        return ["she", "her", "hers"]
    else:
        return ["he", "his", "him", "she", "her", "hers"]

def extract_fraud_sentences(text, person_name):
    """ Extracts sentences with fraud-related keywords and checks if they refer to the person. """
    sentences = sent_tokenize(text)
    fraud_sentences = [sentence for sentence in sentences if any(keyword in sentence.lower() for keyword in FRAUD_KEYWORDS)]

    relevant_sentences = []
    pronouns = get_pronouns(person_name)

    for sentence in fraud_sentences:
        doc = nlp(sentence)
        subjects = [token.text.lower() for token in doc if token.dep_ in ("nsubj", "nsubjpass")]

        # Check if the person is mentioned directly
        if person_name.lower() in sentence.lower():
            relevant_sentences.append(sentence)
            continue

        # Check if pronouns refer to the person
        if any(pronoun in subjects for pronoun in pronouns):
            relevant_sentences.append(sentence)

    return relevant_sentences

def check_fraud_context(text, person_name):
    """ Ensures fraud is actually related to the person, not just mentioned randomly"""
    sentences = sent_tokenize(text)
    pronouns = get_pronouns(person_name)

    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in FRAUD_KEYWORDS) and any(context in sentence.lower() for context in FRAUD_CONTEXT_WORDS):
            doc = nlp(sentence)
            subjects = [token.text.lower() for token in doc if token.dep_ in ("nsubj", "nsubjpass")]

            # Check if the person or their pronouns are referenced
            if person_name.lower() in sentence.lower() or any(pronoun in subjects for pronoun in pronouns):
                return True

    return False

def get_sentiment(fraud_sentences):
    """ Analyzes only the sentiment of fraud-related sentences"""
    sentiment_scores = []

    for sentence in fraud_sentences:
        try:
            result = sentiment_pipeline(sentence)
            label = result[0]['label']
            score = {"1 star": -1, "2 stars": -0.5, "3 stars": 0, "4 stars": 0.5, "5 stars": 1}.get(label, 0)
            sentiment_scores.append(score)
        except Exception:
            continue

    return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0  # Average sentiment

person_name = "Elon Musk"
date = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")
url = f"https://newsapi.org/v2/everything?q={person_name}&from={date}&sortBy=popularity&apiKey=4bc3b879ee624f939ec5e7f1c38451ef"

response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    articles = data.get("articles", [])
else:
    print(f"❌ Error {response.status_code}: {response.text}")
    exit()

filtered_articles = []

for article in articles:
    url = article.get("url", "")

    if any(domain in url for domain in EXCLUDED_DOMAINS):
        continue

    try:
        news_article = Article(url)
        news_article.download()
        news_article.parse()
        content = news_article.text

        # Verify that the article discusses fraud related to the person
        if check_fraud_context(content, person_name):
            fraud_sentences = extract_fraud_sentences(content, person_name)
            sentiment_score = get_sentiment(fraud_sentences)

            if fraud_sentences:
                filtered_articles.append({
                    "title": article["title"],
                    "url": url,
                    "keywords": [kw for kw in FRAUD_KEYWORDS if kw in content.lower()],
                    "summary": " ".join(fraud_sentences[:3]),
                    "sentiment": sentiment_score
                })

    except Exception as e:
        continue

# Compute the average sentiment across all fraud-related articles
if filtered_articles:
    global_sentiment = sum(article["sentiment"] for article in filtered_articles) / len(filtered_articles)

    print(f"\n🎯 **Global Sentiment for '{person_name}': {global_sentiment:.2f}**\n")
    for article in filtered_articles:
        print(f"✅ **{article['title']}**")
        print(f"🔗 {article['url']}")
        print(f"📌 Keywords: {', '.join(article['keywords'])}")
        print(f"📄 Summary: {article['summary']}")
        print(f"📊 Sentiment: {article['sentiment']:.2f}\n")
else:
    print(f"\n⚠️ No relevant fraud-related articles found for {person_name}.")