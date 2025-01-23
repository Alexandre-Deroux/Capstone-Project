import requests
from datetime import datetime, timedelta
from newspaper import Article
#from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

#nltk.download('vader_lexicon') 
sia = SentimentIntensityAnalyzer()

# Liste des domaines à exclure (Yahoo bloque l'accès)
EXCLUDED_DOMAINS = ["consent.yahoo.com"]

# Liste des mots-clés liés à la fraude et au blanchiment
FRAUD_KEYWORDS = [
    "fraud", "money laundering", "corruption", "scandal", "financial crime", 
    "embezzlement", "tax evasion", "scam", "lawsuit", "investigation"
]

def get_sentiment(text):
    score = sia.polarity_scores(text)
    return score['compound'] #score entre -1(négatif) et 1(positif)

# Fonction pour vérifier si un article parle de fraude ou de blanchiment
def contains_fraud_keywords(text):
    return any(keyword in text.lower() for keyword in FRAUD_KEYWORDS)

query = "Tesla"
date = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")  # Un mois avant aujourd'hui

url = f"https://newsapi.org/v2/everything?q={query}&from={date}&sortBy=popularity&apiKey=4bc3b879ee624f939ec5e7f1c38451ef"

response = requests.get(url)

if response.status_code == 200: 
    data = response.json()
    articles = data.get("articles", [])

else : 
    print(f"❌Erreur {response.status_code}: {response.text}")
    exit()

filtered_articles = []

for article in articles:
    url = article.get("url", "")

    # Filtrer les URLs interdites
    if any(domain in url for domain in EXCLUDED_DOMAINS):
        continue 

    try:
        news_article = Article(url)
        news_article.download()
        news_article.parse()
        content = news_article.text

        # Vérifier si l'article parle de fraude/blanchiment et a un sentiment négatif
        sentiment_score = get_sentiment(content)
        if sentiment_score < 0 and contains_fraud_keywords(content):
            filtered_articles.append((article["title"], url, sentiment_score))
            print(f"✅ {article['title']}")
            print(f"🔗 {url}")
            print(f"📊 Sentiment: {sentiment_score:.2f}\n")

    except Exception:
        continue

# Calcul du sentiment global basé sur les articles filtrés
if filtered_articles:
    global_sentiment = sum(s[2] for s in filtered_articles) / len(filtered_articles)
    print(f"\n🎯 **Global sentiment for '{query}': {global_sentiment:.2f}**")
else:
    print("\n⚠️ No relevant articles found.")


"""
article_sentiments = []
for article in articles[:]: 
    try:
        news_article = Article(article['url'])
        news_article.download()
        news_article.parse()
        content = news_article.text
        
        sentiment = get_sentiment(content)
        article_sentiments.append(sentiment)

        print(f"✅ {article['title']}")
        print(f"🔗 {article['url']}")
        print(f"📊 Sentiment: {sentiment:.2f}\n")

    except Exception as e:
        print(f"❌ Impossible d'extraire {article['url']} ({str(e)})")
if article_sentiments:
    global_sentiment = sum(article_sentiments) / len(article_sentiments)
    print(f"🎯 Sentiment global pour '{query}': {global_sentiment:.2f}")
else:
    print("Aucun article valide trouvé pour analyser le sentiment.")

"""