import requests
import pandas as pd
import streamlit as st
import pycountry
from difflib import get_close_matches
import gender_guesser.detector as gender
from datetime import datetime, timedelta
from newspaper import Article
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import spacy
import warnings

# Configuration
API_KEY = "wg8GiGuUTwNfRN90Qmwq"

# Prepare a dictionary of countries with their ISO 3166-1 alpha-2 codes
COUNTRIES = {country.name: country.alpha_2.lower() for country in pycountry.countries}

# Ignore unnecessary warnings
warnings.simplefilter("ignore", category=FutureWarning)

# Load NLP model for text parsing
nlp = spacy.load("en_core_web_sm")

# Gender detection
gender_detector = gender.Detector()

# Sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Domains to exclude
EXCLUDED_DOMAINS = ["consent.yahoo.com"]

# Fraud-related keywords
FRAUD_KEYWORDS = [
    "fraud", "money laundering", "corruption", "financial crime", "embezzlement",
    "tax evasion", "bribery", "scam", "misconduct", "Ponzi scheme", "bank fraud",
    "securities fraud", "tax fraud", "offshore accounts", "insider trading", "market manipulation"
]

# Financial crime context keywords
FRAUD_CONTEXT_WORDS = [
    "financial crime", "money laundering", "embezzlement", "tax fraud", "bribery",
    "securities fraud", "bank fraud", "prosecutors", "charged", "indicted", "Ponzi scheme",
    "illegal transactions", "offshore accounts", "criminal", "felony", "trial"
]

def get_country_code(country_name):
    """
    Converts a country name to its ISO 3166-1 alpha-2 code.
    If the exact country is not found, it searches for the closest match based on letter similarity.
    """
    if not country_name:
        return None

    # Direct lookup
    country_code = COUNTRIES.get(country_name)
    if country_code:
        return country_code

    # Approximate matching
    closest_matches = get_close_matches(country_name, COUNTRIES.keys(), n=1, cutoff=0.6)
    if closest_matches:
        matched_country = closest_matches[0]
        st.warning(f'⚠️ "{country_name}" not found. Using closest match: "{matched_country}".')
        return COUNTRIES[matched_country]

    st.error(f'❌ Country "{country_name}" not found.')
    return None

def search_companies(query, country=None, address=None):
    """
    Searches for companies using the OpenCorporates API.
    """
    country_code = get_country_code(country) if country else None

    params = {
        "q": query,
        "country_code": country_code,
        "registered_address": address,
        "order": 'score',
        "api_token": API_KEY
    }

    params = {key: value for key, value in params.items() if value is not None}

    try:
        response = requests.get("https://api.opencorporates.com/v0.4/companies/search", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error during request: {e}")
        return None

def search_company(company_url):
    """
    Searches for company using the OpenCorporates API.
    """
    params = {
        "api_token": API_KEY
    }

    try:
        response = requests.get(f"https://api.opencorporates.com/v0.4/companies/{company_url.split("/")[-2]}/{company_url.split("/")[-1]}", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error during request: {e}")
        return None

def companies_to_dataframe(api_results):
    """
    Converts OpenCorporates API results into a pandas DataFrame.
    """
    if not api_results or "results" not in api_results or "companies" not in api_results["results"]:
        st.warning("⚠️ No results found.")
        return pd.DataFrame()

    companies = {}
    for company_data in api_results["results"]["companies"]:
        company = company_data["company"]
        company_name = company.get("name", "Unknown Company")

        # Store company details in a dictionary format
        companies[company_name] = {
            "Registration Number": company.get("company_number"),
            "Jurisdiction": company.get("jurisdiction_code").upper(),
            "Incorporation Date": company.get("incorporation_date"),
            "Dissolution Date": company.get("dissolution_date"),
            "Company Type": company.get("company_type"),
            "Status": company.get("current_status"),
            "Address": company.get("registered_address_in_full"),
            "Previous Names": ", ".join(
                [
                    f"{pn.get('company_name', 'Unknown Name')} (from {pn.get('start_date', 'Unknown')} to {pn.get('end_date', 'Unknown')})"
                    for pn in company.get("previous_names", [])
                ]
            ),
            "Industry Descriptions": ", ".join(
                [
                    f"{ic['industry_code'].get('description', 'Unknown')} ({ic['industry_code'].get('code', 'N/A')})"
                    for ic in company.get("industry_codes", [])
                    if "industry_code" in ic
                ]
            ),
            "OpenCorporates URL": company.get("opencorporates_url")
        }

    # Create DataFrame
    df = pd.DataFrame(companies)
    df = df.replace({"": None}).dropna(how="all")
    df = df.T

    return df

def company_to_dataframe(api_results):
    """
    Converts OpenCorporates API results into a detailed pandas DataFrame.
    """
    if not api_results or "results" not in api_results or "company" not in api_results["results"]:
        st.warning("⚠️ No results found.")
        return pd.DataFrame()

    company = api_results["results"]["company"]
    company_name = company.get("name", "Unknown Company")

    def safe_get(dictionary, key, default="Not Available"):
        """
        Safely retrieves a key from a dictionary, returning a default value if missing.
        """
        return dictionary.get(key, default)

    def format_list(data_list, key_name):
        """
        Formats a list of dictionaries into a readable string.
        """
        if isinstance(data_list, list):
            return ", ".join(
                [f"{item.get(key_name, 'Unknown')}" for item in data_list if isinstance(item, dict)]
            )
        return "Not Available"

    # Store company details in a dictionary format
    company_dict = {
        "Registration Number": safe_get(company, "company_number"),
        "Jurisdiction": safe_get(company, "jurisdiction_code", "").upper(),
        "Incorporation Date": safe_get(company, "incorporation_date"),
        "Dissolution Date": safe_get(company, "dissolution_date"),
        "Company Type": safe_get(company, "company_type"),
        "Status": safe_get(company, "current_status"),
        "Registered Address": safe_get(company, "registered_address_in_full"),
        "Registry URL": safe_get(company, "registry_url"),
        "OpenCorporates URL": safe_get(company, "opencorporates_url"),
        "Previous Names": format_list(company.get("previous_names", []), "company_name"),
        "Industry Descriptions": format_list(
            [ic["industry_code"] for ic in company.get("industry_codes", []) if "industry_code" in ic],
            "description"
        ),
        "Officers": format_list(
            [officer["officer"] for officer in company.get("officers", []) if "officer" in officer],
            "name"
        ),
        "Recent Filings": format_list(
            [f["filing"] for f in company.get("filings", []) if "filing" in f],
            "title"
        ),
        "Corporate Groupings": format_list(
            [grouping["corporate_grouping"] for grouping in company.get("corporate_groupings", []) if "corporate_grouping" in grouping],
            "name"
        ),
    }

    # Create DataFrame
    df = pd.DataFrame([company_dict], index=[company_name])
    df = df.replace({"": None}).dropna(axis=1, how="all")
    df = df.T

    return df

def get_pronouns(person_name):
    """
    Detects the person's gender and returns relevant pronouns for fraud detection.
    """
    first_name = person_name.split()[0]
    gender_guess = gender_detector.get_gender(first_name)
    
    if gender_guess in ["male", "mostly_male"]:
        return ["he", "his", "him"]
    elif gender_guess in ["female", "mostly_female"]:
        return ["she", "her", "hers"]
    else:
        return ["he", "his", "him", "she", "her", "hers"]

def extract_fraud_sentences(text, person_name):
    """
    Extracts sentences with fraud-related keywords and checks if they refer to the person.
    """
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
    """
    Ensures fraud is actually related to the person, not just mentioned randomly.
    """
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
    """
    Analyzes only the sentiment of fraud-related sentences.
    """
    sentiment_scores = []
    
    for sentence in fraud_sentences:
        try:
            result = sentiment_pipeline(sentence)
            label = result[0]['label']
            score = {"1 star": -1, "2 stars": -0.5, "3 stars": 0, "4 stars": 0.5, "5 stars": 1}.get(label, 0)
            sentiment_scores.append(score)
        except Exception:
            continue
    
    return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

def fetch_news_articles(person_name):
    """
    Fetches news articles related to the person using NewsAPI.
    """
    date = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")
    url = f"https://newsapi.org/v2/everything?q={person_name}&from={date}&sortBy=popularity&apiKey=4bc3b879ee624f939ec5e7f1c38451ef"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return data.get("articles", [])
    else:
        st.error(f"❌ Error {response.status_code}: {response.text}")
        return []

def process_articles(articles, person_name):
    """
    Processes articles to extract fraud-related information.
    """
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

        except Exception:
            continue

    return filtered_articles

def analyze_fraud_news(person_name):
    """
    Main function to analyze fraud-related news for a given person.
    """
    articles = fetch_news_articles(person_name)

    if not articles:
        st.info("ℹ️ No fraudulent news found.")
        return

    filtered_articles = process_articles(articles, person_name)

    # Compute the average sentiment across all fraud-related articles
    if filtered_articles:
        global_sentiment = sum(article["sentiment"] for article in filtered_articles) / len(filtered_articles)

        st.markdown(
            f"""
            <div style="background-color: #f0f2f6; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
                <p style="margin: 5px 0;">🎯 <b>Global sentiment for "{query}": {global_sentiment:.2f}</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )
        for article in filtered_articles:
            st.markdown(
                f"""
                <div style="background-color: #f0f2f6; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
                    <p style="margin: 5px 0;">📰 <b>{article['title']}</b></p>
                    <p style="margin: 5px 0;">
                        <a href="{article['url']}" target="_blank" style="color: #0078ff; text-decoration: none;">
                            🔗 {article['url']}
                        </a>
                    </p>
                    <p style="margin: 5px 0;">📌 Keywords: {', '.join(article['keywords'])}</p>
                    <p style="margin: 5px 0;">📄 Summary: {article['summary']}</p>
                    <p style="margin: 5px 0;">📊 Sentiment: {article['sentiment']:.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("ℹ️ No relevant fraudulent news found.")

# Application title
st.set_page_config(page_title="Anti-Money Laundering", page_icon="🔍", layout="wide")
st.title("🔍 Anti-Money Laundering")
st.write("This application enables you to combat the risk of money laundering.")

# User inputs
query = st.text_input("🏢 Enter the Company Name:", "BNP Paribas")
country = st.selectbox("🌍 Select the Country (Optional):", ["None"] + sorted(COUNTRIES.keys()), index=0)
country = None if country == "None" else country
address = st.text_input("📍 Enter the Address (Optional):", "")
address = None if address == "" else address

# Search companies
if st.button("🔍 Search Companies"):
    with st.spinner("Retrieving data..."):
        companies_results = search_companies(query=query, country=country, address=address)
        if companies_results:
            company_df = companies_to_dataframe(companies_results)
            st.session_state["company_df"] = company_df
            st.session_state["company_index"] = 0
            st.session_state["company_url"] = company_df["OpenCorporates URL"].iloc[0] if not company_df.empty else None
            st.session_state["company_results"] = None
            st.session_state["selected_name"] = None
        else:
            st.session_state["company_df"] = None
            st.session_state["company_index"] = None

# Display companies
if st.session_state.get("company_df") is not None:
    company_df = st.session_state["company_df"]
    company_dataframe = company_df.drop(columns=["Registration Number", "OpenCorporates URL"], errors="ignore")

    if not company_df.empty:
        st.subheader("📋 Companies Information")
        st.dataframe(company_dataframe, use_container_width=True)

        # Company selection
        company_options = [f"{i + 1} - {name}" for i, name in enumerate(company_df.index)]
        selected_company = st.selectbox("🏢 Select the Company:", company_options, index=st.session_state["company_index"])
        new_index = int(selected_company.split(" - ", 1)[0]) - 1

        if new_index != st.session_state["company_index"]:
            st.session_state["company_index"] = new_index
            st.session_state["company_url"] = company_df["OpenCorporates URL"].iloc[new_index]
            st.session_state["company_results"] = None
            st.session_state["selected_name"] = None

        # Fetch company information
        if st.button("🚀 Fetch Company Information"):
            with st.spinner("Retrieving data..."):
                company_results = search_company(st.session_state["company_url"])
                if company_results:
                    st.session_state["company_results"] = company_results
                else:
                    st.error("❌ No data retrieved from the API.")
    else:
        st.info("ℹ️ No results found for this search.")

# Display company details
if st.session_state.get("company_results"):
    company_results = st.session_state["company_results"]
    df = company_to_dataframe(company_results)

    if not df.empty:
        st.subheader("📋 Company Information")
        st.dataframe(df, use_container_width=True)

        # Name selection
        company_name = df.columns[0]
        officers = df.loc["Officers"].values[0] if "Officers" in df.index else ""
        search_options = [company_name] + officers.split(", ") if officers else [company_name]
        saved_name = st.session_state.get("selected_name", search_options[0])
        if saved_name not in search_options:
            saved_name = search_options[0]
        selected_name = st.selectbox("🔍 Select a Name:", search_options, index=search_options.index(saved_name))
        st.session_state["selected_name"] = selected_name

        # Fetch fraudulent news
        if st.button("📰 Fetch Fraudulent News"):
            with st.spinner("Retrieving data..."):
                analyze_fraud_news(selected_name)
    else:
        st.info("ℹ️ No results found for this search.")

# Authors
st.markdown("""
Made by [Alexandre Deroux](https://www.linkedin.com/in/alexandre-deroux), 
[Alexia Avakian](https://www.linkedin.com/in/alexia-avakian) and 
[Constantin Guillaume](https://www.linkedin.com/in/constantin-guillaume-ldv).
""", unsafe_allow_html=True)