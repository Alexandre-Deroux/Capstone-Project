import requests
import pandas as pd
import streamlit as st
import pycountry
from difflib import get_close_matches
from datetime import datetime, timedelta
from newspaper import Article
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from llama_cpp import Llama

# Configuration
OPENCORPORATES_API_KEY = "wg8GiGuUTwNfRN90Qmwq"

# Prepare a dictionary of countries with their ISO 3166-1 alpha-2 codes
COUNTRIES = {country.name: country.alpha_2.lower() for country in pycountry.countries}

# List of domains to exclude (Yahoo blocks access)
EXCLUDED_DOMAINS = ["consent.yahoo.com"]

# List of keywords related to fraud and money laundering
FRAUD_KEYWORDS = [
    "fraud", "money laundering", "corruption", "scandal", "financial crime", 
    "embezzlement", "tax evasion", "scam", "lawsuit", "investigation"
]

# Download the VADER sentiment analysis model
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Load the GGUF model
llm = Llama(model_path="llama-3.2-3b-instruct.Q4_K_M.gguf", n_ctx=4096)

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
        "api_token": OPENCORPORATES_API_KEY
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
        "api_token": OPENCORPORATES_API_KEY
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
        """Safely retrieves a key from a dictionary, returning a default value if missing."""
        return dictionary.get(key, default)

    def format_list(data_list, key_name):
        """Formats a list of dictionaries into a readable string."""
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

def get_sentiment(text):
    """Calculate the sentiment score of a given text."""
    score = sia.polarity_scores(text)
    return score['compound'] # Score ranges from -1 (negative) to 1 (positive)

def contains_fraud_keywords(text):
    """Check if the article contains any fraud-related keywords."""
    return any(keyword in text.lower() for keyword in FRAUD_KEYWORDS)

def fetch_fraudulent_news(query, days=30):
    """
    Fetch news articles related to the given query from the past specified days, filter those discussing fraud or money laundering with a negative sentiment.
    """
    date = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    url = f"https://newsapi.org/v2/everything?q={query}&from={date}&sortBy=popularity&apiKey=4bc3b879ee624f939ec5e7f1c38451ef"

    response = requests.get(url)

    if response.status_code != 200:
        st.error(f"❌ Error {response.status_code}: {response.text}")
        return []

    data = response.json()
    articles = data.get("articles", [])
    filtered_articles = []

    for article in articles:
        article_url = article.get("url", "")

        # Filter out prohibited URLs
        if any(domain in article_url for domain in EXCLUDED_DOMAINS):
            continue 

        try:
            news_article = Article(article_url)
            news_article.download()
            news_article.parse()
            content = news_article.text

            # Check if the article discusses fraud/money laundering and has a negative sentiment
            sentiment_score = get_sentiment(content)
            if sentiment_score < 0 and contains_fraud_keywords(content):
                filtered_articles.append((article["title"], article_url, sentiment_score))
                st.markdown(
                    f"""
                    <div style="background-color: #f0f2f6; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
                        <p style="margin: 5px 0;">📰 <b>{article['title']}</b></p>
                        <p style="margin: 5px 0;">
                            <a href="{article_url}" target="_blank" style="color: #0078ff; text-decoration: none;">
                                🔗 {article_url}
                            </a>
                        </p>
                        <p style="margin: 5px 0;">📊 Sentiment: {sentiment_score:.2f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        except Exception:
            continue

    # Compute the overall sentiment based on the filtered articles
    if filtered_articles:
        global_sentiment = sum(s[2] for s in filtered_articles) / len(filtered_articles)
        st.markdown(
            f"""
            <div style="background-color: #f0f2f6; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
                <p style="margin: 5px 0;">🎯 <b>Global sentiment for "{query}": {global_sentiment:.2f}</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("ℹ️ No fraudulent news found.")

    return filtered_articles

# Page configuration
st.set_page_config(page_title="Anti-Money Laundering", page_icon="🔍", layout="wide")
st.title("🔍 Anti-Money Laundering")
st.write("This application enables you to combat the risk of money laundering.")

# Function for displaying a company's search fields
def company_search_section(role):
    """
    Displays the company search section (Sender/Recipient).
    """
    query = st.text_input(f"🏢 Enter the {role} Name:", key=f"{role}_query")
    country = st.selectbox(f"🌍 Select the {role} Country (Optional):", ["None"] + sorted(COUNTRIES.keys()), index=0, key=f"{role}_country")
    country = None if country == "None" else country
    address = st.text_input(f"📍 Enter the {role} Address (Optional):", "", key=f"{role}_address")
    address = None if address == "" else address

    if st.button(f"🔍 Search {role} Companies", key=f"{role}_search"):
        with st.spinner("Retrieving data..."):
            companies_results = search_companies(query=query, country=country, address=address)
            if companies_results:
                company_df = companies_to_dataframe(companies_results)
                st.session_state[f"{role}_company_df"] = company_df
                st.session_state[f"{role}_company_index"] = 0
                st.session_state[f"{role}_company_url"] = company_df["OpenCorporates URL"].iloc[0] if not company_df.empty else None
                st.session_state[f"{role}_company_results"] = None
                st.session_state[f"{role}_selected_name"] = None
            else:
                st.session_state[f"{role}_company_df"] = None
                st.session_state[f"{role}_company_index"] = None

# Function for displaying a company's results
def company_results_section(role):
    """
    Displays search results for a company (Sender/Recipient).
    """
    if st.session_state.get(f"{role}_company_df") is not None:
        company_df = st.session_state[f"{role}_company_df"]
        company_dataframe = company_df.drop(columns=["Registration Number", "OpenCorporates URL"], errors="ignore")

        if not company_df.empty:
            st.dataframe(company_dataframe, use_container_width=True)

            # Company selection
            company_options = [f"{i + 1} - {name}" for i, name in enumerate(company_df.index)]
            selected_company = st.selectbox(f"🏢 Select the {role} Company:", company_options, index=st.session_state[f"{role}_company_index"], key=f"{role}_select")
            new_index = int(selected_company.split(" - ", 1)[0]) - 1

            if new_index != st.session_state[f"{role}_company_index"]:
                st.session_state[f"{role}_company_index"] = new_index
                st.session_state[f"{role}_company_url"] = company_df["OpenCorporates URL"].iloc[new_index]
                st.session_state[f"{role}_company_results"] = None
                st.session_state[f"{role}_selected_name"] = None

            # Retrieve company information
            if st.button(f"🚀 Fetch {role} Information", key=f"{role}_fetch"):
                with st.spinner("Retrieving data..."):
                    company_results = search_company(st.session_state[f"{role}_company_url"])
                    if company_results:
                        st.session_state[f"{role}_company_results"] = company_results
                    else:
                        st.error("❌ No data retrieved from the API.")
        else:
            st.info(f"ℹ️ No results found for {role}.")

# Function for displaying company details + fraud search
def company_details_section(role):
    """
    Displays a company's details and allows you to search for fraudulent information.
    """
    if st.session_state.get(f"{role}_company_results"):
        company_results = st.session_state[f"{role}_company_results"]
        df = company_to_dataframe(company_results)

        if not df.empty:
            st.dataframe(df, use_container_width=True)

            # Name selection
            company_name = df.columns[0]
            officers = df.loc["Officers"].values[0] if "Officers" in df.index else ""
            search_options = [company_name] + officers.split(", ") if officers else [company_name]
            saved_name = st.session_state.get(f"{role}_selected_name", search_options[0])
            if saved_name not in search_options:
                saved_name = search_options[0]
            selected_name = st.selectbox(f"🔍 Select a {role} Name:", search_options, index=search_options.index(saved_name), key=f"{role}_name_select")
            st.session_state[f"{role}_selected_name"] = selected_name

            # Search for fraudulent items
            if st.button(f"📰 Fetch {role} Fraudulent News", key=f"{role}_news"):
                with st.spinner("Retrieving data..."):
                    fetch_fraudulent_news(selected_name)
        else:
            st.info(f"ℹ️ No results found for {role}.")

# Function for extracting company information in text form
def extract_company_info(role):
    if st.session_state.get(f"{role}_company_results"):
        df = company_to_dataframe(st.session_state[f"{role}_company_results"])
        if not df.empty:
            return df.to_string(index=True)
    return "No information available."

# Function to generate the LLM prompt
def generate_risk_analysis_prompt(sender_info, recipient_info):
    prompt = f"""
    You are a financial crime expert specializing in Anti-Money Laundering (AML). 
    Analyze the risk of money laundering between the following two entities based on their details:

    📌 **Sender Company Information**:
    {sender_info}

    📌 **Recipient Company Information**:
    {recipient_info}

    🎯 **Task**:
    1. Assign a **risk score from 0 to 100**, where:
    - 0 = No risk
    - 100 = Extremely high risk
    2. Explain the reasoning behind the score.
    3. Identify potential risk indicators (e.g., offshore accounts, politically exposed persons, high-risk industries).
    4. Suggest actions to mitigate the risk.

    Please provide a structured response with the **Risk Score** followed by an **Explanation**.
    """
    return prompt

# Function for analyse AML risk
def analyse_aml_risk(sender_info, recipient_info):
    """
    Analyse the risk of money laundering using Llama 3.2 3B Instruct with llama.cpp.
    """
    prompt = generate_risk_analysis_prompt(sender_info, recipient_info)
    response = llm(prompt, max_tokens=500, temperature=0.3)
    return response["choices"][0]["text"]

# Sender company information
st.header("🚀 Sender Company Information")
company_search_section("Sender")
company_results_section("Sender")
company_details_section("Sender")

# Recipient company information
st.header("🎯 Recipient Company Information")
company_search_section("Recipient")
company_results_section("Recipient")
company_details_section("Recipient")

# AML risk assessment
st.header("⚠️ AML Risk Assessment")
if st.session_state.get("Sender_company_results") and st.session_state.get("Recipient_company_results"):
    sender_info = extract_company_info("Sender")
    recipient_info = extract_company_info("Recipient")

    if st.button("🛡️ Analyse AML Risk"):
        with st.spinner("Analysing money laundering risk..."):
            risk_analysis_result = analyse_aml_risk(sender_info, recipient_info)
            st.subheader("📋 Risk Analysis Result")
            st.write(risk_analysis_result)