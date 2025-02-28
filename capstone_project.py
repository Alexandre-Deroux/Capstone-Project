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
OPENSANCTIONS_API_KEY = "ae47a6e429340b312bb752205031da77"
NEWS_API_KEY = "4bc3b879ee624f939ec5e7f1c38451ef"

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
            "Previous Names": "; ".join(
                [
                    f"{pn.get('company_name', 'Unknown Name')} (from {pn.get('start_date', 'Unknown')} to {pn.get('end_date', 'Unknown')})"
                    for pn in company.get("previous_names", [])
                ]
            ),
            "Industry Descriptions": "; ".join(
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

def check_sanctions(entity_type, name):
    """
    Queries OpenSanctions API to check if a person or company is sanctioned, with score > 0.70.
    """
    session = requests.Session()
    session.headers["Authorization"] = f"ApiKey {OPENSANCTIONS_API_KEY}"
    query = {"schema": entity_type, "properties": {"name": [name]}}
    batch = {"queries": {"q1": query}}

    try:
        response = session.post("https://api.opensanctions.org/match/sanctions?algorithm=best", json=batch)
        response.raise_for_status()
        responses = response.json().get("responses", {})
        results = responses.get("q1", {}).get("results", [])
        high_confidence_results = [res for res in results if res.get("score", 0) > 0.70]
        return high_confidence_results
    except requests.exceptions.RequestException as e:
        st.error(f"❌ OpenSanctions API Error: {e}")
        return None

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
            return "; ".join(
                [f"{item.get(key_name, 'Unknown')}" for item in data_list if isinstance(item, dict)]
            )
        return "Not Available"

    # Check sanctions
    sanctions = {"company": None, "officers": {}}
    company_sanctions = check_sanctions("Company", company_name)
    sanctioned_company = "No Sanctions" if not company_sanctions else "Sanctioned"
    sanctions["company"] = company_sanctions
    officers_list = [
        officer["officer"]["name"] for officer in company.get("officers", []) if "officer" in officer
    ]
    sanctioned_officers = []
    for officer in officers_list:
        officer_sanctions = check_sanctions("Person", officer)
        if officer_sanctions:
            sanctioned_officers.append(f"{officer}: Sanctioned")
        sanctions["officers"][officer] = officer_sanctions
    sanctioned_officers_display = "; ".join(sanctioned_officers) if sanctioned_officers else "No Sanctions" if officers_list else ""

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
        "Sanctioned Company": sanctioned_company,
        "Officers": format_list(
            [officer["officer"] for officer in company.get("officers", []) if "officer" in officer],
            "name"
        ),
        "Sanctioned Officers": sanctioned_officers_display,
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

    return df, sanctions

def display_sanctions_results(sanctions):
    """
    Displays sanctions results in the Streamlit UI.
    """
    # Check company sanctions
    company_sanctions = sanctions.get("company", None)
    if not company_sanctions:
        st.markdown(
            """
            <div style="background-color: #d4edda; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
                ✅ <b>The company is not sanctioned.</b>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="background-color: #f8d7da; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
                ⚠️ <b>The company is sanctioned!</b>
            </div>
            """,
            unsafe_allow_html=True
        )
        for sanction in company_sanctions:
            sanctioned_name = sanction.get("caption", "Unknown")
            topics = "; ".join(sanction.get("properties", {}).get("topics", ["Unknown"]))
            datasets = "; ".join(sanction.get("datasets", ["Unknown"]))
            source_links = sanction.get("properties", {}).get("sourceUrl", [])

            st.markdown(
                f"""
                <div style="background-color: #f0f2f6; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
                    <p style="margin: 5px 0;">📌 <b>Match found:</b> {sanctioned_name}</p>
                    <p style="margin: 5px 0;">📂 <b>Sanction topics:</b> {topics.capitalize()}</p>
                    <p style="margin: 5px 0;">📜 <b>Sanctions list:</b> {datasets}</p>
                    {"".join(f'<p style="margin: 5px 0;"><a href="{source_link}" target="_blank" style="color: #0078ff; text-decoration: none;">🔗 View source</a></p>' for source_link in source_links) if source_links else ""}
                </div>
                """,
                unsafe_allow_html=True
            )

    # Check officer sanctions
    sanctioned_officers = {officer: details for officer, details in sanctions.get("officers", {}).items() if details}
    if not sanctioned_officers:
        st.markdown(
            """
            <div style="background-color: #d4edda; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
                ✅ <b>No officers are sanctioned.</b>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        for officer, officer_sanctions in sanctioned_officers.items():
            st.markdown(
                f"""
                <div style="background-color: #f8d7da; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
                    👤 <b>{officer} is sanctioned!</b>
                </div>
                """,
                unsafe_allow_html=True
            )
            for sanction in officer_sanctions:
                sanctioned_name = sanction.get("caption", "Unknown")
                topics = "; ".join(sanction.get("properties", {}).get("topics", ["Unknown"]))
                datasets = "; ".join(sanction.get("datasets", ["Unknown"]))
                source_links = sanction.get("properties", {}).get("sourceUrl", [])

                st.markdown(
                    f"""
                    <div style="background-color: #f0f2f6; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
                        <p style="margin: 5px 0;">📌 <b>Match found:</b> {sanctioned_name}</p>
                        <p style="margin: 5px 0;">📂 <b>Sanction topics:</b> {topics.capitalize()}</p>
                        <p style="margin: 5px 0;">📜 <b>Sanctions list:</b> {datasets}</p>
                        {"".join(f'<p style="margin: 5px 0;"><a href="{source_link}" target="_blank" style="color: #0078ff; text-decoration: none;">🔗 View source</a></p>' for source_link in source_links) if source_links else ""}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

def get_sentiment(text):
    """
    Calculate the sentiment score of a given text.
    """
    score = sia.polarity_scores(text)
    return score['compound'] # Score ranges from -1 (negative) to 1 (positive)

def contains_fraud_keywords(text):
    """
    Check if the article contains any fraud-related keywords.
    """
    return any(keyword in text.lower() for keyword in FRAUD_KEYWORDS)

def fetch_adverse_news(query, days=30):
    """
    Fetch news articles related to the given query from the past specified days, filter those discussing fraud or money laundering with a negative sentiment.
    """
    date = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    url = f"https://newsapi.org/v2/everything?q={query}&from={date}&sortBy=popularity&apiKey={NEWS_API_KEY}"

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
        st.info("ℹ️ No adverse news found.")

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
                companies_df = companies_to_dataframe(companies_results)
                st.session_state[f"{role}_companies_df"] = companies_df
                st.session_state[f"{role}_company_index"] = 0
                st.session_state[f"{role}_company_url"] = companies_df["OpenCorporates URL"].iloc[0] if not companies_df.empty else None
                st.session_state[f"{role}_company_results"] = None
                st.session_state[f"{role}_company_df"] = None
                st.session_state[f"{role}_selected_name"] = None
            else:
                st.session_state[f"{role}_companies_df"] = None
                st.session_state[f"{role}_company_index"] = None

# Function for displaying a company's results
def company_results_section(role):
    """
    Displays search results for a company (Sender/Recipient).
    """
    if st.session_state.get(f"{role}_companies_df") is not None:
        companies_df = st.session_state[f"{role}_companies_df"]
        company_dataframe = companies_df.drop(columns=["Registration Number", "OpenCorporates URL"], errors="ignore")

        if not companies_df.empty:
            st.dataframe(company_dataframe, use_container_width=True)

            # Company selection
            company_options = [f"{i + 1} - {name}" for i, name in enumerate(companies_df.index)]
            selected_company = st.selectbox(f"🏢 Select the {role} Company:", company_options, index=st.session_state[f"{role}_company_index"], key=f"{role}_select")
            new_index = int(selected_company.split(" - ", 1)[0]) - 1

            if new_index != st.session_state[f"{role}_company_index"]:
                st.session_state[f"{role}_company_index"] = new_index
                st.session_state[f"{role}_company_url"] = companies_df["OpenCorporates URL"].iloc[new_index]
                st.session_state[f"{role}_company_results"] = None
                st.session_state[f"{role}_company_df"] = None
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
    Displays a company's details and allows you to search for adverse information.
    """
    if st.session_state.get(f"{role}_company_results"):
        company_results = st.session_state[f"{role}_company_results"]
        df, sanctions = company_to_dataframe(company_results)

        if not df.empty:
            st.session_state[f"{role}_company_df"] = df
            st.dataframe(df, use_container_width=True)
            if st.button(f"📋 Display {role} Sanctions Results", key=f"{role}_sanctions"):
                with st.spinner("Retrieving data..."):
                    display_sanctions_results(sanctions)

            # Name selection
            company_name = df.columns[0]
            officers = df.loc["Officers"].values[0] if "Officers" in df.index else ""
            search_options = [company_name] + [officer for officer in officers.split("; ")] if officers else [company_name]
            saved_name = st.session_state.get(f"{role}_selected_name", search_options[0])
            if saved_name not in search_options:
                saved_name = search_options[0]
            selected_name = st.selectbox(f"🔍 Select a {role} Name:", search_options, index=search_options.index(saved_name), key=f"{role}_name_select")
            st.session_state[f"{role}_selected_name"] = selected_name

            # Search for adverse items
            if st.button(f"📰 Fetch {role} Adverse News", key=f"{role}_news"):
                with st.spinner("Retrieving data..."):
                    fetch_adverse_news(selected_name)
        else:
            st.info(f"ℹ️ No results found for {role}.")

# Function for extracting company information in text form
def extract_company_info(role):
    df = st.session_state.get(f"{role}_company_df", None)
    if df is not None and not df.empty:
        return df.to_string(index=True)
    return "No information available."

# Function to generate the LLM prompt
# def generate_risk_analysis_prompt(sender_info, recipient_info, transaction_info):
#     prompt = f"""
#     You are a financial crime expert specializing in Anti-Money Laundering (AML). 
#     Analyze the risk of money laundering between the following two entities based on their details:

#     📌 **Sender Company Information**:
#     {sender_info}

#     📌 **Recipient Company Information**:
#     {recipient_info}

#     📌 **Transaction Information**:
#     {transaction_info}

#     🎯 **Task**:
#     1. Assign a **risk score from 0 to 100**, where:
#     - 0 = No risk
#     - 100 = Extremely high risk
#     2. Explain the reasoning behind the score.
#     3. Identify potential risk indicators (e.g., offshore accounts, politically exposed persons, high-risk industries).
#     4. Suggest actions to mitigate the risk.

#     Please provide a structured response with the **Risk Score** followed by an **Explanation**.
#     """
#     return prompt
def generate_risk_analysis_prompt(sender_info, recipient_info, transaction_info):
    prompt = f"""
    You are a financial crime expert specialising in Anti-Money Laundering (AML).  
    Analyse the risk of money laundering for the following transaction based on entity details, transaction characteristics, and risk indicators.

    ---
    ### 📌 **Step 1: Company & Transaction Information**
    #### 🔹 **Sender Company Information:**
    {sender_info}

    #### 🔹 **Recipient Company Information:**
    {recipient_info}

    #### 🔹 **Transaction Information:**
    {transaction_info}

    ---
    ### 📌 **Step 2: Red Flags Assessment**
    Evaluate the presence of the following high-risk attributes and quantify them on a scale of **0 (Not Present) to 5 (Highly Present)**:
    1. **Sensitive Jurisdiction** (e.g., sanctioned countries, high-risk regions)
    2. **Tax Haven** (e.g., offshore entities, shell companies)
    3. **Multiple Locations** (e.g., complex multi-national structures)
    4. **Sensitive Legal Structure** (e.g., trusts, foundations, bearer shares)

    **Scoring Instructions:**
    - **0:** Not detected
    - **1-2:** Low relevance
    - **3-4:** Moderate relevance
    - **5:** Highly relevant

    🔍 **Provide a breakdown of these scores and justify each rating.**

    ---
    ### 📌 **Step 3: Transaction Rationale Assessment**
    Classify the transaction under one of the following rationales:
    1. **Client-Supplier** (e.g., commercial payment for goods/services)
    2. **Client-FSP (Financial Service Provider)** (e.g., bank, payment institution)
    3. **Client-PSP/MSB (Payment Service Provider/Money Service Business)**
    4. **Client-Individual** (e.g., personal remittance, salary payment)
    5. **Other** (Provide details)
    6. **Not Established** (Lack of transparent justification)

    🔍 **Explain how well the transaction rationale aligns with the sender/recipient profile.**

    ---
    ### 📌 **Step 4: Mitigating Factors Assessment**
    Determine if any of the following **risk-reducing factors** are present:
    1. **Inter-Account** (Internal transfer within the same entity)
    2. **Intra-Group** (Transaction between entities within the same corporate group)
    3. **G-SIFI (Globally Systemically Important Financial Institution)** (Highly regulated institution)

    🔍 **If applicable, explain how these factors reduce overall risk.**

    ---
    ### 📌 **Step 5: Risk Score & Explanation**
    Using the **company and transaction information (Step 1), quantified red flags (Step 2), transaction rationale (Step 3), and mitigating factors (Step 4)**, assign a **risk score from 0 to 100**:
    - **0-20:** Low Risk
    - **21-50:** Moderate Risk
    - **51-80:** High Risk
    - **81-100:** Critical Risk

    📊 **Final Rating:** _XX/100_
    📝 **Risk Justification:**
    - Summarise key findings from previous steps.
    - Highlight inconsistencies or suspicious patterns.
    """
    return prompt

# Function for analyse AML risk
def analyse_aml_risk(sender_info, recipient_info, transaction_info):
    """
    Analyse the risk of money laundering using Llama 3.2 3B Instruct with llama.cpp.
    """
    prompt = generate_risk_analysis_prompt(sender_info, recipient_info, transaction_info)
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

# Transaction information
st.header("💰 Transaction Information")
amount = st.number_input("💵 Enter the Amount:", min_value=0.01, value=1000.00, step=1.00, key="amount")
currency = st.selectbox("💱 Select the Currency:", ["USD", "EUR", "GBP", "CHF", "JPY", "CNY", "AUD"], key="currency")
reference = st.text_input("📜 Enter the Reference (Optional):", key="reference")
reason = st.text_area("📝 Enter the Reason (Optional):", key="reason")
transaction_info = {"Amount": amount, "Currency": currency, "Reference": reference, "Reason": reason}

# AML risk assessment
st.header("⚠️ AML Risk Assessment")
if st.session_state.get("Sender_company_results") and st.session_state.get("Recipient_company_results"):
    sender_info = extract_company_info("Sender")
    recipient_info = extract_company_info("Recipient")

    if st.button("🛡️ Analyse AML Risk"):
        with st.spinner("Analysing money laundering risk..."):
            risk_analysis_result = analyse_aml_risk(sender_info, recipient_info, transaction_info)
            st.subheader("📋 Risk Analysis Result")
            st.markdown(
                f"""
                <div style="background-color: #f0f2f6; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
                    <p style="margin: 5px 0;">{risk_analysis_result}</p>
                </div>
                """,
                unsafe_allow_html=True
            )