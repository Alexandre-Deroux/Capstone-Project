import requests
import pandas as pd
import streamlit as st
import pycountry
from difflib import get_close_matches

# Configuration
API_KEY = "wg8GiGuUTwNfRN90Qmwq"

# Prepare a dictionary of countries with their ISO 3166-1 alpha-2 codes
COUNTRIES = {country.name: country.alpha_2.lower() for country in pycountry.countries}

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
        print(f"❌ Error during request: {e}")
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
            "Industry Codes": ", ".join(
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
    df = df.dropna(how="all")
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
        "Industry Codes": format_list(
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
        else:
            st.session_state["company_df"] = None
            st.session_state["company_index"] = None

# Display companies
if "company_df" in st.session_state and st.session_state["company_df"] is not None:
    company_df = st.session_state["company_df"]
    company_dataframe = company_df.drop(columns=["Registration Number", "OpenCorporates URL"], errors="ignore")
    if not company_df.empty:
        st.subheader("📋 Companies Information")
        st.dataframe(company_dataframe, use_container_width=True)
        if "company_index" not in st.session_state:
            st.session_state["company_index"] = 0
        if "company_url" not in st.session_state:
            st.session_state["company_url"] = company_df["OpenCorporates URL"].iloc[0]
        company_options = [f"{i + 1} - {name}" for i, name in enumerate(company_df.index)]
        selected_company = st.selectbox("🏢 Select the Company:", company_options, index=st.session_state["company_index"])
        new_index = int(selected_company.split(" - ", 1)[0]) - 1
        if new_index != st.session_state["company_index"]:
            st.session_state["company_index"] = new_index
            st.session_state["company_url"] = company_df["OpenCorporates URL"].iloc[new_index]

        # Display company
        if st.button("🚀 Fetch Company Information") and "company_url" in st.session_state:
            with st.spinner("Retrieving data..."):
                company_results = search_company(st.session_state["company_url"])
                if company_results:
                    df = company_to_dataframe(company_results)
                    if not df.empty:
                        st.subheader("📋 Company Information")
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("ℹ️ No results found for this search.")
                else:
                    st.error("❌ No data retrieved from the API.")
    else:
        st.info("ℹ️ No results found for this search.")

# Authors
st.markdown("""
Made by [Alexandre Deroux](https://www.linkedin.com/in/alexandre-deroux), 
[Alexia Avakian](https://www.linkedin.com/in/alexia-avakian), and 
[Constantin Guillaume](https://www.linkedin.com/in/constantin-guillaume-ldv).
""", unsafe_allow_html=True)