import requests
import pandas as pd
import streamlit as st

def get_lei_information(lei):
    """
    Retrieves company information based on an LEI identifier via the GLEIF API.

    :param lei: (str) The LEI identifier of the company
    :return: (dict) Company data or an error message
    """
    # API URL with the provided LEI
    url = f"https://api.gleif.org/api/v1/lei-records/{lei}"

    # Headers to specify the JSON API format
    headers = {
        "Accept": "application/vnd.api+json"
    }

    try:
        # Make a GET request to the API
        response = requests.get(url, headers=headers)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Return the JSON data from the response
            return response.json()
        else:
            # If the status code is not 200, return an error
            return {
                "error": f"Error retrieving data (status code {response.status_code}).",
                "details": response.text
            }
    except requests.exceptions.RequestException as e:
        # Handle errors related to the request (e.g., network issues)
        return {
            "error": "An error occurred during the request.",
            "details": str(e)
        }

def json_to_dataframe(data):
    """
    Transforms a JSON response from the GLEIF API into a Pandas DataFrame
    with all available information organised into separate columns.

    :param data: (dict) JSON data from the GLEIF API
    :return: (pd.DataFrame) DataFrame with the extracted information
    """
    # Extracting the main parts of the JSON response
    attributes = data["data"]["attributes"]
    entity = attributes["entity"]
    registration = attributes["registration"]

    # Flattening the data into a dictionary
    flat_data = {
        # General LEI Information
        "LEI": attributes["lei"],
        "Legal Name": entity["legalName"]["name"],
        "Legal Name Language": entity["legalName"]["language"],
        "Transliterated Legal Name": entity["transliteratedOtherNames"][0]["name"] if entity["transliteratedOtherNames"] else None,
        "Transliterated Legal Name Language": entity["transliteratedOtherNames"][0]["language"] if entity["transliteratedOtherNames"] else None,
        "Transliterated Legal Name Type": entity["transliteratedOtherNames"][0]["type"] if entity["transliteratedOtherNames"] else None,
        "Legal Address": ", ".join(entity["legalAddress"]["addressLines"]),
        "Legal Address Language": entity["legalAddress"]["language"],
        "Legal Address City": entity["legalAddress"]["city"],
        "Legal Address Region": entity["legalAddress"]["region"],
        "Legal Address Country": entity["legalAddress"]["country"],
        "Legal Address Postal Code": entity["legalAddress"]["postalCode"],
        "Headquarters Address": ", ".join(entity["headquartersAddress"]["addressLines"]),
        "Headquarters Address Language": entity["headquartersAddress"]["language"],
        "Headquarters Address City": entity["headquartersAddress"]["city"],
        "Headquarters Address Region": entity["headquartersAddress"]["region"],
        "Headquarters Address Country": entity["headquartersAddress"]["country"],
        "Headquarters Address Postal Code": entity["headquartersAddress"]["postalCode"],
        "Jurisdiction": entity["jurisdiction"],
        "Category": entity["category"],
        "Legal Form ID": entity["legalForm"]["id"],
        "Registered At ID": entity["registeredAt"]["id"],
        "Registered As": entity["registeredAs"],
        "Entity Status": entity["status"],
        "Creation Date": entity["creationDate"],
        "Expiration Date": entity["expiration"]["date"],
        "Expiration Reason": entity["expiration"]["reason"],
        "Associated Entity LEI": entity["associatedEntity"]["lei"],
        "Associated Entity Name": entity["associatedEntity"]["name"],
        "BIC Codes": ", ".join(attributes["bic"]) if attributes["bic"] else None,
        "MIC": attributes["mic"],
        "OCID": attributes["ocid"],
        "SP Global IDs": ", ".join(attributes["spglobal"]) if attributes["spglobal"] else None,
        "Conformity Flag": attributes["conformityFlag"],
        
        # Registration Information
        "Initial Registration Date": registration["initialRegistrationDate"],
        "Last Update Date": registration["lastUpdateDate"],
        "Next Renewal Date": registration["nextRenewalDate"],
        "Registration Status": registration["status"],
        "Managing LOU": registration["managingLou"],
        "Corroboration Level": registration["corroborationLevel"],
        "Validated At ID": registration["validatedAt"]["id"],
        "Validated As": registration["validatedAs"],
    }

    # Converting the flattened dictionary into a DataFrame
    df = pd.DataFrame([flat_data])

    return df.set_index("Legal Name")

def fetch_relationship_data(base_data):
    """
    Fetches data for all relationships in the 'relationships' section
    by performing API requests to the provided links.

    :param base_data: (dict) JSON data from the original API response.
    :return: (pd.DataFrame) DataFrame containing data for all relationships.
    """
    relationships = base_data["data"].get("relationships", {})
    results = []

    for rel_name, rel_data in relationships.items():
        # Extract the related link
        related_link = rel_data.get("links", {}).get("related")
        if related_link:
            try:
                # Perform the API request
                response = requests.get(related_link)
                response.raise_for_status()  # Raise an exception for HTTP errors
                fetched_data = response.json()  # Parse the JSON response

                # Append the result to the list as a dictionary
                results.append({
                    "Relationship Name": rel_name,
                    "Link": related_link,
                    "Data": fetched_data
                })

            except requests.exceptions.RequestException as e:
                # Log any errors that occur during the request
                results.append({
                    "Relationship Name": rel_name,
                    "Link": related_link,
                    "Error": str(e)
                })

    # Convert the results into a DataFrame for organisation
    return pd.DataFrame(results)

def extract_related_leis(relationship_data, input_lei):
    """
    Extracts all unique LEIs from relationship data that are different from the input LEI.

    :param relationship_data: (pd.Series) The "Data" column from the relationships DataFrame.
    :param input_lei: (str) The LEI given as input (to exclude from results).
    :return: (pd.DataFrame) DataFrame containing unique LEIs and their relationship details.
    """
    related_leis = []

    # Loop through each relationship in the data
    for relationship in relationship_data:
        # Handle the "data" field, which can be a list or a dictionary
        data_field = relationship.get("data")

        if isinstance(data_field, list):  # If "data" is a list
            for entry in data_field:
                lei = entry.get("attributes", {}).get("lei")
                if lei and lei != input_lei:
                    related_leis.append(lei)

        elif isinstance(data_field, dict):  # If "data" is a single object
            lei = data_field.get("attributes", {}).get("lei")
            if lei and lei != input_lei:
                related_leis.append(lei)

    return list(set(related_leis))

# Application title
st.set_page_config(page_title="LEI Lookup Service", layout="wide")
st.title("🔍 LEI Lookup Service")
st.write("This application allows you to search for company information using its **LEI (Legal Entity Identifier)**.")

# User input
input_lei = st.text_input("Enter the LEI to look up:", "529900W18LQJJN6SJ336")

# Add a button to trigger the search
if st.button("🚀 Fetch Company Information"):
    with st.spinner("Retrieving data..."):
        # Placeholder function to fetch LEI information
        data = get_lei_information(input_lei)

        if "error" in data:
            st.error(f"❌ Error: {data['error']}")
            st.write(f"🔍 Details: {data['details']}")
        else:
            # Display company information
            st.subheader("📋 Company Information")
            company_df = json_to_dataframe(data)
            st.dataframe(company_df.T.dropna(how="all").style.format(na_rep="N/A"), use_container_width=True)

            # Fetch relationships
            st.subheader("🔗 Company Relationships")
            relationship_df = fetch_relationship_data(data)

            if not relationship_df.empty:
                # Extract related LEIs
                relationship_data = relationship_df["Data"]
                related_leis = extract_related_leis(relationship_data, input_lei)
                
                relationship_df_display = pd.DataFrame()
                for related_lei in related_leis:
                    data = get_lei_information(related_lei)
                    if "error" in data:
                        st.error(f"❌ Error: {data['error']}")
                        st.write(f"🔍 Details: {data['details']}")
                    else:
                        relationship_df_display = pd.concat(
                            [relationship_df_display, json_to_dataframe(data)], ignore_index=False
                        )
                        
                if not relationship_df_display.empty:
                    st.dataframe(relationship_df_display.T.dropna(how="all").style.format(na_rep="N/A"), use_container_width=True)
                else:
                    st.info("ℹ️ No relationships found.")
            else:
                st.info("ℹ️ No relationships found.")

# Authors
st.markdown("""
Made by [Alexandre Deroux](https://www.linkedin.com/in/alexandre-deroux), 
[Alexia Avakian](https://www.linkedin.com/in/alexia-avakian), and 
[Constantin Guillaume](https://www.linkedin.com/in/constantin-guillaume-ldv).
""", unsafe_allow_html=True)