import requests
import streamlit as st

OPENSANCTIONS_API_URL = "https://api.opensanctions.org/match/sanctions?algorithm=best"
OPENSANCTIONS_API_KEY = "ae47a6e429340b312bb752205031da77"

session = requests.Session()
session.headers["Authorization"] = f"ApiKey {OPENSANCTIONS_API_KEY}"

def check_sanctions(entity_type, name):
    """Queries OpenSanctions API to check if a person or company is sanctioned, with score > 0.70 """
    query = {"schema": entity_type, "properties": {"name": [name]}}
    batch = {"queries": {"q1": query}}

    try:
        response = session.post(OPENSANCTIONS_API_URL, json=batch)
        response.raise_for_status()
        responses = response.json().get("responses", {})
        results = responses.get("q1", {}).get("results", [])

        # filter results to only include matches with score > 0.70
        high_confidence_results = [res for res in results if res.get("score", 0) > 0.70]

        return high_confidence_results
    except requests.exceptions.RequestException as e:
        st.error(f"❌ OpenSanctions API Error: {e}")
        return None

st.set_page_config(page_title="🚨 AML OpenSanctions Check", layout="centered")
st.title("🚨 AML OpenSanctions Check")
st.markdown("Enter a name to check if the individual or company is sanctioned")

if "sanctions_data" not in st.session_state:
    st.session_state["sanctions_data"] = None

entity_type = st.selectbox("Select Entity Type:", ["Person", "Company"])

entity_name = st.text_input("🔍 Enter Name:")

if st.button("Check Sanctions"):
    if entity_name:
        with st.spinner("Checking..."):
            sanctions = check_sanctions(entity_type, entity_name)
            st.session_state["sanctions_data"] = sanctions

            if not sanctions:
                st.success(f"✅ {entity_name} is NOT sanctioned for AML related offenses")
            else:
                st.error(f"⚠️ {entity_name} is sanctioned for financial crime or AML violations !")

sanctions = st.session_state["sanctions_data"]

if sanctions:
    if st.button("🔍 View Additional Details"):
        for sanction in sanctions:
            sanctioned_name = sanction.get("caption", "Unknown")
            topics = ", ".join(sanction.get("properties", {}).get("topics", ["Unknown"]))
            dataset = ", ".join(sanction.get("datasets", ["Unknown"]))
            source_links = sanction.get("properties", {}).get("sourceUrl", [])

            st.write(f"📌 **Match Found:** {sanctioned_name}")
            st.write(f"📂 **Sanction Topics:** {topics}")
            st.write(f"📜 **Sanction Dataset:** {dataset}")

            if source_links:
                st.markdown("🔗 **Source Links:**")
                for link in source_links:
                    st.markdown(f"- [🔍 View Source]({link})")
            else:
                st.write("❌ No source links available.")
