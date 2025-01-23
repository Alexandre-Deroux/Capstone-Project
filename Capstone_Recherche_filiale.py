import requests
import pandas as pd
import streamlit as st

def search_company_by_name_and_country(query, country_code):
    """
    Recherche une entreprise en fonction de son nom et de son pays via l'API de la GLEIF.
    Si aucune correspondance exacte n'est trouvée, retourne le meilleur match dans le pays spécifié.
    Si aucun pays ne correspond, propose une liste de choix à l'utilisateur.

    :param query: (str) Le nom de l'entreprise.
    :param country_code: (str) Le code ISO 3166-1 alpha-2 du pays.
    :return: (dict) Dictionnaire contenant les informations de l'entreprise sélectionnée.
    """
    base_url = "https://api.gleif.org/api/v1/lei-records"
    params = {
        "filter[fulltext]": query
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        if data['data']:
            # Parcourir les résultats pour trouver une correspondance exacte
            for company in data['data']:
                country = company['attributes']['entity']['legalAddress']['country']
                if country == country_code and query.lower() in company['attributes']['entity']['legalName']['name'].lower():
                    return company
            
            # Si aucune correspondance exacte, retourner le premier résultat dans le bon pays
            matches_in_country = [
                company for company in data['data']
                if company['attributes']['entity']['legalAddress']['country'] == country_code
            ]
            if matches_in_country:
                best_match = matches_in_country[0]
                st.warning(f"Aucune correspondance exacte trouvée. Meilleur match : {best_match['attributes']['entity']['legalName']['name']}")
                return best_match

            # Si aucun résultat ne correspond au pays donné, proposer une liste à l'utilisateur
            best_match=data['data'][0]
            st.warning(f"Aucun résultat trouvé pour le pays {country_code}. Meilleur Match:{best_match['attributes']['entity']['legalName']['name']}")
            
            return best_match
        else:
            st.warning("Aucun enregistrement correspondant trouvé.")
            return None
    else:
        st.error(f"Erreur lors de la requête : {response.status_code}")
        return None





# Fonction pour obtenir les informations complètes d'une entreprise via son LEI
def get_lei_information(lei):
    url = f"https://api.gleif.org/api/v1/lei-records/{lei}"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        return None


# Fonction pour obtenir la maison mère d'une entreprise via son LEI
def get_parent_entity(lei):
    base_url = "https://api.gleif.org/api/v1/lei-records"
    params = {
        "filter[owns]": lei
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        if data['data']:
            return data['data'][0]  # Retourner la première maison mère trouvée
        else:
            return None
    else:
        st.error(f"Erreur lors de la requête : {response.status_code}")
        return None


# Fonction pour obtenir les filiales d'une entreprise via son LEI
def get_child_entities(lei):
    base_url = "https://api.gleif.org/api/v1/lei-records"
    params = {
        "filter[ownedBy]": lei
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        if data['data']:
            return data['data']
        else:
            return []
    else:
        st.error(f"Erreur lors de la requête : {response.status_code}")
        return []


# Fonction pour transformer JSON en DataFrame
def json_to_dataframe(data):
    attributes = data["data"]["attributes"]
    entity = attributes["entity"]
    registration = attributes["registration"]

    flat_data = {
        "LEI": attributes["lei"],
        "Legal Name": entity["legalName"]["name"],
        "Legal Address": ", ".join(entity["legalAddress"]["addressLines"]),
        "City": entity["legalAddress"]["city"],
        "Country": entity["legalAddress"]["country"],
        "Postal Code": entity["legalAddress"]["postalCode"],
        "Headquarters Address": ", ".join(entity["headquartersAddress"]["addressLines"]),
        "Headquarters City": entity["headquartersAddress"]["city"],
        "Headquarters Country": entity["headquartersAddress"]["country"],
        "Initial Registration Date": registration["initialRegistrationDate"],
        "Last Update Date": registration["lastUpdateDate"],
        "Next Renewal Date": registration["nextRenewalDate"],
        "Registration Status": registration["status"],
    }

    return pd.DataFrame([flat_data])


# Fonction principale de l'application
def main():
    st.set_page_config(page_title="LEI Lookup Service", page_icon="🔍", layout="wide")
    st.title("🔍 LEI Lookup Service")
    st.write("Recherchez des informations sur les entreprises en utilisant leur **LEI (Legal Entity Identifier)**.")

    # Entrées utilisateur
    input_company = st.text_input("Nom de l'entreprise :", "ALBIS PLASTIC SRL")
    input_country = st.text_input("Code pays (ISO 3166-1 alpha-2) :", "RO")

    # Bouton de recherche
    if st.button("🔍 Rechercher"):
        with st.spinner("Recherche en cours..."):
            company = search_company_by_name_and_country(input_company, input_country)
            if company:
                company_lei = company['id']
                company_name = company['attributes']['entity']['legalName']['name']
                st.success(f"Entreprise trouvée : {company_name} (LEI : {company_lei})")

                # Obtenir les informations complètes de l'entreprise
                company_info = get_lei_information(company_lei)
                if company_info:
                    st.subheader(f"Informations sur {company_name}")
                    st.dataframe(json_to_dataframe(company_info))

                # Vérifier si l'entreprise a une maison mère
                st.subheader("Recherche de la maison mère...")
                parent = get_parent_entity(company_lei)
                if parent:
                    parent_name = parent['attributes']['entity']['legalName']['name']
                    parent_lei = parent['id']
                    st.success(f"Maison mère trouvée : {parent_name} (LEI : {parent_lei})")

                    # Afficher les informations sur la maison mère
                    parent_info = get_lei_information(parent_lei)
                    if parent_info:
                        st.subheader(f"Informations sur la maison mère : {parent_name}")
                        st.dataframe(json_to_dataframe(parent_info))

                    # Rechercher les filiales de la maison mère
                    st.subheader(f"Filiales de la maison mère : {parent_name}")
                    children = get_child_entities(parent_lei)
                else:
                    # Si pas de maison mère, rechercher les filiales de l'entreprise
                    st.info(f"Aucune maison mère trouvée pour {company_name}. Recherche des filiales...")
                    children = get_child_entities(company_lei)

                # Afficher les filiales
                if children:
                    for child in children:
                        child_name = child['attributes']['entity']['legalName']['name']
                        child_lei = child['id']
                        st.write(f"- {child_name} (LEI : {child_lei})")

                        # Afficher les informations détaillées des filiales
                        child_info = get_lei_information(child_lei)
                        if child_info:
                            st.dataframe(json_to_dataframe(child_info))
                else:
                    st.info("Aucune filiale trouvée.")
            else:
                st.error("Aucune entreprise correspondante trouvée pour les critères spécifiés.")


# Exécuter l'application
if __name__ == "__main__":
    main()
