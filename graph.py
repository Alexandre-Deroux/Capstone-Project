import streamlit as st
import requests
import pycountry
import pandas as pd
from pyvis.network import Network
import tempfile, os
from difflib import get_close_matches

# ================================
# 1) CONFIGURATION & OUTILS
# ================================
OPENCORPORATES_API_KEY = "wg8GiGuUTwNfRN90Qmwq"
ALL_COUNTRIES = ["(Aucun)"] + sorted([c.name for c in pycountry.countries])

# ================================
# 2) FONCTIONS OPEN CORPORATES
# ================================
def get_country_code(country_name):
    """Convertit le nom d'un pays en code ISO 3166-1 alpha-2 (minuscules)."""
    if not country_name or country_name == "(Aucun)":
        return None
    countries_dict = {c.name: c.alpha_2.lower() for c in pycountry.countries}
    if country_name in countries_dict:
        return countries_dict[country_name]
    closest = get_close_matches(country_name, countries_dict.keys(), n=1, cutoff=0.6)
    if closest:
        st.warning(f"Pays '{country_name}' non trouvé. Utilisation de '{closest[0]}'.")
        return countries_dict[closest[0]]
    st.error(f"❌ Pays '{country_name}' introuvable.")
    return None

def search_companies(query, country=None, address=None):
    """Recherche des entreprises via l'API OpenCorporates."""
    base_url = "https://api.opencorporates.com/v0.4/companies/search"
    params = {
        "q": query,
        "api_token": OPENCORPORATES_API_KEY,
        "order": "score",
    }
    if country:
        code = get_country_code(country)
        if code:
            params["country_code"] = code
    if address:
        params["registered_address"] = address
    try:
        resp = requests.get(base_url, params=params)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Erreur lors de la recherche (OpenCorporates) : {e}")
        return None

def companies_to_dataframe(search_results):
    """Transforme les résultats de recherche OpenCorporates en DataFrame."""
    if not search_results:
        return pd.DataFrame()
    results = search_results.get("results", {})
    companies = results.get("companies", [])
    data_list = []
    for c in companies:
        comp = c.get("company", {})
        # Extraction du LEI s'il existe dans "identifiers"
        lei = None
        if comp.get("identifiers"):
            for item in comp["identifiers"]:
                if item.get("identifier", {}).get("identifier_system_code") == "lei":
                    lei = item["identifier"]["uid"]
                    break
        if not lei:
            lei = comp.get("company_number", "")
        data_list.append({
            "Company Name": comp.get("name", ""),
            "Jurisdiction": comp.get("jurisdiction_code", "").upper(),
            "Company Number": comp.get("company_number", ""),
            "Current Status": comp.get("current_status", ""),
            "Incorporation Date": comp.get("incorporation_date", ""),
            "OpenCorporates URL": comp.get("opencorporates_url", ""),
            "LEI": lei
        })
    return pd.DataFrame(data_list)

def fetch_company_details_by_name(name, country=None):
    """Récupère les détails complets (OpenCorporates) via :jurisdiction_code/:company_number."""
    search_json = search_companies(name, country=country)
    if not search_json:
        return None
    results = search_json.get("results", {})
    companies = results.get("companies", [])
    if not companies:
        st.warning("Aucune entreprise trouvée pour ce nom.")
        return None
    first_company = companies[0].get("company", {})
    juri = first_company.get("jurisdiction_code", "")
    compnum = first_company.get("company_number", "")
    if not juri or not compnum:
        st.warning("Impossible de déterminer la juridiction ou le numéro d'enregistrement.")
        return None
    detail_url = f"https://api.opencorporates.com/v0.4/companies/{juri}/{compnum}"
    try:
        r = requests.get(detail_url, params={"api_token": OPENCORPORATES_API_KEY})
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Erreur lors de la récupération des détails (OpenCorporates) : {e}")
        return None


# ================================
# 3) FONCTIONS GLEIF
# ================================
def get_gleif_data(lei):
    """Récupère l'enregistrement complet GLEIF pour un LEI donné, ou None si erreur."""
    url = f"https://api.gleif.org/api/v1/lei-records/{lei}"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur Gleif (record): {e}")
        return None

def get_parent_entity(lei):
    """Récupère la maison mère via l'API GLEIF en filtrant sur filter[owns]."""
    base_url = "https://api.gleif.org/api/v1/lei-records"
    params = {"filter[owns]": lei}
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        if data['data']:
            return data['data'][0]
        else:
            st.info("Aucune maison mère GLEIF trouvée.")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur Gleif (parent): {e}")
        return None

def get_child_entities(lei):
    """Récupère les filiales directes via GLEIF (filter[ownedBy]=lei)."""
    base_url = "https://api.gleif.org/api/v1/lei-records"
    params = {"filter[ownedBy]": lei}
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur Gleif (child): {e}")
        return []

def get_all_gleif_info(lei):
    """Récupère : enregistrement GLEIF + maison mère + filiales directes."""
    record = get_gleif_data(lei)
    parent = get_parent_entity(lei)
    children = get_child_entities(lei)
    return {
        "record": record,
        "parent": parent,
        "children": children
    }


# ================================
# 4) CONSTRUCTION DU GRAPHE D'ANALYSE AML
# ================================
def create_analysis_graph(opco_data, gleif_data):
    """
    Construit un graphe PyVis avec contour coloré (fond transparent) et flèches colorées.
    """
    net = Network(height="700px", width="100%", directed=True)
    net.barnes_hut(gravity=-25000, 
                   central_gravity=0.3, 
                   spring_length=300, 
                   spring_strength=0.005, 
                   damping=0.09)

    # Utilitaires pour ajouter nœud/edge
    def add_transparent_node(net, node_id, label, border_color, shape="ellipse", title=""):
        net.add_node(
            node_id,
            label=label,
            shape=shape,
            title=title,
            borderWidth=2,
            color={
                "border": border_color,
                "background": "rgba(0,0,0,0)",  # Transparent
                "highlight": {
                    "border": border_color,
                    "background": "rgba(0,0,0,0)"
                }
            }
        )

    def add_colored_edge(net, source, target, color, title=""):
        net.add_edge(
            source,
            target,
            title=title,
            color=color
        )

    center = opco_data.get("company", {})
    center_id = f"OPCO_{center.get('company_number','')}"
    center_label = center.get("name", "Société Centrale")

    dissolution_date = center.get("dissolution_date")
    capital = center.get("capital")
    status_info = center.get("current_status","")
    tooltip_center = f"{center_label}<br>Status: {status_info}"
    if dissolution_date:
        tooltip_center += f"<br>Dissolution: {dissolution_date}"
    if capital:
        tooltip_center += f"<br>Capital: {capital}"

    # Nœud central
    add_transparent_node(net, center_id, center_label, "blue", "ellipse", tooltip_center)

    # NOMS PRECEDENTS
    for pn in center.get("previous_names", []) or []:
        old_name = pn.get("company_name","Unknown")
        start = pn.get("start_date","?")
        end = pn.get("end_date","?")
        old_id = f"OLDNAME_{old_name.replace(' ','_')}"
        old_label = old_name
        old_title = f"{old_name}<br>De {start} à {end}"
        add_transparent_node(net, old_id, old_label, border_color="lightgrey", shape="box", title=old_title)
        add_colored_edge(net, center_id, old_id, color="lightgrey", title="Previous Name")

    # ANCIENS NUMÉROS
    for pcn in center.get("previous_company_numbers", []) or []:
        pcn_id = f"OLDNUM_{pcn}"
        add_transparent_node(net, pcn_id, pcn, "lightgrey", "dot", "Ancien Numéro d'Entreprise")
        add_colored_edge(net, center_id, pcn_id, color="lightgrey", title="Previous Num")

    # OFFICERS, BENEFICIAL OWNERS, PEP
    personnes = {}
    def ajouter_personne(nom, role, extra=None):
        if not nom:
            return
        key = nom.strip().lower()
        if key not in personnes:
            personnes[key] = {"roles": set(), "info": {}}
        personnes[key]["roles"].add(role)
        if extra:
            personnes[key]["info"].update(extra)

    # Officers Actifs
    for item in center.get("officers", []):
        off = item.get("officer", {})
        nom = off.get("name")
        extra = {
            "position": off.get("position"),
            "address": off.get("address"),
            "nationality": off.get("nationality"),
            "dob": off.get("date_of_birth"),
            "oc_url": off.get("opencorporates_url"),
        }
        ajouter_personne(nom, "Officer (Active)", extra)

    # Anciens Officers
    for item in center.get("previous_officers", []):
        off = item.get("officer", {})
        nom = off.get("name")
        extra = {
            "position": off.get("position"),
            "inactive": True,
            "address": off.get("address"),
            "nationality": off.get("nationality"),
            "dob": off.get("date_of_birth"),
            "oc_url": off.get("opencorporates_url"),
        }
        ajouter_personne(nom, "Officer (Inactive)", extra)

    # Beneficial Owners
    for bo in opco_data.get("beneficial_owners", []) or []:
        nom = bo.get("name")
        if "percentage" in bo:
            pct = bo["percentage"]
            ajouter_personne(nom, f"Beneficial Owner ({pct}%)")
        else:
            ajouter_personne(nom, "Beneficial Owner")

    # PEP
    for pep in opco_data.get("politically_exposed_persons", []) or []:
        ajouter_personne(pep.get("name"), "PEP")

    # OFFICERS côté GLEIF
    gleif_record = gleif_data.get("record", {})
    if gleif_record and gleif_record.get("data") and gleif_record["data"].get("attributes"):
        attr_gleif = gleif_record["data"]["attributes"]
        if "officers" in attr_gleif:
            for off_g in attr_gleif["officers"] or []:
                ajouter_personne(off_g.get("name"), "Officer (GLEIF)")

    # Ajout au graphe
    for key_person, data_person in personnes.items():
        roles = data_person["roles"]
        info = data_person["info"]
        # Couleur du contour selon le type
        if any("PEP" in r for r in roles):
            border_col = "orange"
        elif any("Beneficial Owner" in r for r in roles):
            border_col = "yellow"
        else:
            border_col = "pink"

        label_name = key_person.title()
        node_id = f"PERSON_{key_person.replace(' ','_')}"
        tool = f"{label_name}<br>Rôles: {', '.join(roles)}"
        if info.get("position"):
            tool += f"<br>Position: {info['position']}"
        if info.get("inactive"):
            tool += "<br>(Inactif)"
        if info.get("address"):
            tool += f"<br>Adresse: {info['address']}"
        if info.get("nationality"):
            tool += f"<br>Nationalité: {info['nationality']}"
        if info.get("dob"):
            tool += f"<br>Date Naiss.: {info['dob']}"
        if info.get("oc_url"):
            tool += f"<br><a href='{info['oc_url']}' target='_blank'>OpenCorporates</a>"

        add_transparent_node(net, node_id, label_name, border_col, "box", tool)
        add_colored_edge(net, center_id, node_id, color=border_col, title="Personne Clé")

    # CODES D'INDUSTRIE
    raw_ind = center.get("industry_codes", [])
    industry_seen = set()
    for ic_struct in raw_ind:
        ic = ic_struct.get("industry_code", {})
        desc = (ic.get("description","Unknown") or "").strip()
        code_val = (ic.get("code","N/A") or "").strip()
        sig = (desc.lower(), code_val.lower())
        if sig not in industry_seen:
            industry_seen.add(sig)
            label_sector = f"{desc} ({code_val})"
            sec_id = f"SECTOR_{len(industry_seen)}_{code_val}"
            add_transparent_node(net, sec_id, label_sector, "purple", "dot", "Secteur d'Activité")
            add_colored_edge(net, center_id, sec_id, color="purple", title="Secteur")

    # PAYS / JURISDICTION
    juris_code = center.get("jurisdiction","")
    if juris_code:
        c_obj = pycountry.countries.get(alpha_2=juris_code.split("_")[0].upper())
        c_name = c_obj.name if c_obj else juris_code
        c_id = f"COUNTRY_{juris_code}"
        add_transparent_node(net, c_id, c_name, "grey", "triangle", "Pays / Juridiction Code")
        add_colored_edge(net, center_id, c_id, color="grey", title="Pays")

    # DONNÉES GLEIF
    if gleif_record and gleif_record.get("data") and gleif_record["data"].get("attributes"):
        attribs = gleif_record["data"]["attributes"]
        entity = attribs.get("entity", {})

        # Adresse légale
        la = entity.get("legalAddress", {})
        address_lines = la.get("addressLines", [])
        addr_str = " ".join(address_lines) if address_lines else ""
        city = la.get("city","")
        pays_leg = la.get("country","")
        address_title = f"{city} - {pays_leg}\n{addr_str}"
        gleif_addr_id = f"GLEIF_ADDR_{center_id}"
        add_transparent_node(net, gleif_addr_id, address_title, "darkgrey", "diamond",
                             "Adresse Légale (GLEIF)")
        add_colored_edge(net, center_id, gleif_addr_id, "darkgrey", "Juridiction (GLEIF)")

        # Forme Légale
        l_form = entity.get("legalForm", {}).get("id","")
        if l_form:
            lf_id = f"LEGALFORM_{l_form}"
            add_transparent_node(net, lf_id, l_form, "lightgrey", "dot", "Forme Juridique GLEIF")
            add_colored_edge(net, center_id, lf_id, "lightgrey", "Legal Form")

        # BIC
        bic_list = attribs.get("bic")
        if bic_list is None:
            bic_list = []
        for bic_code in bic_list:
            bic_id = f"BIC_{bic_code}"
            add_transparent_node(net, bic_id, bic_code, "lightblue", "dot", "Code BIC (Banque)")
            add_colored_edge(net, center_id, bic_id, "lightblue", "BIC")

        # Validation LEI
        reg_info = attribs.get("registration", {})
        val_level = reg_info.get("corroborationLevel", "Unknown")
        managing_lou = reg_info.get("managingLou","?")
        val_id = f"VALIDATION_{center_id}"
        val_label = f"Validation: {val_level}"
        add_transparent_node(net, val_id, val_label, "lightgreen", "dot", f"LOU: {managing_lou}")
        add_colored_edge(net, center_id, val_id, "lightgreen", "LEI Validation")

        # Expiration
        expiration_data = entity.get("expiration", {})
        exp_date = expiration_data.get("date","")
        if exp_date:
            exp_id = f"EXP_{center_id}"
            exp_label = f"Expiration LEI: {exp_date}"
            add_transparent_node(net, exp_id, exp_label, "lightgreen", "dot", "Date d'expiration du LEI")
            add_colored_edge(net, center_id, exp_id, "lightgreen", "LEI Expiration")

        # Field modifications
        relationships_data = gleif_record["data"].get("relationships", {})
        fm_links = relationships_data.get("field-modifications", {}).get("links", {})
        if "related" in fm_links:
            fm_url = fm_links["related"]
            fm_id = f"FM_{center_id}"
            add_transparent_node(net, fm_id, "Field Mods GLEIF", "lightgrey", "box",
                                 f"<a href='{fm_url}' target='_blank'>Historique GLEIF</a>")
            add_colored_edge(net, center_id, fm_id, "lightgrey", "Field Mods")

    # MAISON MÈRE
    parent_info = gleif_data.get("parent")
    if parent_info and "attributes" in parent_info:
        p_name = parent_info["attributes"]["entity"]["legalName"]["name"]
        p_lei = parent_info["id"]
        p_id = f"GLEIF_PARENT_{p_lei}"
        add_transparent_node(net, p_id, p_name, "red", "ellipse", f"Maison Mère (LEI: {p_lei})")
        add_colored_edge(net, center_id, p_id, "red", "Holding")

    # FILIALES
    for child in gleif_data.get("children", []) or []:
        c_lei = child["id"]
        c_name = child["attributes"]["entity"]["legalName"]["name"]
        c_id = f"CHILD_{c_lei}"
        add_transparent_node(net, c_id, c_name, "green", "ellipse", f"Filiale Directe (LEI: {c_lei})")
        add_colored_edge(net, center_id, c_id, "green", "Subsidiary")

    return net

def display_pyvis_network(pyvis_net):
    """Affiche le graphe PyVis dans Streamlit via un fichier temporaire."""
    from streamlit.components.v1 import html

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        tmp_path = tmp_file.name
        pyvis_net.write_html(tmp_path, open_browser=False)
    try:
        with open(tmp_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        html(html_content, height=700)
    except Exception as e:
        st.error(f"Erreur lors de l'affichage du graphe : {e}")
    finally:
        os.remove(tmp_path)


# ================================
# 5) INTERFACE STREAMLIT
# ================================
st.set_page_config(page_title="Analyse AML – Données Étendues", layout="wide")
st.title("Analyse AML : Récupération de Données & Construction du Graphe")

with st.expander("🔍 Rechercher une entreprise (OpenCorporates)"):
    query = st.text_input("Nom de l'entreprise :", "")
    col1, col2 = st.columns(2)
    with col1:
        sel_country = st.selectbox("Pays (optionnel) :", ALL_COUNTRIES, index=0)
        country = None if sel_country == "(Aucun)" else sel_country
    with col2:
        address = st.text_input("Adresse (optionnel) :", "")
    if st.button("Rechercher"):
        if not query:
            st.warning("Veuillez saisir un nom d'entreprise.")
        else:
            with st.spinner("Recherche en cours..."):
                results_json = search_companies(query, country, address)
                df_companies = companies_to_dataframe(results_json)
                if df_companies.empty:
                    st.info("Aucun résultat.")
                else:
                    st.success(f"{len(df_companies)} résultat(s) trouvé(s).")
                    st.session_state["df_companies"] = df_companies

if "df_companies" in st.session_state:
    st.subheader("Résultats de la recherche")
    df = st.session_state["df_companies"]
    st.dataframe(df, use_container_width=True)

    if not df.empty:
        lst = [f"{i+1} - {row['Company Name']}" for i, row in df.iterrows()]
        sel = st.selectbox("Sélectionnez une entreprise :", lst)
        idx = int(sel.split(" - ")[0]) - 1

        if st.button("Obtenir détails & Graph"):
            with st.spinner("Récupération des détails..."):
                detail_json = fetch_company_details_by_name(df.iloc[idx]["Company Name"], country)
                if detail_json:
                    # Données OpenCorporates (principal)
                    comp_details = detail_json["results"]["company"]
                    opco_data = {"company": comp_details, "raw": detail_json}

                    # Extraction LEI ou fallback
                    lei = None
                    if comp_details.get("identifiers"):
                        for item in comp_details["identifiers"]:
                            if item.get("identifier", {}).get("identifier_system_code") == "lei":
                                lei = item["identifier"]["uid"]
                                break
                    if not lei:
                        lei = comp_details.get("company_number")

                    gleif_info = {}
                    parent_other_children = []
                    if lei and len(lei) == 20:
                        # Récupérer GLEIF
                        gleif_info = get_all_gleif_info(lei)

                        # Si on a une maison mère, on récupère les autres filiales
                        parent_info = gleif_info.get("parent")
                        if parent_info and "id" in parent_info:
                            parent_lei = parent_info["id"]
                            # On va chercher toutes les filiales (children) de la maison mère
                            all_children_of_parent = get_child_entities(parent_lei)
                            # Filtrer pour exclure la société elle-même
                            parent_other_children = [
                                c for c in all_children_of_parent
                                if c["id"] != lei
                            ]

                    # =====================
                    # Affichage textuel des données utilisées
                    # =====================
                    st.markdown("## Données Utilisées pour le Graph")

                    center = opco_data["company"]
                    st.markdown("### Données OpenCorporates")
                    st.write(f"**Nom :** {center.get('name','')}")
                    st.write(f"**Numéro d'entreprise :** {center.get('company_number','')}")
                    st.write(f"**Statut :** {center.get('current_status','')}")
                    if center.get("capital"):
                        st.write(f"**Capital :** {center['capital']}")
                    if center.get("dissolution_date"):
                        st.write(f"**Date de dissolution :** {center['dissolution_date']}")

                    # Noms précédents
                    prev_names = center.get("previous_names", [])
                    if prev_names:
                        st.write("**Noms précédents :**")
                        for pn in prev_names:
                            st.write(f"- {pn.get('company_name','Inconnu')} (de {pn.get('start_date','?')} à {pn.get('end_date','?')})")

                    # Anciens numéros d'entreprise
                    prev_nums = center.get("previous_company_numbers", [])
                    if prev_nums:
                        st.write("**Numéros d'entreprise précédents :**")
                        for num in prev_nums:
                            st.write(f"- {num}")

                    # Officers
                    officers = center.get("officers", [])
                    if officers:
                        st.write("**Officers Actifs :**")
                        for off_item in officers:
                            off = off_item.get("officer", {})
                            st.write(f"- {off.get('name','(Inconnu)')} / Position : {off.get('position','')}")

                    # Anciens Officers
                    previous_officers = center.get("previous_officers", [])
                    if previous_officers:
                        st.write("**Officers Anciens :**")
                        for off_item in previous_officers:
                            off = off_item.get("officer", {})
                            st.write(f"- {off.get('name','(Inconnu)')} / Position : {off.get('position','')} (Inactif)")

                    # Beneficial owners
                    if "beneficial_owners" in opco_data:
                        bo_list = opco_data["beneficial_owners"] or []
                        if bo_list:
                            st.write("**Beneficial Owners :**")
                            for bo in bo_list:
                                pct = f" ({bo['percentage']}%)" if bo.get("percentage") else ""
                                st.write(f"- {bo.get('name','(Inconnu)')}{pct}")

                    # PEP
                    if "politically_exposed_persons" in opco_data:
                        pep_list = opco_data["politically_exposed_persons"] or []
                        if pep_list:
                            st.write("**PEP (Politically Exposed Persons) :**")
                            for pep in pep_list:
                                st.write(f"- {pep.get('name','(Inconnu)')}")

                    # Industry Codes
                    industry_codes = center.get("industry_codes", [])
                    if industry_codes:
                        st.write("**Industry Codes :**")
                        for ic_struct in industry_codes:
                            ic = ic_struct.get("industry_code", {})
                            desc = ic.get("description","Unknown")
                            code_val = ic.get("code","N/A")
                            st.write(f"- {desc} ({code_val})")

                    # =====================
                    # Données GLEIF
                    # =====================
                    st.markdown("### Données GLEIF")
                    if lei and len(lei) == 20:
                        st.write(f"**LEI détecté :** {lei}")
                        record = gleif_info.get("record")
                        if record and record.get("data") and record["data"].get("attributes"):
                            attr_gleif = record["data"]["attributes"]
                            entity = attr_gleif.get("entity", {})

                            # Adresse légale GLEIF
                            la = entity.get("legalAddress", {})
                            address_lines = la.get("addressLines", [])
                            city = la.get("city","")
                            country = la.get("country","")
                            address_str = " ".join(address_lines)
                            st.write("**Adresse Légale GLEIF :**")
                            st.write(f"{address_str}, {city}, {country}")

                            # Forme légale
                            legal_form = entity.get("legalForm", {}).get("id","")
                            if legal_form:
                                st.write(f"**Forme légale GLEIF :** {legal_form}")

                            # BIC
                            bic_list = attr_gleif.get("bic", [])
                            if bic_list:
                                st.write("**Codes BIC :**")
                                for bic_code in bic_list:
                                    st.write(f"- {bic_code}")

                            # Validation LEI
                            reg_info = attr_gleif.get("registration", {})
                            val_level = reg_info.get("corroborationLevel", "Unknown")
                            managing_lou = reg_info.get("managingLou", "?")
                            st.write(f"**Niveau de validation LEI :** {val_level} (LOU: {managing_lou})")

                            # Date d'expiration
                            expiration_data = entity.get("expiration", {})
                            exp_date = expiration_data.get("date","")
                            if exp_date:
                                st.write(f"**Date d'expiration du LEI :** {exp_date}")

                            # Officers GLEIF
                            if "officers" in attr_gleif:
                                gleif_officers = attr_gleif["officers"] or []
                                if gleif_officers:
                                    st.write("**Officers (GLEIF) :**")
                                    for off_g in gleif_officers:
                                        st.write(f"- {off_g.get('name','(Inconnu)')}")

                        # Filiales directes
                        children_data = gleif_info.get("children", [])
                        if children_data:
                            st.write("**Filiales (directes) de cette entité (via GLEIF) :**")
                            for child in children_data:
                                c_lei = child["id"]
                                c_name = child["attributes"]["entity"]["legalName"]["name"]
                                st.write(f"- {c_name} (LEI: {c_lei})")

                        # Autres filiales (sœurs) de la maison mère
                        parent_info = gleif_info.get("parent")
                        if parent_info and parent_other_children:
                            mother_name = parent_info["attributes"]["entity"]["legalName"]["name"]
                            st.write(f"**Autres filiales de la maison mère {mother_name} :**")
                            for sibling in parent_other_children:
                                s_lei = sibling["id"]
                                s_name = sibling["attributes"]["entity"]["legalName"]["name"]
                                st.write(f"- {s_name} (LEI: {s_lei})")

                    else:
                        st.warning("LEI inexistant ou invalide (20 caractères). Aucune donnée GLEIF.")

                    # =====================
                    # Construction + Affichage du graphe
                    # =====================
                    net = create_analysis_graph(opco_data, gleif_info)
                    st.subheader("Graphe Interactif AML")
                    display_pyvis_network(net)

                else:
                    st.warning("Impossible de récupérer les détails pour cette entreprise.")
