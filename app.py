import streamlit as st
from src import data, charts

st.set_page_config(page_title="COVID FR – Story App", page_icon="📈", layout="wide")

st.title("COVID-19 in France – Data Visualization")
st.caption("Source: [Data.gouv.fr – COVID-19 Indicators (Santé publique France)](https://www.data.gouv.fr/datasets/synthese-des-indicateurs-de-suivi-de-lepidemie-covid-19/)")

@st.cache_data(show_spinner=True)
def load_default_data():
    path = "data/table-indicateurs-open-data-dep-2023-06-30-17h59.csv"
    with open(path, "rb") as f:
        csv_bytes = f.read()
    return data.load_and_prepare(csv_bytes)

df_ini, df, nat = load_default_data()
st.success(f"Dataset loaded automatically ({len(df):,} rows).".replace(",", " "))

opt = st.sidebar.selectbox(
    "Section",
    options=[
        "Data Preview",
        "Data processing",
        "National View",
        "Regional View",
        "Map view",
        "Conclusion"
    ],
    index=0
)

if opt == "Data processing":
    st.header("Data Processing"); st.divider()
    charts.data_processing(df_ini)

elif opt == "Data Preview":
    st.header("Data Preview"); st.divider()
    charts.data_preview(df)

elif opt == "National View":
    st.header("National View"); st.divider()
    charts.national_view(nat)

elif opt == "Regional View":
    st.header("Regional View"); st.divider()
    charts.regional_view(df)

elif opt == "Map view":
    st.header("Map view"); st.divider()
    charts.map_view(df)

elif opt == "Conclusion":
    st.header("Conclusion"); st.divider()
    charts.conclusion_page()
