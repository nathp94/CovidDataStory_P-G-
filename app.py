import streamlit as st
from src import data, charts

st.set_page_config(page_title="COVID FR – Story App", page_icon="📈", layout="wide")

st.title("COVID-19 in France – Data Visualization")
uploaded = st.sidebar.file_uploader("CSV départemental (SPF)", type=["csv"])

if uploaded is None:
    st.info("Charge un CSV pour continuer.")
    st.stop()

df_ini, df, nat = data.load_and_prepare(uploaded.getvalue())

st.success(f"{len(df):,} lignes chargées.".replace(",", " "))

opt = st.sidebar.selectbox(
    "Section",
    options=["Data Preview","Data processing", "National View", "Regional View","Map view","Conclusion"],
    index=0
)

if opt== 'Data processing':
    st.header("Data Processing"); st.divider(); charts.data_processing(df_ini)
elif opt == "Data Preview":
    st.header("Data Preview"); st.divider(); charts.data_preview(df)
elif opt == "National View":
    st.header("National View"); st.divider(); charts.national_view(nat)
elif opt == "Regional View":
    st.header("Regional View"); st.divider(); charts.regional_view(df)
elif opt == "Map view":
    st.header("Map view"); st.divider(); charts.map_view(df)
elif opt == "Conclusion":
    st.header("Conclusion"); st.divider(); charts.conclusion_page()


