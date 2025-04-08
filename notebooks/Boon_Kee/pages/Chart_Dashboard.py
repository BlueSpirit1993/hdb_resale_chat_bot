
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from hdb_charts import df_initial_preproc
from hdb_charts import plot_sqm_all_town
from hdb_charts import plot_sqm_single_twn_room
from hdb_charts import plot_resale_price_all
from hdb_charts import plot_resale_price_single
from hdb_charts import plot_pricePerMonth_all
from hdb_charts import plot_pricePerMonth_single
from hdb_charts import plot_priceTrend_all
from hdb_charts import plot_priceTrend_single


st.title("HDB buying & selling companion")

st.header("Chart Dashboard")

df = pd.read_csv("../../data/intermediate/data_concat.csv", header=0)
df_initial_preproc(df)
# Get unique values for dropdowns
flat_types = sorted(df['flat_type'].unique())
towns = sorted(df['town'].unique())

# Create select boxes for user input
st.markdown("\n\n\n")
selected_room = st.selectbox("Select Flat Type:", options=flat_types, index=3)
selected_town = st.selectbox("Select Town:", options=towns, index=0)
st.markdown("\n\n\n")

# Call the plotting function and display the plot
if selected_room and selected_town:
    chart1 = plot_priceTrend_single(df, selected_room, selected_town)
    st.markdown("\n\n\n")
    st.pyplot(chart1)

    chart2 = plot_priceTrend_all(df)
    st.markdown("\n\n\n")
    st.pyplot(chart2)

    chart3 = plot_resale_price_single(df, selected_town)
    st.markdown("\n\n\n")
    st.pyplot(chart3)

    chart4 = plot_resale_price_all(df)
    st.markdown("\n\n\n")
    st.pyplot(chart4)

    chart5 = plot_sqm_single_twn_room(df, selected_room, selected_town)
    st.markdown("\n\n\n")
    st.pyplot(chart5)

    chart6 = plot_sqm_all_town(df)
    st.markdown("\n\n\n")
    st.pyplot(chart6)

    chart7 = plot_pricePerMonth_single(df, selected_room, selected_town)
    st.markdown("\n\n\n")
    st.pyplot(chart7)

    chart8 = plot_pricePerMonth_all(df)
    st.markdown("\n\n\n")
    st.pyplot(chart8)



else:
    st.info("Please select a Flat Type and Town to view the trend.")
