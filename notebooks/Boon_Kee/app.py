import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def df_initial_preproc(df):
    df.month = pd.to_datetime(df.month)
    df["year_of_sales"] = df["month"].dt.year
    df["month_of_sales"] = df["month"].dt.month
    # Clean up the 'MULTI-GENERATION' entries
    df["flat_type"] = df["flat_type"].str.replace(
        "MULTI GENERATION", "MULTI-GENERATION"
    )
    # add price per sqm
    df["price_per_sqm"] = df.resale_price / df.floor_area_sqm


# resale price trend across single town and flat type
def plot_priceTrend_single(df, room, twn):
    df_initial_preproc(df)
    df_query = df.query("flat_type == @room & town == @twn")
    sns.set_style("ticks")
    sns.set_palette("bright")
    g8 = sns.relplot(
        data=df_query,
        x="month",
        y="resale_price",
        kind="line",
        height=5,
        aspect=2,
        errorbar=None,
    )
    g8.fig.suptitle(
        f"Resale price trend across {twn} and flat type: {room}", y=1.01, fontsize=16
    )
    g8.set(xlabel="Year", ylabel="Resale Price")
    g8.ax.set_xlabel(
        g8.ax.get_xlabel(), fontsize=15
    )  # Access and set x-axis label font size

    g8.ax.set_ylabel(
        g8.ax.get_ylabel(), fontsize=15
    )  # Access and set y-axis label font size

    # Format the y-axis tick labels to include commas using a lambda function
    formatter = mtick.FuncFormatter(lambda x, pos: f"{int(x):,}")
    g8.ax.yaxis.set_major_formatter(formatter)

    return g8

def plot_sqm_all_town(df):
    df_initial_preproc(df)
    sns.set_style("whitegrid")
    sns.set_palette("bright")
    g1 = sns.catplot(
        data=df,
        x="price_per_sqm",
        y="town",
        kind="bar",
        height=7,
        aspect=1,
        errorbar=None,
    )
    g1.fig.suptitle("Price Per Square Meter across different town", y=1.02, fontsize=15)
    g1.set(ylabel="Town", xlabel="Price Per Square Meter")

    g1.ax.set_xlabel(
        g1.ax.get_xlabel(), fontsize=14
    )  # Access and set x-axis label font size

    g1.ax.set_ylabel(
        g1.ax.get_ylabel(), fontsize=14
    )  # Access and set y-axis label font size

    plt.xticks(rotation=0)

    formatter = mtick.FuncFormatter(lambda x, pos: f"{int(x):,}")
    g1.ax.xaxis.set_major_formatter(formatter)
    return g1


st.title("HDB buying & selling companion")


st.header("Resale Price Trend Explorer")

df = pd.read_csv("../../data/intermediate/data_concat.csv", header=0)
df_initial_preproc(df)
# Get unique values for dropdowns
flat_types = sorted(df['flat_type'].unique())
towns = sorted(df['town'].unique())

# Create select boxes for user input
selected_room = st.selectbox("Select Flat Type:", flat_types)
selected_town = st.selectbox("Select Town:", towns)

# Call the plotting function and display the plot
if selected_room and selected_town:
    g8 = plot_priceTrend_single(df.copy(), selected_room, selected_town)
    st.pyplot(g8)


    g1 = plot_sqm_all_town(df.copy())
    st.pyplot(g1)
else:
    st.info("Please select a Flat Type and Town to view the trend.")
