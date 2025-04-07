# this package consolidates all the charting functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

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

# price per sqm across different town
def plot_sqm_all_town(df):
    df_initial_preproc(df)
    sns.set_style("whitegrid")
    sns.set_palette("bright")
    g = sns.catplot(
        data=df,
        x="price_per_sqm",
        y="town",
        kind="bar",
        height=7,
        aspect=1,
        errorbar=None,
    )
    g.fig.suptitle("Price Per Square Meter across different town", y=1.02, fontsize=15)
    g.set(xlabel="Town", ylabel="Price Per Square Meter")

    g.ax.set_xlabel(
        g.ax.get_xlabel(), fontsize=14
    )  # Access and set x-axis label font size

    g.ax.set_ylabel(
        g.ax.get_ylabel(), fontsize=14
    )  # Access and set y-axis label font size

    plt.xticks(rotation=0)
    plt.show()

# price per sqm across single town and flat type
def plot_sqm_single_twn_room(df, room, twn):
    df_initial_preproc(df)
    df_query = df.query("flat_type == @room & town == @twn")
    sns.set_style("whitegrid")
    sns.set_palette("bright")

    g = sns.catplot(
        data=df_query,
        x="year_of_sales",
        y="price_per_sqm",
        kind="bar",
        height=5,
        aspect=2.5,
        errorbar=None,
    )
    g.fig.suptitle(
        f"Price Per Square Meter across {twn} and flat type: {room}",
        y=1.01,
        fontsize=18,
    )
    g.set(xlabel="Year of sales", ylabel="Price Per Square Meter")
    g.ax.set_xlabel(
        g.ax.get_xlabel(), fontsize=16
    )  # Access and set x-axis label font size

    g.ax.set_ylabel(
        g.ax.get_ylabel(), fontsize=16
    )  # Access and set y-axis label font size
    plt.xticks(rotation=45)

    # Format the y-axis tick labels to include commas using a lambda function
    formatter = mtick.FuncFormatter(lambda x, pos: f"{int(x):,}")
    g.ax.yaxis.set_major_formatter(formatter)
    plt.show()

# mean resale price across different town
def plot_resale_price_all(df):
    df_initial_preproc(df)
    sns.set_style("whitegrid")
    sns.set_palette("bright")

    # Calculate the mean resale price for each town
    mean_prices = df.groupby("town")["resale_price"].mean().sort_values()

    # Create a new categorical order based on the sorted mean prices
    town_order = mean_prices.index.tolist()

    g = sns.catplot(
        data=df,
        x="town",
        y="resale_price",
        kind="bar",
        height=5,
        aspect=2.5,
        errorbar=None,
        order=town_order,
    )
    g.fig.suptitle("Mean Resale Price across different town", y=1.01, fontsize=16)
    g.set(xlabel="Town", ylabel="Resale Price")
    g.ax.set_xlabel(
        g.ax.get_xlabel(), fontsize=15
    )  # Access and set x-axis label font size

    g.ax.set_ylabel(
        g.ax.get_ylabel(), fontsize=15
    )  # Access and set y-axis label font size
    plt.xticks(rotation=90)

    # Format the y-axis tick labels to include commas using a lambda function
    formatter = mtick.FuncFormatter(lambda x, pos: f"{int(x):,}")
    g.ax.yaxis.set_major_formatter(formatter)

    plt.show()


# mean resale price across single town and different flat type
def plot_resale_price_single(twn):
    df_initial_preproc(df)
    df_query = df.query("town == @twn")
    plt.clf()
    sns.set_style("whitegrid")
    sns.set_palette("bright")
    hue_order = [
        "1 ROOM",
        "2 ROOM",
        "3 ROOM",
        "4 ROOM",
        "5 ROOM",
        "EXECUTIVE",
        "MULTI-GENERATION",
    ]
    g = sns.catplot(
        data=df_query,
        x="town",
        y="resale_price",
        kind="bar",
        height=5,
        aspect=2,
        errorbar=None,
        hue="flat_type",
        hue_order=hue_order,
        palette="bright",
    )
    g.fig.suptitle(
        f"Mean Resale Price across {twn} and different flat type", y=1.01, fontsize=16
    )
    g.set(xlabel="Town", ylabel="Resale Price")

    g.ax.set_xlabel(
        g.ax.get_xlabel(), fontsize=15
    )  # Access and set x-axis label font size

    g.ax.set_ylabel(
        g.ax.get_ylabel(), fontsize=15
    )  # Access and set y-axis label font size

    plt.xticks(rotation=0)

    # Format the y-axis tick labels to include commas using a lambda function
    formatter = mtick.FuncFormatter(lambda x, pos: f"{int(x):,}")
    g.ax.yaxis.set_major_formatter(formatter)

    plt.show()

# mean resale price per month across all town
def plot_pricePerMonth_all(df):
    df_initial_preproc(df)
    sns.set_style("ticks")
    sns.set_palette("bright")
    # hue_order = ["1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM","EXECUTIVE", "MULTI-GENERATION"]
    g = sns.catplot(
        data=df,
        x="month_of_sales",
        y="resale_price",
        kind="bar",
        height=5,
        aspect=2,
        errorbar=None,
    )
    g.fig.suptitle(
        "Mean resale Price per month across different town and flat type",
        y=1.01,
        fontsize=16,
    )
    g.set(xlabel="Month of Sales", ylabel="Resale Price")

    g.ax.set_xlabel(
        g.ax.get_xlabel(), fontsize=15
    )  # Access and set x-axis label font size

    g.ax.set_ylabel(
        g.ax.get_ylabel(), fontsize=15
    )  # Access and set y-axis label font size

    # Format the y-axis tick labels to include commas using a lambda function
    formatter = mtick.FuncFormatter(lambda x, pos: f"{int(x):,}")
    g.ax.yaxis.set_major_formatter(formatter)

    plt.show()


# mean resale price per month across single town
def plot_pricePerMonth_single(df, room, twn):
    df_initial_preproc(df)
    df_query = df.query("flat_type == @room & town == @twn")
    sns.set_style("ticks")
    sns.set_palette("bright")
    # hue_order = ["1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM","EXECUTIVE", "MULTI-GENERATION"]
    g = sns.catplot(
        data=df,
        x="month_of_sales",
        y="resale_price",
        kind="bar",
        height=5,
        aspect=1.5,
        errorbar=None,
    )
    g.fig.suptitle(
        f"Resale Price per month across {twn} and flat type: {room}",
        y=1.01,
        fontsize=15,
    )
    g.set(xlabel="Month of Sales", ylabel="Resale Price")
    g.ax.set_xlabel(
        g.ax.get_xlabel(), fontsize=14
    )  # Access and set x-axis label font size

    g.ax.set_ylabel(
        g.ax.get_ylabel(), fontsize=14
    )  # Access and set y-axis label font size

    # Format the y-axis tick labels to include commas using a lambda function
    formatter = mtick.FuncFormatter(lambda x, pos: f"{int(x):,}")
    g.ax.yaxis.set_major_formatter(formatter)

    plt.show()


# resale price trend across all town
def plot_priceTrend_all(df):
    df_initial_preproc(df)
    sns.set_style("ticks")
    sns.set_palette("bright")
    hue_order = [
        "1 ROOM",
        "2 ROOM",
        "3 ROOM",
        "4 ROOM",
        "5 ROOM",
        "EXECUTIVE",
        "MULTI-GENERATION",
    ]
    g = sns.relplot(
        data=df,
        x="month",
        y="resale_price",
        kind="line",
        height=5,
        aspect=2,
        palette="bright",
        errorbar=None,
        hue="flat_type",
        hue_order=hue_order,
    )
    g.fig.suptitle(f"Resale price trend across all town", y=1.01, fontsize=17)
    g.set(xlabel="Year", ylabel="Resale Price")

    g.ax.set_xlabel(
        g.ax.get_xlabel(), fontsize=16
    )  # Access and set x-axis label font size

    g.ax.set_ylabel(
        g.ax.get_ylabel(), fontsize=16
    )  # Access and set y-axis label font size

    plt.ticklabel_format(style="plain", axis="y")

    # Format the y-axis tick labels to include commas using a lambda function
    formatter = mtick.FuncFormatter(lambda x, pos: f"{int(x):,}")
    g.ax.yaxis.set_major_formatter(formatter)

    plt.show()


# resale price trend across single town and flat type
def plot_priceTrend_single(df, room, twn):
    df_initial_preproc(df)
    df_query = df.query("flat_type == @room & town == @twn")
    sns.set_style("ticks")
    sns.set_palette("bright")
    g = sns.relplot(
        data=df_query,
        x="month",
        y="resale_price",
        kind="line",
        height=5,
        aspect=2,
        errorbar=None,
    )
    g.fig.suptitle(
        f"Resale price trend across {twn} and flat type: {room}", y=1.01, fontsize=16
    )
    g.set(xlabel="Year", ylabel="Resale Price")
    g.ax.set_xlabel(
        g.ax.get_xlabel(), fontsize=15
    )  # Access and set x-axis label font size

    g.ax.set_ylabel(
        g.ax.get_ylabel(), fontsize=15
    )  # Access and set y-axis label font size

    # Format the y-axis tick labels to include commas using a lambda function
    formatter = mtick.FuncFormatter(lambda x, pos: f"{int(x):,}")

    g.ax.yaxis.set_major_formatter(formatter)

    plt.show()
