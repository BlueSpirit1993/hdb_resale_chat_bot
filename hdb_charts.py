# this package consolidates all the charting functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from openai import OpenAI
import pandas as pd
import json
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment
openai.api_key = os.getenv("OPENAI_API_KEY")

df = pd.read_csv(
    "data_concat.csv", header=0, parse_dates=["month"],low_memory=False
)


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
    
df_initial_preproc(df)


client = OpenAI(api_key=openai.api_key)
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": "resale price trend across JURONG WEST and 5 ROOM"
        }
    ],
    functions=[
        {
            "name": "get_average_prices",
            "description": "Return the rows in a DataFrame about average HDB Prices for a town for a flat type from a certain year onwards",
            "parameters": {
                "type": "object",
                "properties": {
                    "town": {"type": "string", "description": "Town in Singapore"},
                    "flat_type": {"type": "string", "description": "Flat type"},
                    "resale_price": {"type": "number", "description": "Minimum resale price in SGD"},
                    "lease_commence_date": {"type": "number", "description": "Lease commence year (optional)"},
                    "region": {"type": "string", "description": "Region in Singapore"},
                },
                "required": ["town"],
            },
        },
        {
            "name": "plot_sqm_all_town",
            "description": "price per sqm across different town",
            "parameters": {
                "type": "object",
                "properties": {

                },
                "required": [],
            },
        },
        {
            "name": "plot_sqm_single_twn_room",
            "description": "price per sqm across single town and flat type",
            "parameters": {
                "type": "object",
                "properties": {
                    "town": {"type": "string", "description": "Town in Singapore"},
                    "flat_type": {"type": "string", "description": "Flat type"},
                },
                "required": ["flat_type","town"],
            },
        },
        {
            "name": "plot_resale_price_all",
            "description": "mean resale price across different town",
            "parameters": {
                "type": "object",
                "properties": {
                },
                "required": [],
            },
        },
        {
            "name": "plot_resale_price_single",
            "description": "mean resale price across single town and different flat type",
            "parameters": {
                "type": "object",
                "properties": {
                    "town": {"type": "string", "description": "Town in Singapore"},
                },
                "required": ["town"],
            },
        },
        {
            "name": "plot_pricePerMonth_all",
            "description": "mean resale price per month across all town",
            "parameters": {
                "type": "object",
                "properties": {
                },
                "required": [],
            },
        },
        {
            "name": "plot_pricePerMonth_single",
            "description": "mean resale price per month across single town and flat type",
            "parameters": {
                "type": "object",
                "properties": {
                    "town": {"type": "string", "description": "Town in Singapore"},
                    "flat_type": {"type": "string", "description": "Flat type"},
                },
                "required": ["flat_type","town"],
            },
        },
        {
            "name": "plot_priceTrend_all",
            "description": "resale price trend across all town",
            "parameters": {
                "type": "object",
                "properties": {
                },
                "required": [],
            },
        },
        {
            "name": "plot_priceTrend_single",
            "description": "resale price trend across single town and flat type",
            "parameters": {
                "type": "object",
                "properties": {
                    "town": {"type": "string", "description": "Town in Singapore"},
                    "flat_type": {"type": "string", "description": "Flat type"},
                },
                "required": ["flat_type","town"],
            },
        }
    ],
    function_call="auto",
)

def get_average_prices(town: str, flat_type: str, lease_commence_date: int):
    # Ensure column names match your dataset
    filtered_df = df[
        (df["town"].str.upper() == town.upper()) &
        (df["flat_type"].str.upper() == flat_type.upper()) &
        (df["lease_commence_date"] >= lease_commence_date)
    ]

    if filtered_df.empty:
        return f"No data found for {flat_type} in {town} from {lease_commence_date} onwards."

    avg_price = filtered_df["resale_price"].mean()

    return {
        "town": town.title(),
        "flat_type": flat_type.upper(),
        "average_resale_price": round(avg_price, 2)
    }




# price per sqm across different town
def plot_sqm_all_town():
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
def plot_sqm_single_twn_room( town, flat_type ):
    df_query = df.query("flat_type == @flat_type & town == @town")
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
        f"Price Per Square Meter across {town} and flat type: {flat_type}",
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
def plot_resale_price_all():
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
def plot_resale_price_single(town):
    df_query = df.query("town == @town")
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
        f"Mean Resale Price across {town} and different flat type", y=1.01, fontsize=16
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
def plot_pricePerMonth_all():
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
def plot_pricePerMonth_single(flat_type, town):
    df_query = df.query("flat_type == @flat_type & town == @town")
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
        f"Resale Price per month across {town} and flat type: {flat_type}",
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
def plot_priceTrend_all():
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
def plot_priceTrend_single( flat_type, town):
    df_query = df.query("flat_type == @flat_type & town == @town")
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
        f"Resale price trend across {town} and flat type: {flat_type}", y=1.01, fontsize=16
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


args = json.loads(completion.choices[0].message.function_call.arguments)
print(args)
message = completion.choices[0].message

if message.function_call is not None:
    fn_name = message.function_call.name
    args = json.loads(message.function_call.arguments)

    if fn_name == "get_average_prices":
        result = get_average_prices(**args)
    elif fn_name == "plot_sqm_all_town":
        result = plot_sqm_all_town(**args)
    elif fn_name == "plot_sqm_single_twn_room":
        result = plot_sqm_single_twn_room(**args)
    elif fn_name == "plot_resale_price_all":
        result = plot_resale_price_all(**args)
    elif fn_name == "plot_resale_price_single":
        result = plot_resale_price_single(**args)
    elif fn_name == "plot_pricePerMonth_all":
        result = plot_pricePerMonth_all(**args)
    elif fn_name == "plot_pricePerMonth_single":
        result = plot_pricePerMonth_single(**args)
    elif fn_name == "plot_priceTrend_all":
        result = plot_priceTrend_all(**args)
    elif fn_name == "plot_priceTrend_single":
        result = plot_priceTrend_single(**args)
                       
    print("Function called:", fn_name)
    print("Arguments:", args)
    print("Result:", result)
else:
    print("No function call returned. Model responded with text only.")

