import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import io
import base64
from hdb_charts import (
    df_initial_preproc,
    plot_sqm_all_town,
    plot_sqm_all_town_2,
    plot_sqm_single_twn_room,
    plot_resale_price_all,
    plot_resale_price_single,
    plot_resale_price_all_2,
    plot_pricePerMonth_all,
    plot_pricePerMonth_single,
    plot_pricePerMonth_all_2,
    plot_priceTrend_all,
    plot_priceTrend_single,
    plot_priceTrend_allFlat,
    data_resale_price_single,
    data_sqm_single_twn_room,
    data_last_resale_price
)
import requests



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
import streamlit as st
from langchain.vectorstores import FAISS

openai.api_key = st.secrets["openai"]["api_key"]
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

import asyncio
import sys

if sys.platform.startswith("linux") and sys.version_info >= (3, 10):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


df = pd.read_csv(
    "data_concat.csv", header=0, parse_dates=["month"],low_memory=False
)

df_initial_preproc(df)

def plot_to_base64_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return encoded

st.set_page_config(page_title="HDB Companion", layout="wide")
st.title("🏡 HDB Buying & Selling Companion")
st.header("💬 Ask the Chatbot about HDB trends")

user_input = None
user_input = st.chat_input("Ask about resale prices, trends, towns, or flat types!")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display existing chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        elif msg["role"] == "HDB Compansion that analyses HDB trends":
            if "content" in msg:
                st.markdown(msg["content"])
            if "chart" in msg:
                st.pyplot(msg["chart"])
            if "analysis" in msg:
                st.markdown(f"**🧠 GPT Insight:** {msg['analysis']}")

# Process new input
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare and send message to GPT with function calling

    client = OpenAI(api_key=openai.api_key)
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history if m["role"] == "user"],
        functions=[
            {
            "name": "predict_resale_price",
            "description": "Predict HDB resale price using a trained ML model",
            "parameters": {
                "type": "object",
                "properties": {
                "block": {"type": "string"},
                "street_name": {"type": "string"},
                "town": {"type": "string"},
                "flat_type": {"type": "string"},
                "storey_range": {"type": "string"},
                "floor_area_sqm": {"type": "number"},
                "remaining_lease_year": {"type": "number"},
                "month_year": {"type": "string", "description": "Format YYYY-MM"}
                },
                "required": ["block", "street_name", "town", "flat_type", "storey_range", "floor_area_sqm", "remaining_lease_year", "month_year"]
            }
            },
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
            },
            {
                "name": "answer_pdf_question",
                "description": "Answer questions using the 'buy_sell_eligibility.pdf' document, where is boon kee from",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Question about HDB buying/selling eligibility ,where is boon kee from"}
                    },
                    "required": ["query"]
                }
            }

        ],
        function_call="auto",
    )

    def get_average_prices(town: str = None, flat_type: str = None, lease_commence_date: int = 2024):
        # Check for missing inputs
        if not town or not flat_type or lease_commence_date is None:
            msg = "⚠️ One or more required parameters are missing: town, flat type, or lease commence date."
            return msg

        # Filter DataFrame
        filtered_df = df[
            (df["town"].str.upper() == town.upper()) &
            (df["flat_type"].str.upper() == flat_type.upper()) &
            (df["lease_commence_date"] >= lease_commence_date)
        ]

        # Handle empty data
        if filtered_df.empty:
            msg = f"⚠️ No data found for {flat_type} in {town} from {lease_commence_date} onwards."
            return msg

        # Compute average price
        avg_price = filtered_df["resale_price"].mean()
        result = f"✅ Average resale price for {flat_type} in {town} (from {lease_commence_date} onwards): ${avg_price:,.0f}"


        return result



    
    @st.cache_resource
    def load_vector_db():
        loader = PyPDFLoader("buy_sell_eligibility.pdf")
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
        texts = text_splitter.split_documents(data)

        embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
        vectorstore = FAISS.from_documents(texts, embedding=embeddings)
        
        return vectorstore

    def answer_pdf_question(query: str):
        vector_db = load_vector_db()
        llm = ChatOpenAI(temperature=0, openai_api_key=openai.api_key)
        chain = load_qa_chain(llm, chain_type="map_reduce")
        docs = vector_db.similarity_search(query, k=2)
        return chain.run(input_documents=docs, question=query)

    
    def plot_to_base64_img(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return encoded

    def get_gpt_analysis_for_plot(g, prompt="Analyze this HDB resale price chart."):
        # 1. Show the chart
        st.pyplot(g.fig)

        # 2. Convert the chart to base64 image
        img_base64 = plot_to_base64_img(g)

        # 3. Send to GPT-4o for analysis
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}]}
            ]
        )

        # 4. Extract content
        analysis_text = response.choices[0].message.content

        # 5. Show and store analysis
        st.subheader("📈 GPT's Analysis")
        st.write(analysis_text)

        # 6. Save to chat history
        st.session_state.chat_history.append({
            "role": "HDB Compansion that analyses HDB trends",
            "chart": g.fig,
            "analysis": analysis_text
        })



    def predict_resale_price_via_api(args):
        base_url = "https://demodayprediction-28523621686.asia-southeast1.run.app/predict"
        response = requests.get(base_url, params=args)  # 👈 send as GET with query string
        if response.status_code == 200:
            return response.json()["HDB Price"]
        else:
            return f"❌ API Error: {response.status_code} - {response.text}"



    message = completion.choices[0].message

    if message.function_call is not None:
        fn_name = message.function_call.name
        args = json.loads(message.function_call.arguments)
        if "town" in args:
            args["town"] = args["town"].upper()

        if "flat_type" in args:
            args["flat_type"] = args["flat_type"].upper()

        if fn_name == "get_average_prices":
            result = get_average_prices(**args)
            st.write(result)
            st.session_state.chat_history.append({
            "role": "HDB Compansion that analyses HDB trends",
            "content": result
        })
        elif fn_name == "plot_sqm_all_town":
            result = get_gpt_analysis_for_plot(plot_sqm_all_town(df,**args))
        elif fn_name == "plot_sqm_single_twn_room":
            result = get_gpt_analysis_for_plot(plot_sqm_single_twn_room(df,**args))
        elif fn_name == "plot_resale_price_all":
            result = get_gpt_analysis_for_plot(plot_resale_price_all(df,**args))
        elif fn_name == "plot_resale_price_single":
            result = get_gpt_analysis_for_plot(plot_resale_price_single(df,**args))
        elif fn_name == "plot_pricePerMonth_all":
            result = get_gpt_analysis_for_plot(plot_pricePerMonth_all(df,**args))
        elif fn_name == "plot_pricePerMonth_single":
            result = get_gpt_analysis_for_plot(plot_pricePerMonth_single(df,**args))
        elif fn_name == "plot_priceTrend_all":
            result = get_gpt_analysis_for_plot(plot_priceTrend_all(df,**args))
        elif fn_name == "plot_priceTrend_single":
            result = get_gpt_analysis_for_plot(plot_priceTrend_single(df,**args))
        elif fn_name == "answer_pdf_question":
            result = answer_pdf_question(**args)
 #           st.subheader("📄 Answer from PDF")
            st.write(result)
            st.session_state.chat_history.append({
            "role": "HDB Compansion that analyses HDB trends",
            "content": result
        })

        elif fn_name == "predict_resale_price":
            prediction = predict_resale_price_via_api(args)
            st.subheader("🏠 Predicted Resale Price")
            st.write(f"💰 ${prediction:.2f}")

            st.session_state.chat_history.append({
                "role": "HDB Compansion that analyses HDB trends",
                "content": f"Based on the details, the predicted resale price is **${prediction:.2f}**"
            })

        
    else:
    # Send general message again without function schema
        general_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": user_input}]
        )
        result = general_response.choices[0].message.content
        st.subheader("🤖 Answer")
        st.write(result)
        st.session_state.chat_history.append({"role": "HDB Compansion that analyses HDB trends", "content": result})

    

