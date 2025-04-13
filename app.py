import pandas as pd
import asyncio
import os

import graphrag.api as api
from graphrag.config.load_config import load_config
from pathlib import Path

from openai import OpenAI

import streamlit as st

PROJECT_DIRECTORY = os.getcwd() + "/data"

graphrag_config = load_config(Path(PROJECT_DIRECTORY))
entities = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/entities.parquet")
communities = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/communities.parquet")
community_reports = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/community_reports.parquet")
text_units = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/text_units.parquet")
relationships = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/relationships.parquet")


async def search(query):
    response, context = await api.local_search(
        config=graphrag_config,
        entities=entities,
        communities=communities,
        community_reports=community_reports,
        text_units=text_units,
        relationships=relationships,
        covariates=None,
        community_level=2,
        response_type="Multiple Paragraphs",
        query=query,
    )
    return response

def get_search(query):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(search(query))
    loop.close()
    return result

def get_prompt(retriever_result, query):
    return f"""Using the information provided below, please answer the following question: {query}
    
    {retriever_result}"""

client = OpenAI(api_key=st.secrets["GRAPHRAG_API_KEY"])

st.title("Pokemon Team Builder")

GEN = st.selectbox("Generation", ["Gen 1 - RB", "Gen 2 - GS", "Gen 3 - RS", "Gen 4 - DP", "Gen 5 - BW", "Gen 6 - XY", "Gen 7 - SM", "Gen 8 - SS", "Gen 9 - SV"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

if prompt := st.chat_input("Can you help me build a team..."):

    prompt = f"{GEN}: Can you help me build a team {prompt}"

    with st.chat_message("user"):
        st.markdown(prompt)

    graph_rag_result = get_search(prompt)
    prompt = get_prompt(graph_rag_result, prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=True,
        )
        response = st.write_stream(stream)
