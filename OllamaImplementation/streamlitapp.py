

import os

from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st 
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.tracers import LangChainTracer


load_dotenv()


# Access secrets using st.secrets
LANGCHAIN_API_KEY = st.secrets.get("LANGCHAIN_API_KEY")
LANGCHAIN_TRACKING_V2 = st.secrets.get("LANGCHAIN_TRACKING_V2")
LANGCHAIN_PROJECT = st.secrets.get("LANGCHAIN_PROJECT")

#Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACKING_V2"]=LANGCHAIN_TRACKING_V2 
os.environ["LANGCHAIN_PROJECT"]=LANGCHAIN_PROJECT

# Setting up Langchain Tracer for tracing
tracer = LangChainTracer()
callback_manager = CallbackManager([tracer])



# Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant.Please respond to the queries"),
        ("user","Question:{question}")
    ]

)

#streamlit framework
st.title("Langchain Demo with Gemma Model")

input_text=st.text_input("What question do you have in your mind?")

#Ollama Gemma2b model
llm=OllamaLLM(model='gemma:2b',callback_manager=callback_manager)
output_parser=StrOutputParser()
chain=prompt|llm|output_parser


if input_text:
    st.write(chain.invoke({"question":input_text}))














