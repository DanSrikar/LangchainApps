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


#Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

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











