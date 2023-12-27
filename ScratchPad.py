import pandas as pd
import numpy as np
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
import json
import ollama.client as client



splitter = RecursiveCharacterTextSplitter(
    chunk_size = 800,
    chunk_overlap  = 100,
    length_function = len,
    is_separator_regex = False,
)

# !pip install transformers
# !pip install tensorflow
import tensorflow as tf

print("TensorFlow version:", tf.__version__)

from transformers import pipeline

## Roberta based NER
# ner = pipeline("token-classification", model="2rtl3/mn-xlm-roberta-base-named-entity", aggregation_strategy="simple")
ner = pipeline("token-classification", model="dslim/bert-large-NER", aggregation_strategy="simple") # 這行會去抓 1.33G 的東西下來


print("Number of parameters ->", ner.model.num_parameters()/1000000, "Mn")
