import logging
import os
import sys

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.openai import OpenAI

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

openai_api_key = None
azure_openai_api_key = None
azure_endpoint = None
azure_openai_api_version = None
llm = None
embed_model = None

load_dotenv()

data_folder = os.getenv('DATA_FOLDER')
vector_folder = os.getenv('VECTOR_FOLDER')

###################### AZURE OPENAI ####################

if os.getenv('IS_AZURE_OPENAI') == 'True':
    azure_openai_api_key = os.getenv('AZURE_OPENAI_API_KEY')
    azure_endpoint = os.getenv('AZURE_OPENAI_END_POINT')
    azure_openai_api_version = os.getenv('AZURE_OPENAI_API_VERSION') #https://learn.microsoft.com/en-us/azure/ai-services/openai/reference?WT.mc_id=AZ-MVP-5004796

    llm = AzureOpenAI(
        #model="gpt-35-turbo-16k",
        #deployment_name="rag16k",
        model="gpt-35-turbo",
        deployment_name="rag35",
        api_key=azure_openai_api_key,
        azure_endpoint=azure_endpoint,
        api_version=azure_openai_api_version,
    )

    # You need to deploy your own embedding model as well as your own chat completion model
    embed_model = AzureOpenAIEmbedding(
        model="text-embedding-ada-002",
        deployment_name="ragada",
        api_key=azure_openai_api_key,
        azure_endpoint=azure_endpoint,
        api_version=azure_openai_api_version,
    )

###################### OPENAI ####################

else:
    openai_api_key = os.getenv('OPENAI_API_KEY')

    llm = OpenAI(
        model="gpt-3.5-turbo",
        api_key=openai_api_key,
    )

    embed_model = OpenAIEmbedding(
        model="text-embedding-ada-002",
        api_key=openai_api_key,
    )

####################

Settings.llm = llm
Settings.embed_model = embed_model