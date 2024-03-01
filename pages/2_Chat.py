import os

import streamlit as st
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import (
    load_index_from_storage, StorageContext, VectorStoreIndex
)
from llama_index.core.tools import QueryEngineTool
from llama_index.vector_stores.azureaisearch import MetadataIndexFieldType, AzureAISearchVectorStore, IndexManagement
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

from init import vector_folder

st.subheader('Chat with PDF')

# chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant",
                                      "content": "Welcome, you can ask me anything about the document"})

# chatbot agent
with st.spinner(text="..."):
    index_name = os.getenv('INDEX_NAME')

    if os.getenv('VECTOR_SERVICE') == 'LOCAL':
        storage_context = StorageContext.from_defaults(persist_dir=vector_folder)
        index = load_index_from_storage(storage_context)
        query_engine = index.as_query_engine()


    elif os.getenv('VECTOR_SERVICE') == 'AISEARCH':
        metadata_fields = {
            "author": "author",
            "theme": ("topic", MetadataIndexFieldType.STRING),
            "director": "director",
        }

        # Use search client to demonstration using existing index
        search_client = SearchClient(
            endpoint=os.getenv('AISEARCH_END_POINT'),
            index_name=index_name,
            credential=AzureKeyCredential(os.getenv('AISEARCH_API_KEY')),
        )

        vector_store = AzureAISearchVectorStore(
            search_or_index_client=search_client,
            filterable_metadata_field_keys=metadata_fields,
            index_management=IndexManagement.VALIDATE_INDEX,
            id_field_key="id",
            chunk_field_key="chunk",
            embedding_field_key="embedding",
            embedding_dimensionality=1536,
            metadata_string_field_key="metadata",
            doc_id_field_key="doc_id",
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents([],storage_context=storage_context)
        query_engine = index.as_query_engine()

    elif os.getenv('VECTOR_SERVICE') == 'PINECONE':
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        pinecone_index = pc.Index(index_name)

        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents([],storage_context=storage_context)
        query_engine = index.as_query_engine()

    tools = [QueryEngineTool.from_defaults(query_engine=query_engine)]
    agent = OpenAIAgent.from_tools(tools, verbose=True)


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask the document"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner(text="..."):
        is_agent_processing = True
        response = agent.chat(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
