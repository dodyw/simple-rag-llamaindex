import os
import time

import streamlit as st
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader, StorageContext,
)
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore, IndexManagement, MetadataIndexFieldType
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from init import data_folder, vector_folder

st.header('ChatPdf')
st.subheader('Data')

source_doc = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")

if st.button("Ingest"):
    if not source_doc:
        st.warning(f"Please select the pdf document")
    else:

        # Delete files in data_folder
        for filename in os.listdir(data_folder):
            file_path = os.path.join(data_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Delete files in vector_folder (similar approach)
        for filename in os.listdir(vector_folder):
            file_path = os.path.join(vector_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        try:
            with st.spinner(text="Uploading ..."):
                with open(data_folder + "/" + source_doc.name, "wb") as f:
                    f.write(source_doc.read())

            with st.spinner(text="Processing ..."):
                documents = SimpleDirectoryReader(data_folder).load_data()
                index_name = os.getenv('INDEX_NAME')

                if os.getenv('VECTOR_SERVICE') == 'LOCAL':
                    index = VectorStoreIndex.from_documents(documents)
                    index.storage_context.persist(persist_dir=vector_folder)

                elif os.getenv('VECTOR_SERVICE') == 'AISEARCH':
                    # https://docs.llamaindex.ai/en/stable/examples/vector_stores/AzureAISearchIndexDemo.html

                    # Use index client to demonstrate creating an index
                    index_client = SearchIndexClient(
                        endpoint=os.getenv('AISEARCH_END_POINT'),
                        credential=AzureKeyCredential(os.getenv('AISEARCH_API_KEY')),
                    )

                    # Use search client to demonstration using existing index
                    search_client = SearchClient(
                        endpoint=os.getenv('AISEARCH_END_POINT'),
                        index_name=index_name,
                        credential=AzureKeyCredential(os.getenv('AISEARCH_API_KEY')),
                    )

                    metadata_fields = {
                        "author": "author",
                        "theme": ("topic", MetadataIndexFieldType.STRING),
                        "director": "director",
                    }

                    vector_store = AzureAISearchVectorStore(
                        search_or_index_client=index_client,
                        filterable_metadata_field_keys=metadata_fields,
                        index_name=index_name,
                        index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
                        id_field_key="id",
                        chunk_field_key="chunk",
                        embedding_field_key="embedding",
                        embedding_dimensionality=1536,
                        metadata_string_field_key="metadata",
                        doc_id_field_key="doc_id",
                        language_analyzer="en.lucene",
                        vector_algorithm_type="exhaustiveKnn",
                    )

                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

                elif os.getenv('VECTOR_SERVICE') == 'PINECONE':
                    # https://docs.llamaindex.ai/en/stable/examples/vector_stores/PineconeIndexDemo.html

                    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
                    indexes = pc.list_indexes().names()
                    pinecone_index = None

                    if index_name in indexes:
                        pinecone_index = pc.Index(index_name)
                        pinecone_index.delete(delete_all=True)

                    else:
                        st.warning(f"index not found: {index_name}")
                        index = pc.create_index(
                            name=index_name,
                            dimension=1536,
                            metric="euclidean",
                            spec=ServerlessSpec(cloud="aws", region="us-west-2"),
                        )
                        pinecone_index = pc.Index(index_name)

                    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

                st.success(f"File processed")

        except Exception as e:
            st.error(f"An error occurred: {e}")
