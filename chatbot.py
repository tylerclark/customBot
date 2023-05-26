import os
import streamlit as st
import openai
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# Set up Azure Key Vault client with managed identity
credential = DefaultAzureCredential()
key_vault_url = "https://webapp-keys.vault.azure.net/"
secret_name = "openai-key"
secret_client = SecretClient(vault_url=key_vault_url, credential=credential)

# Retrieve the secret key
secret = secret_client.get_secret(secret_name)
secret_value = secret.value

openai.api_key = secret_value


def main():
    st.header("Chat with PDF 💬")

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # # embeddings
        store_name = pdf.name[:-4]
        st.success("Successfully uploaded: "f'{store_name}')
        # st.write(chunks)

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)

        if query:
            k = 3  # Number of nearest neighbors to retrieve
            distances = []  # List to store the distances
            labels = []
            docs = VectorStore.similarity_search(
                query=query, k=k, distances=distances, labels=labels)

            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.divider()
            st.subheader("Answer: ")
            st.write(response)
            st.divider()


if __name__ == '__main__':
    main()
