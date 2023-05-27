import os
import streamlit as st
import openai
import pandas
import docx
import pickle
from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain

# from azure.identity import ClientSecretCredential, DefaultAzureCredential
# from azure.keyvault.secrets import SecretClient

# KEY_VAULT_NAME = "webappkeys"
# CLIENT_ID = os.environ.get("")
# TENANT_ID = os.environ["TENANT_ID"]
# CLIENT_SECRET = os.environ["CLIENT_SECRET"]

# KeyVault_URI = f"https://{KEY_VAULT_NAME}.vault.azure.net/"

# # _credential = DefaultAzureCredential()

# _credential = ClientSecretCredential(
#     tenant_id=TENANT_ID,
#     client_id=CLIENT_ID,
#     client_secret=CLIENT_SECRET
# )
# _sc = SecretClient(vault_url=KeyVault_URI, credential=_credential)
# OPENAI_API_KEY = _sc.get_secret("openai-api-key").value

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY
# st.write(OPENAI_API_KEY)


def main():
    st.header("Chat with your Documents 💬")

    # upload a PDF file
    file = st.file_uploader("Upload your PDF/CSV file.",
                            type=['pdf', '.csv', '.xlsx', '.xls', '.docx'])

    if file is not None:
        extension = file.name[len(file.name)-3:]
        text = ""
        if(extension == "pdf"):
            file_reader = PdfReader(file)
            for page in file_reader.pages:
                text += page.extract_text()
        elif(extension == "csv"):
            file_reader = pandas.read_csv(file)
            text = file_reader.to_string(index=False)
        elif(extension == "lsx" or extension == "xls"):
            file_reader = pandas.read_excel(file)
            text = file_reader.to_string(index=False)
        elif(extension == "ocx"):
            file_reader = docx.Document(file)
            list = [paragraph.text for paragraph in file_reader.paragraphs]
            text = ' '.join(list)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # # embeddings
        store_name = file.name[:-4]
        st.success("Successfully uploaded: "f'{store_name}')

        # if os.path.exists(f"{store_name}.pkl"):
        #     with open(f"{store_name}.pkl", "rb") as f:
        #         VectorStore = pickle.load(f)
        # else:
        #     embeddings = OpenAIEmbeddings()
        #     VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        #     with open(f"{store_name}.pkl", "wb") as f:
        #         pickle.dump(VectorStore, f)

        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")

        if query:
            k = 3  # Number of nearest neighbors to retrieve
            distances = []  # List to store the distances
            labels = []
            docs = VectorStore.similarity_search(
                query=query, k=k, distances=distances, labels=labels)

            llm = OpenAI(model_name="gpt-3.5-turbo")
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
