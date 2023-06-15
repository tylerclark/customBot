import os
import streamlit as st
import pandas
import docx
from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain

# 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
with st.sidebar:
    uploadedFiles = st.file_uploader("Upload your files.",
                                     type=['pdf', '.csv', '.xlsx', '.xls', '.docx'], accept_multiple_files=True)
    with st.expander("Don't have an OpenAI key?"):
        st.write("To get an OpenAI key do the following:")
        st.markdown("- Go to *openai.com* and Log in with your account.")
        st.markdown(
            "- You'll get three options to choose from, choose *API* section.")
        st.markdown(
            "- You'll be redirected to *OpenAI Platform*")
        st.markdown(
            "- Here on the top-right corner, tap on your profile and choose *Manage Account* ")
        st.markdown(
            "- In the *API Keys* section, you can create a new secret key, that'll be your API key")

        st.markdown(
            ">Note that, if your free usage limit has expired, you will need to buy OpenAI credits")


def main():
    st.header("Chat with your Documents 💬")

    openaikey = st.text_input("Your Open API key: ")
    os.environ["OPENAI_API_KEY"] = openaikey

    # upload a PDF file

    text = ""
    for file in uploadedFiles:
        extension = file.name[len(file.name)-3:]
        if(extension == "pdf"):
            file_reader = PdfReader(file)
            for page in file_reader.pages:
                text += page.extract_text()
        elif(extension == "csv"):
            file_reader = pandas.read_csv(file)
            text += "\n".join(
                file_reader.apply(lambda row: ', '.join(row.values.astype(str)), axis=1))
        elif(extension == "lsx" or extension == "xls"):
            file_reader = pandas.read_excel(file)
            text += "\n".join(
                file_reader.apply(lambda row: ', '.join(row.values.astype(str)), axis=1))
        elif(extension == "ocx"):
            file_reader = docx.Document(file)
            list = [paragraph.text for paragraph in file_reader.paragraphs]
            text += ' '.join(list)

    # st.write(text)
    if(len(text) != 0):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # # embeddings
        st.success("Successfully uploaded files")

        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        # Accept user questions/query
        query = st.text_input("Ask questions about your file:")

        if query:
            k = 10  # Number of nearest neighbors to retrieve
            distances = []  # List to store the distances
            labels = []
            docs = VectorStore.similarity_search(
                query=query, k=k, distances=distances, labels=labels)

            llm = OpenAI(temperature=0.07, model_name="gpt-3.5-turbo")
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
