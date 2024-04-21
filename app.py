import langchain
from PyPDF2 import PdfReader
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
import pickle
import os
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
with st.sidebar:
    st.title('PDF Chat app')
    st.markdown('''
    ## About
    Hi from team Pennywise India. This App is mainly focused on to tell you about the crucx of the pdf you upload 😊.
    
    ''')
    add_vertical_space(5)
    st.write("Made by team Pennywise India")

def qna(query,vectorstore):
    model=Ollama(model="llama2")
    template="""
    Answer the questions based on the context below. If you can't
    answer the question, reply "I don't know".

    context:{context}

    Question: {question}
    """
    prompt=PromptTemplate.from_template(template)
    parser = StrOutputParser()
    chain= prompt| model | parser
    retriever=vectorstore.as_retriever()
    chain=(
        {'context':itemgetter("question")| retriever,"question": itemgetter("question")}
        |prompt
        |model
        |parser
    )
    return chain.invoke({"question":query})
def history(dict,query,response):
    dict[query]=response
    return dict
    
def main():
    st.header("Budget Buddy")
    pdf=st.file_uploader("Upload your PDF file",type='pdf')
    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        st.write(pdf_reader)
        text=""
        for page in pdf_reader.pages:
            text+= page.extract_text()
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks=text_splitter.split_text(text=text)
        store_name=pdf.name[:-4]
        if os.path.exists(f"D:\chatbotpdf\store_vector\{store_name}.pkl"):
            with open(f"D:\chatbotpdf\store_vector\{store_name}.pkl","rb") as f:
                vectorstore=pickle.load(f)
            print('Embeddings loaded from the disk')
        else:
            print("Embeddings are creating this may take a while \n")
            embeddings=OllamaEmbeddings()
            vectorstore=FAISS.from_texts(chunks,embedding=embeddings)
            
            with open(f"D:\chatbotpdf\store_vector\{store_name}.pkl",'wb') as f:
                pickle.dump(vectorstore,f)
            print('Embeddings are created ')
        query=st.text_input("What is your question ")
        dic={}
        if(len(dic)!=0):
            st.header("Chat History")
            for i in dic:
                st.write("Query")
                st.write(i)
                st.write('Answer')
                st.write(dic[i])
        if query:
            st.write("Please have patient... giving response soon")
            response=qna(query,vectorstore)
            st.write(response)
            history(dic,query,response)
        print(dic)
if __name__== '__main__':
    main()