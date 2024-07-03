from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import  RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_together import TogetherEmbeddings
from langchain_community.llms import Together
from langchain_core.output_parsers import StrOutputParser
import os
import re
import time


api_key=""
os.environ["TOGETHER_API_KEY"] = api_key



documents_to_read = [
                    "Full-text\data\CASE OF A AND B v. GEORGIA.txt",
                    "Full-text\data\CASE OF A. v. CROATIA.txt",
                    "Full-text\data\CASE OF A.A. AND OTHERS v. SWEDEN.txt",
                    "Full-text\data\CASE OF AYDIN v. TURKEY.txt",
                    "Full-text\data\CASE OF HAJDUOVÁ v. SLOVAKIA.txt"
                     ]

kg_from_documents = [
                    "Full-text\ontology_creation\Knowledge_graph\CASE OF A AND B v. GEORGIA.txt",
                    "Full-text\ontology_creation\Knowledge_graph\CASE OF A. v. CROATIA.txt",
                    "Full-text\ontology_creation\Knowledge_graph\CASE OF A.A. AND OTHERS v. SWEDEN.txt",
                    "Full-text\ontology_creation\Knowledge_graph\CASE OF AYDIN v. TURKEY.txt",
                    "Full-text\ontology_creation\Knowledge_graph\CASE OF HAJDUOVÁ v. SLOVAKIA.txt"
]


def load_text(pdf):
    loader = TextLoader(pdf, encoding='UTF-8')
    docs = loader.load()
    return docs

#Take the document and split them recursively in little chunk. Then each chunk will be used to built the embedding vector for the RAG
def splitting_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)
    return documents

#creation of the retriever Rag of each document. 
def creation_embedding_vector(documents):
    embeddings = TogetherEmbeddings(model = "togethercomputer/m2-bert-80M-8K-retrieval")
    vector = None
    for d in documents:
        print(d)
        if vector is None:
            vector = FAISS.from_documents([d], embeddings)
        else:
            vector.add_documents([d])
        time.sleep(0.4)
    retriever = vector.as_retriever()

    return retriever


def read_txt(txt_path):
    with open(txt_path,'r', encoding='utf-8') as f:
        content = f.read()
    return content


def load_cqs(CQs_path):
    with open(CQs_path, encoding='utf-8') as f:
        lines = f.readlines()
    CQs = [l[:-1] for l in lines]
    return CQs

#final function to call in order to extract the retriever of a specific document.
def create_retriever(document):
    d=load_text(document)
    splitted= splitting_text(d)
    retriever = creation_embedding_vector(splitted)
    return retriever


#Function to specialize base ontology taking the information from the document
def specialize_ontology(document):
    time.sleep(5)
    model = Together(model = "mistralai/Mixtral-8x22B-Instruct-v0.1", max_tokens=1000, temperature= 0.6) 
    template = read_txt("Prompt\expand_onto_prompt.txt")
    prompt = PromptTemplate.from_template(template)
    retriever = create_retriever(document)
    document_chain = create_stuff_documents_chain(model, prompt) #
    retrieval_chain = create_retrieval_chain(retriever, document_chain) #
    onto = read_txt("Commons\\base_ontology.txt")                  
    response = retrieval_chain.invoke({"input" : onto})
    with open(f"Commons\\final_onto1.owl","w",encoding="utf-8") as f:
            f.write(response["answer"])

#function to generate cqs
def generate_CQ():
    onto=read_txt("Commons\\final_onto.owl")
    model = Together(model = "mistralai/Mixtral-8x22B-Instruct-v0.1", max_tokens=1000, temperature= 0.6) 
    time.sleep(5)
    template = read_txt("Prompt\Generate_cqs_prompt.txt")
    prompt = PromptTemplate.from_template(template)
    chain = prompt | model | StrOutputParser()
    response = chain.invoke({"source": onto})
    with open(f"Commons\CQ.txt", 'w', encoding="utf-8") as f:
        f.write(response)

#function to generate kgs from the final ontology          
def kg_from_ontology():
    time.sleep(5)
    model = Together(model = "mistralai/Mixtral-8x22B-Instruct-v0.1", max_tokens=3500, temperature= 0.6) 
    template = read_txt("Prompt\\Kg_creation_prompt.txt")
    prompt = PromptTemplate.from_template(template)
    
    for document in documents_to_read:
        answers=dict()
        pattern = r'CASE.*'
        corrispondenza = re.search(pattern, document)
        retriever = create_retriever(document)
        document_chain = create_stuff_documents_chain(model, prompt) 
        retrieval_chain = create_retrieval_chain(retriever, document_chain) 
        onto = read_txt("Commons\\final_onto.owl")                  
        response = retrieval_chain.invoke({"input" : onto})
        with open(f"Full-text\ontology_creation\Knowledge_graph\{corrispondenza.group()}",'w',encoding="utf-8") as f:
            f.write(response["answer"])

#function to make answer for each knwoledge graph
def make_answers():
    model = Together(model = "mistralai/Mixtral-8x22B-Instruct-v0.1", max_tokens=600, temperature= 0.6) 

    for kg in kg_from_documents:
        cqs = load_cqs("Commons\CQ.txt")
        pattern = r'CASE.*'
        corrispondenza = re.search(pattern, kg)
        kg = read_txt(kg)
        template = read_txt("Prompt\\answering_cqs_prompt.txt")
        
        prompt = PromptTemplate.from_template(template)
        chain = prompt | model | StrOutputParser()
        response = chain.invoke({"kg": kg, "cqs": cqs})
        with open(f"Full-text\ontology_creation\Cqs_Answering\{corrispondenza.group()}", 'w', encoding="utf-8") as f:
            f.write(response)



#PIPELINE TO GENERATE ALL THE FILES NEEDED

#specialize_ontology("Full-text\data\CASE OF A AND B v. GEORGIA.txt")
# kg_from_ontology()
# generate_CQ()
# make_answers()