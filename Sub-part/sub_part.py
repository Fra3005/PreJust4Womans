from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_together import TogetherEmbeddings
from langchain_together import Together
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import os
import time


def read_txt(txt_path):
    with open(txt_path,'r',encoding="utf-8") as f:
        content = f.read()
    return content


api_key="" 
os.environ["TOGETHER_API_KEY"] = api_key

# Riferimento numero-nome_documento
# 1-CASE_OF_A._v._CROATIA
# 2-CASE_OF_AYDIN_v._TURKEY
# 3-CASE_OF_HAJDUOVA_v._SLOVAKIA
# 4-CASE_OF_A_AND_B_v._GEORGIA
# 5-CASE_OF_A.A._AND_OTHERS_v._SWEDEN



#Creation of the retriever for the specified document
numero_doc = 3

loader = TextLoader(f"Sub-part\data\\{numero_doc}.txt", encoding = 'UTF-8')
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(docs)

embeddings = TogetherEmbeddings(model = "togethercomputer/m2-bert-80M-8K-retrieval")
vector = None
for d in documents:
    if vector is None:
        vector = FAISS.from_documents([d], embeddings)
    else:
        vector.add_documents([d])
    time.sleep(0.3)
retriever = vector.as_retriever()
###

#Function to specialize base ontology taking the information from the document
def specialize_ontology():
    time.sleep(5)
    model = Together(model = "mistralai/Mixtral-8x22B-Instruct-v0.1", max_tokens=1000, temperature= 0.6) 
    template = read_txt("Prompt\expand_onto_prompt.txt")
    prompt = PromptTemplate.from_template(template)
    document_chain = create_stuff_documents_chain(model, prompt) #
    retrieval_chain = create_retrieval_chain(retriever, document_chain) #
    onto = read_txt("Commons\\base_ontology.txt")                  
    response = retrieval_chain.invoke({"input" : onto})
    with open(f"Commons\\final_onto.owl","w",encoding="utf-8") as f:
            f.write(response["answer"])

#function to generate kgs from the final ontology 
def triple_from_ontology():
    model = Together(model = "mistralai/Mixtral-8x22B-Instruct-v0.1", max_tokens=2500, temperature= 0.6) 
    template = read_txt("Prompt\Kg_creation_prompt.txt")
    prompt = PromptTemplate.from_template(template)
    document_chain = create_stuff_documents_chain(model, prompt) 
    retrieval_chain = create_retrieval_chain(retriever, document_chain) 
    tmp = read_txt("Commons\\final_onto.owl")                  
    response = retrieval_chain.invoke({"input" : tmp})
    with open(f"Sub-part\ontology_creation\Knowledge_graph\individuals_doc_{numero_doc}.txt",'w',encoding="utf-8") as f:
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
    with open("Commons\CQ1.txt", 'w', encoding="utf-8") as f:
        f.write(response)        

#function to make answer for each knwoledge graph
def CQ_Answering():
    model = Together(model = "mistralai/Mixtral-8x22B-Instruct-v0.1", max_tokens=1500, temperature= 0.6) 
    template = read_txt("Prompt\\answering_cqs_prompt.txt")
    kg = read_txt(f"Sub-part\ontology_creation\Knowledge_graph\individuals_doc_{numero_doc}.txt")
    cqs = read_txt("Commons\CQ.txt")
    prompt = PromptTemplate.from_template(template)
    chain = prompt | model | StrOutputParser()
    response = chain.invoke({"kg": kg, "cqs": cqs})
    with open(f"Sub-part\ontology_creation\Cqs_Answering\eval_{numero_doc}1.txt",'w',encoding="utf-8") as f:
            f.write(response)


print(f"\nnumero doc : {numero_doc}")


#PIPELINE TO GENERATE ALL THE FILES NEEDED

#specialize_ontology()
#triple_from_ontology()
#generate_CQ()
#CQ_Answering()
