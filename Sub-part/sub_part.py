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


numero_doc = 3

loader = TextLoader(f"doc\\{numero_doc}.txt", encoding = 'UTF-8')
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

#Utilizzata per specializzare l'ontologia
def specialize_ontology():
    time.sleep(5)
    model = Together(model = "mistralai/Mixtral-8x22B-Instruct-v0.1", max_tokens=1000, temperature= 0.6) 
    template = read_txt("ontology\expand_onto_prompt.txt")
    prompt = PromptTemplate.from_template(template)
    document_chain = create_stuff_documents_chain(model, prompt) #
    retrieval_chain = create_retrieval_chain(retriever, document_chain) #
    onto = read_txt("ontology\\base_ontology.txt")                  
    response = retrieval_chain.invoke({"input" : onto})
    with open(f"ontology\\final_onto.owl","w",encoding="utf-8") as f:
            f.write(response["answer"])

#Genera le istanze partendo dal docmuneto
def triple_from_ontology():
    model = Together(model = "mistralai/Mixtral-8x22B-Instruct-v0.1", max_tokens=2500, temperature= 0.6) 
    template = read_txt("KG_creation\KG_prompt.txt")
    prompt = PromptTemplate.from_template(template)
    document_chain = create_stuff_documents_chain(model, prompt) 
    retrieval_chain = create_retrieval_chain(retriever, document_chain) 
    tmp = read_txt("ontology\\final_onto.owl")                  
    response = retrieval_chain.invoke({"input" : tmp})
    with open(f"KG_creation\individuals_doc_{numero_doc}.txt",'w',encoding="utf-8") as f:
        f.write(response["answer"])

#Genera il singolo KG, va ripetuto utilizzando due doc per volta
def generate_single_KG():
    txt1="KG_creation\individuals_doc_1.txt"
    txt2="KG_creation\individuals_doc_2.txt"

    model = Together(model = "mistralai/Mixtral-8x22B-Instruct-v0.1", max_tokens=3000, temperature= 0.6) 
    time.sleep(5)
    template = read_txt("KG_creation\single_KG_prompt.txt")
    prompt = PromptTemplate.from_template(template)
    chain = prompt | model | StrOutputParser()
    response = chain.invoke({"txt1": txt1, "txt2": txt2})
    with open("KG_creation\\all_doc_in_one.ttl", 'w', encoding="utf-8") as f:
        f.write(response)

#Genera  le CQs
def generate_CQ():
    onto=read_txt("ontology\\final_onto.owl")
    model = Together(model = "mistralai/Mixtral-8x22B-Instruct-v0.1", max_tokens=1000, temperature= 0.6) 
    time.sleep(5)
    template = read_txt("CQ_answering\CQ_generated_prompt")
    prompt = PromptTemplate.from_template(template)
    chain = prompt | model | StrOutputParser()
    response = chain.invoke({"source": onto})
    with open("CQ_answering\CQ.txt", 'w', encoding="utf-8") as f:
        f.write(response)        

#Risponde alle CQs
def CQ_Answering():
    model = Together(model = "mistralai/Mixtral-8x22B-Instruct-v0.1", max_tokens=1500, temperature= 0.6) 
    template = read_txt("evaluation\eval_prompt.txt")
    kg = read_txt(f"KG_creation\individuals_doc_{numero_doc}.txt")
    cqs = read_txt("CQ_answering\CQ.txt")
    prompt = PromptTemplate.from_template(template)
    chain = prompt | model | StrOutputParser()
    response = chain.invoke({"kg": kg, "cqs": cqs})
    with open(f"evaluation\eval_{numero_doc}.txt",'w',encoding="utf-8") as f:
            f.write(response)

#Genera il singolo KG, va ripetuto utilizzando due doc per volta
def generate_single_KG():
    txt1="KG_creation\individuals_doc_1.txt"
    txt2="KG_creation\individuals_doc_2.txt"

    model = Together(model = "mistralai/Mixtral-8x22B-Instruct-v0.1", max_tokens=3000, temperature= 0.6) 
    time.sleep(5)
    template = read_txt("KG_creation\single_KG_prompt.txt")
    prompt = PromptTemplate.from_template(template)
    chain = prompt | model | StrOutputParser()
    response = chain.invoke({"txt1": txt1, "txt2": txt2})
    with open("KG_creation\\all_doc_in_one.ttl", 'w', encoding="utf-8") as f:
        f.write(response)


print(f"\nnumero doc : {numero_doc}")