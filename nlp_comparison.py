import spacy
import re
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from itertools import product

##TODO first python -m spacy download en_core_web_sm

def preprocess_text(text):
    # 1. Convertire il testo in minuscolo
    text = text.lower()

    # 2. Rimuovere la punteggiatura
    text = re.sub(r'[^\w\s]', '', text)

    # 3. Tokenizzare il testo
    tokens = word_tokenize(text)

    # 4. Rimuovere le stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # 5. Lemmatizzare i token
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # 6. Unire i token pre-processati in una stringa
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


def read_txt(txt_path):
    with open(txt_path,'r', encoding="utf-8") as f:
        content = f.read()
    return content


#function to extract POS from the document
def simple_srl():
    
    txt = read_txt("Full-text\data\CASE OF A AND B v. GEORGIA.txt")
    pre_processed = preprocess_text(txt)
    nlp = spacy.load('en_core_web_sm')  
    doc = nlp(pre_processed)
    subjects = []
    verbs = []
    objects = []
    
    for token in doc:
        if "subj" in token.dep_:
            subjects.append(token.text)
        if "VERB" in token.pos_:
            verbs.append(token.lemma_)
        if "obj" in token.dep_:
            objects.append(token.text)
            
    return {
        'subjects': subjects,
        'verbs': verbs,
        'objects': objects
    }


#final function to create triples
def create_triple():
    
    my_dict = simple_srl()

    triples = list(product(my_dict['subjects'], my_dict['verbs'], my_dict['objects']))
    return triples

triples_from_document = create_triple()
print(len(triples_from_document))