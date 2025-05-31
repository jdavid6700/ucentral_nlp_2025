import spacy
import re
from nltk.corpus import stopwords
import nltk
from tqdm import tqdm

try:
    STOPWORDS_ES = set(stopwords.words("spanish"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS_ES = set(stopwords.words("spanish"))

class TextPreprocessor:
    def __init__(self, lang='spanish'):
        self.nlp = spacy.load("es_core_news_lg")
        self.stopwords = set(stopwords.words(lang))

    def clean_text(self, text: str) -> str:
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#\w+", "", text)
        text = re.sub(r"[^A-Za-zÁÉÍÓÚÑáéíóúñ ]+", " ", text)
        return text.lower().strip()

    def lemmatize(self, text: str) -> str:
        doc = self.nlp(text)
        lemmas = []
        for token in doc:
            if token.text not in self.stopwords and not token.is_punct:
                lemmas.append(token.lemma_)
        return " ".join(lemmas)
    
    def transform(self, text: str) -> str:
        return self.lemmatize(self.clean_text(text))
