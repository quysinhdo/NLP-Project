import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, FastText
from gensim import downloader as api
from scipy.sparse import csr_matrix
from transformers import pipeline, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, GPT2Tokenizer, GPT2Model
import torch
from utils.helpers import display_title_icon, process_uploaded_file

display_title_icon("🧮", "Biểu Diễn Dữ Liệu")

class TextVectorizer:
    def __init__(self):
        self.models = {
            'bow': None,
            'ngram': None,
            'tfidf': None,
            'w2v': None,
            'fasttext': None,
            'bert': None,
            'roberta': None,
            'gpt': None
        }

    def vectorize(self, docs, method='bow'):
        if not docs or len(docs) == 0:
            return "Vui lòng nhập dữ liệu hợp lệ"
        
        docs = [d for d in docs if d.strip()]
        
        try:
            if method == 'bow':
                return self._bow_vectorize(docs)
            elif method == 'onehot':
                return self._onehot_encode(docs)
            elif method == 'ngram':
                return self._ngram_vectorize(docs)
            elif method == 'tfidf':
                return self._tfidf_vectorize(docs)
            elif method == 'w2v':
                return self._word2vec_embed(docs)
            elif method == 'glove':
                return self._glove_embed(docs)
            elif method == 'fasttext':
                return self._fasttext_embed(docs)
            elif method == 'bert':
                return self._bert_embed(docs)
            elif method == 'roberta':
                return self._roberta_embed(docs)
            elif method == 'gpt':
                return self._gpt_embed(docs)
            else:
                return "Phương pháp không được hỗ trợ"
        except Exception as e:
            return f"Lỗi: {str(e)}"

    def _bow_vectorize(self, docs):
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(docs)
        return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    def _onehot_encode(self, docs):
        vectorizer = CountVectorizer(binary=True)
        X = vectorizer.fit_transform(docs)
        return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    
    def _ngram_vectorize(self, docs):
        """Bag of N-grams (n=1,2)"""
        vectorizer = CountVectorizer(ngram_range=(1, 2))  # Kết hợp unigram và bigram
        X = vectorizer.fit_transform(docs)
        return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    def _tfidf_vectorize(self, docs):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(docs)
        return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    def _word2vec_embed(self, docs):
        tokenized = [d.split() for d in docs]
        model = Word2Vec(tokenized, vector_size=100, window=5, min_count=1, workers=4)
        vectors = []
        for doc in tokenized:
            vectors.append(np.mean([model.wv[word] for word in doc if word in model.wv] or [np.zeros(100)], axis=0))
        return pd.DataFrame(vectors, columns=[f'dim_{i}' for i in range(100)])

    def _glove_embed(self, docs):
        glove = api.load("glove-wiki-gigaword-100")
        vectors = []
        for doc in docs:
            words = doc.split()
            vectors.append(np.mean([glove[word] for word in words if word in glove] or [np.zeros(100)], axis=0))
        return pd.DataFrame(vectors, columns=[f'dim_{i}' for i in range(100)])

    def _fasttext_embed(self, docs):
        tokenized = [d.split() for d in docs]
        model = FastText(tokenized, vector_size=100, window=5, min_count=1, workers=4)
        vectors = []
        for doc in tokenized:
            vectors.append(np.mean([model.wv[word] for word in doc if word in model.wv] or [np.zeros(100)], axis=0))
        return pd.DataFrame(vectors, columns=[f'dim_{i}' for i in range(100)])

    def _bert_embed(self, docs):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        
        vectors = []
        for doc in docs:
            inputs = tokenizer(doc, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            # Lấy embedding của [CLS] token (tầng cuối cùng)
            cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
            vectors.append(cls_embedding[0])  # Lấy vector đầu tiên
        
        return pd.DataFrame(vectors, columns=[f'dim_{i}' for i in range(768)])

    def _roberta_embed(self, docs):
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base')
        
        vectors = []
        for doc in docs:
            inputs = tokenizer(doc, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
            vectors.append(cls_embedding[0])
        
        return pd.DataFrame(vectors, columns=[f'dim_{i}' for i in range(768)])

    def _gpt_embed(self, docs):
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2Model.from_pretrained('gpt2')
        
        vectors = []
        for doc in docs:
            inputs = tokenizer(doc, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            # GPT-2 không có [CLS] token nên lấy trung bình các token
            mean_embedding = torch.mean(outputs.last_hidden_state, dim=1).numpy()
            vectors.append(mean_embedding[0])
        
        return pd.DataFrame(vectors, columns=[f'dim_{i}' for i in range(768)])


def get_input_data():
    input_type = st.radio("Chọn cách nhập liệu:", ["Nhập trực tiếp", "Tải file"], horizontal=True)
    
    if input_type == "Nhập trực tiếp":
        text = st.text_area("Nhập văn bản (mỗi dòng 1 tài liệu):", height=150)
        return [line.strip() for line in text.split('\n') if line.strip()]
    else:
        file = st.file_uploader("Chọn file (.txt, .csv, .docx, .pdf, .json)", type=["txt", "csv", "docx", "pdf", "json"])
        if file:
            return process_uploaded_file(file)
    return []

def main():
    st.write("""
    Công cụ này giúp chuyển đổi dữ liệu sang các dạng biểu diễn số khác nhau.
    """)

    docs = get_input_data()
    
    method = st.selectbox(
        "Chọn phương pháp biểu diễn:",
        [
            "Bag-of-Words",
            "One-Hot Encoding",
            "Bag of N-grams", 
            "TF-IDF",
            "Word2Vec",
            "GloVe",
            "FastText",
            "Bert",
            "Roberta",
            "GPT-2"
        ]
    )

    method_map = {
        "Bag-of-Words": "bow",
        "One-Hot Encoding": "onehot",
        "Bag of N-grams": "ngram",
        "TF-IDF": "tfidf",
        "Word2Vec": "w2v",
        "GloVe": "glove",
        "FastText": "fasttext",
        "Bert": "bert",
        "Robert": "roberta",
        "GPT-2": "gpt"
    }

    if st.button("Thực hiện biểu diễn"):
        if not docs:
            st.warning("Vui lòng nhập dữ liệu trước")
            return
            
        with st.spinner("Đang xử lý..."):
            vectorizer = TextVectorizer()
            result = vectorizer.vectorize(docs, method_map[method])
            
            if isinstance(result, pd.DataFrame):
                st.success("Hoàn thành!")
                st.dataframe(result)
                
                csv = result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Tải kết quả",
                    data=csv,
                    file_name="ket_qua.csv",
                    mime="text/csv"
                )
            else:
                st.error(result)

if __name__ == "__main__":
    main()