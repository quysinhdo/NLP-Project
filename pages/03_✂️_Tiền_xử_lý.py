import streamlit as st
import pandas as pd
import spacy
import nltk
import contractions
import difflib
from nltk.stem import PorterStemmer
from spellchecker import SpellChecker
from spacy import displacy
from utils.helpers import display_title_icon, process_uploaded_file

display_title_icon("✂️", "Tiền Xử Lý Văn Bản")

# Tải các tài nguyên cần thiết cho tiếng Anh
nltk.download('punkt')
nltk.download('wordnet')

# Tải mô hình ngôn ngữ tiếng Anh
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("Vui lòng cài đặt mô hình ngôn ngữ tiếng Anh trước:")
    st.code("python -m spacy download en_core_web_sm")
    st.stop()

# Khởi tạo các công cụ xử lý cho tiếng Anh
stemmer = PorterStemmer()
spell = SpellChecker(language='en')  # Bộ kiểm tra chính tả tiếng Anh

# =============================================
# PHẦN 1: LỚP XỬ LÝ TIỀN XỬ LÝ VĂN BẢN TIẾNG ANH
# =============================================

class EnglishTextPreprocessor:
    def __init__(self, text):
        self.text = text
        self.original_words_with_spaces = self._tokenize_with_spaces(text)
        self.doc = nlp(text)

    def _tokenize_with_spaces(self, text_input):
        """Tách văn bản thành từ và khoảng trắng xen kẽ."""
        import re        
        return re.findall(r"[\w'-]+|\s+", text_input)

    def _reconstruct_text(self, tokens_with_spaces):
        """Ghép lại các token và khoảng trắng."""
        return "".join(tokens_with_spaces)
    
    def sentence_tokenization(self):
        """Tách câu"""
        return [sent.text for sent in self.doc.sents]
    
    def word_tokenization(self):
        """Tách từ"""
        return [token.text for token in self.doc]
    
    def remove_stopwords(self):
        """Xóa stopwords và trả về câu hoàn chỉnh"""
        filtered_tokens = [token.text for token in self.doc if not token.is_stop and not token.is_punct]
        return " ".join(filtered_tokens)
    
    def to_lowercase(self):
        """Chuyển thành chữ thường"""
        return self.text.lower()
    
    def stemming(self):
        """Stemming (Porter)"""
        return [stemmer.stem(token.text) for token in self.doc]
    
    def lemmatization(self):
        """Lemmatization"""
        return [(token.text, token.lemma_) for token in self.doc]
    
    def pos_tagging(self):
        """POS Tagging"""
        return [(token.text, token.pos_) for token in self.doc]
    
    def expand_contractions(self):
        """Sửa từ viết tắt và highlight các từ được sửa."""
        original_text_reconstructed = self._reconstruct_text(self.original_words_with_spaces)
        fixed_text = contractions.fix(original_text_reconstructed)        
        fixed_words_with_spaces = self._tokenize_with_spaces(fixed_text)        
        s = difflib.SequenceMatcher(None, self.original_words_with_spaces, fixed_words_with_spaces)        
        highlighted_parts = []
        original_parts_for_comparison = []

        for tag, i1, i2, j1, j2 in s.get_opcodes():
            original_segment = self.original_words_with_spaces[i1:i2]
            fixed_segment = fixed_words_with_spaces[j1:j2]
            if tag == 'replace':    
                original_joined = "".join(original_segment).strip()
                fixed_joined = "".join(fixed_segment).strip()
                if original_joined != fixed_joined:
                    highlighted_parts.append(f"<mark>{self._reconstruct_text(fixed_segment)}</mark>")
                else:
                    highlighted_parts.append(self._reconstruct_text(fixed_segment))
                original_parts_for_comparison.append(self._reconstruct_text(original_segment))
            elif tag == 'delete':                
                original_parts_for_comparison.append(self._reconstruct_text(original_segment))
                pass
            elif tag == 'insert':
                highlighted_parts.append(f"<mark>{self._reconstruct_text(fixed_segment)}</mark>")                
            elif tag == 'equal':
                highlighted_parts.append(self._reconstruct_text(fixed_segment))
                original_parts_for_comparison.append(self._reconstruct_text(original_segment))        
        highlighted_html = "".join(highlighted_parts)        
        if original_text_reconstructed.lower() == fixed_text.lower() and original_text_reconstructed == fixed_text:
            is_truly_changed = any(tag != 'equal' for tag, _, _, _, _ in s.get_opcodes())
            if not is_truly_changed:
                 highlighted_html = fixed_text

        return highlighted_html, fixed_text
    
    def correct_spelling(self):
        """Sửa lỗi chính tả và highlight các từ được sửa."""        
        corrected_text_parts = []
        highlighted_html_parts = []        
        words_only = [token for token in self.original_words_with_spaces if token.strip() and not token.isspace()]
        misspelled = spell.unknown(words_only)
        current_word_index = 0
        for part in self.original_words_with_spaces:
            if part.strip() and not part.isspace():
                original_word = part                
                if original_word in misspelled:
                    corrected_word = spell.correction(original_word)
                    if corrected_word and corrected_word != original_word:
                        highlighted_html_parts.append(f"<mark>{corrected_word}</mark>")
                        corrected_text_parts.append(corrected_word)
                    else:
                        highlighted_html_parts.append(original_word)
                        corrected_text_parts.append(original_word)
                else:
                    highlighted_html_parts.append(original_word)
                    corrected_text_parts.append(original_word)
                current_word_index += 1
            else:
                highlighted_html_parts.append(part)
                corrected_text_parts.append(part)
                
        highlighted_html = "".join(highlighted_html_parts)
        corrected_text = "".join(corrected_text_parts)
        
        return highlighted_html, corrected_text
    
    def named_entity_recognition(self):
        """Nhận diện thực thể (NER)"""
        return [(ent.text, ent.label_) for ent in self.doc.ents]
    
    def visualize_entities(self):
        """Hiển thị NER trực quan"""
        html = displacy.render(self.doc, style="ent", page=True)
        return html

# =============================================
# PHẦN 2: GIAO DIỆN NGƯỜI DÙNG (TIẾNG VIỆT)
# =============================================

st.markdown("""
## Các chức năng tiền xử lý văn bản

Chọn phương pháp xử lý và nhập văn bản cần xử lý:
""")

# Phần nhập liệu
input_method = st.radio("Chọn cách nhập liệu:", 
                       ["Nhập trực tiếp", "Tải lên file văn bản"], 
                       horizontal=True)

text_input = ""
if input_method == "Nhập trực tiếp":
    text_input = st.text_area("Nhập văn bản cần xử lý:", 
                            "Natural language processing (NLP) is a subfield of artificial intelligence.")
else:
    uploaded_file = st.file_uploader("Tải lên file (.txt, .csv, .docx, .pdf, .json)", type=["txt", "csv", "docx", "pdf", "json"])
    if uploaded_file:
        texts = process_uploaded_file(uploaded_file)
        if texts:
            text_input = "\n".join(texts)
            st.text_area("Nội dung file:", text_input, height=150)

# Lựa chọn phương pháp xử lý
st.subheader("Lựa chọn phương pháp tiền xử lý")
preprocess_method = st.selectbox(
    "Chọn chức năng xử lý:",
    options=[
        "Tách câu",
        "Tách từ", 
        "Xóa stopwords",
        "Chuyển thành chữ thường",
        "Stemming (Porter)",
        "Lemmatization",
        "POS Tagging",
        "Sửa từ viết tắt",
        "Sửa lỗi chính tả",
        "Nhận diện thực thể (NER)"
    ],
    index=0
)

# Nút thực thi
if st.button("⚡ Thực Hiện Tiền Xử Lý") and text_input:
    processor = EnglishTextPreprocessor(text_input)
    result = None
    
    with st.spinner("Đang xử lý văn bản..."):
        if preprocess_method == "Tách câu":
            result = processor.sentence_tokenization()
        elif preprocess_method == "Tách từ":
            result = processor.word_tokenization()
        elif preprocess_method == "Xóa stopwords":
            result = processor.remove_stopwords()
        elif preprocess_method == "Chuyển thành chữ thường":
            result = processor.to_lowercase()
        elif preprocess_method == "Stemming (Porter)":
            result = processor.stemming()
        elif preprocess_method == "Lemmatization":
            result = processor.lemmatization()
        elif preprocess_method == "POS Tagging":
            result = processor.pos_tagging()
        elif preprocess_method == "Sửa từ viết tắt":
            result = processor.expand_contractions()
        elif preprocess_method == "Sửa lỗi chính tả":
            result = processor.correct_spelling()
        elif preprocess_method == "Nhận diện thực thể (NER)":
            result = processor.named_entity_recognition()
    
    # Hiển thị kết quả
    st.subheader("Kết quả tiền xử lý")
    
    if preprocess_method == "Nhận diện thực thể (NER)":
        st.markdown(processor.visualize_entities(), unsafe_allow_html=True)
        st.write("Danh sách thực thể nhận diện được:")
        df = pd.DataFrame(result, columns=["Thực thể", "Loại"])
        st.dataframe(df, use_container_width=True)
    elif preprocess_method in ["Sửa từ viết tắt", "Sửa lỗi chính tả"]:
        highlighted_text, plain_text = result
        st.write("Văn bản gốc:", processor.text)
        st.write("Văn bản đã sửa:")
        st.markdown(highlighted_text, unsafe_allow_html=True)
    elif isinstance(result, list):
        if all(isinstance(item, tuple) for item in result):
            df = pd.DataFrame(result, columns=["Từ gốc", "Kết quả"])
            st.dataframe(df, use_container_width=True)
        else:
            st.write(" ".join(result) if preprocess_method == "Tách từ" else result)
    else:
        st.write(result)
    # st.subheader("Kết quả tiền xử lý")
    
    # if preprocess_method == "Nhận diện thực thể (NER)":
    #     st.markdown(processor.visualize_entities(), unsafe_allow_html=True)
    #     st.write("Danh sách thực thể nhận diện được:")
    #     df = pd.DataFrame(result, columns=["Thực thể", "Loại"])
    #     st.dataframe(df, use_container_width=True)
    # elif isinstance(result, list):
    #     if all(isinstance(item, tuple) for item in result):
    #         df = pd.DataFrame(result, columns=["Từ gốc", "Kết quả"])
    #         st.dataframe(df, use_container_width=True)
    #     else:
    #         st.write(result)
    # else:
    #     st.write(result)
    
    # Hiển thị thông tin thống kê
    with st.expander("📊 Thống kê văn bản"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Số câu", len(processor.sentence_tokenization()))
        with col2:
            st.metric("Số từ", len(processor.word_tokenization()))
        with col3:
            st.metric("Số từ (không stopwords)", len(processor.remove_stopwords()))

elif not text_input:
    st.warning("Vui lòng nhập hoặc tải lên văn bản tiếng Anh trước khi thực thi")

# # Giải thích các phương pháp
# st.markdown("---")
# st.subheader("📚 Giải thích phương pháp")
# methods_info = {
#     "Tách câu": "Chia văn bản tiếng Anh thành các câu riêng biệt",
#     "Tách từ": "Chia câu tiếng Anh thành các từ/token riêng biệt",
#     "Xóa stopwords": "Loại bỏ các từ phổ biến ít mang ý nghĩa trong tiếng Anh",
#     "Chuyển thành chữ thường": "Chuẩn hóa văn bản về dạng chữ thường",
#     "Stemming (Porter)": "Rút gọn từ tiếng Anh về dạng gốc (stem)",
#     "Lemmatization": "Đưa từ tiếng Anh về dạng từ điển (lemma)",
#     "POS Tagging": "Gán nhãn từ loại cho từng từ tiếng Anh",
#     "Sửa từ viết tắt": "Khai triển các từ viết tắt tiếng Anh thành đầy đủ",
#     "Sửa lỗi chính tả": "Sửa các từ tiếng Anh viết sai chính tả",
#     "Nhận diện thực thể (NER)": "Nhận diện tên người, địa điểm, tổ chức... trong tiếng Anh"
# }

# selected_method_info = methods_info.get(preprocess_method, "")
# if selected_method_info:
#     st.info(f"**{preprocess_method}**: {selected_method_info}")