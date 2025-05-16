import streamlit as st
from utils.helpers import display_title_icon, process_uploaded_file
import nlpaug.augmenter.word as naw
from transformers import MarianMTModel, MarianTokenizer
import pandas as pd

display_title_icon("🛠️", "Tăng cường Dữ liệu Văn bản")

# Lớp xử lý tăng cường dữ liệu
class DataAugmentation:
    def __init__(self):
        """Khởi tạo các bộ tăng cường dữ liệu NLP"""
        self.synonym_aug = naw.SynonymAug(aug_src="wordnet")  # Thay từ đồng nghĩa
        self.swap_aug = naw.RandomWordAug(action="swap")  # Đảo vị trí từ
        self.delete_aug = naw.RandomWordAug(action="delete")  # Xóa từ
        self.insert_aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert", aug_p=0.3)  # Thêm từ
        
        # Khởi tạo mô hình dịch máy
        self.translator_en_de = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        self.tokenizer_en_de = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        self.translator_de_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-de-en")
        self.tokenizer_de_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
        
    def synonym_augmentation(self, text):
        """Thay thế từ đồng nghĩa"""
        return self.synonym_aug.augment(text)[0]

    def swap_words(self, text):
        """Hoán đổi vị trí từ"""
        return self.swap_aug.augment(text)[0]

    def delete_words(self, text):
        """Xóa từ ngẫu nhiên"""
        return self.delete_aug.augment(text)[0]

    def insert_words(self, text):
        """Thêm từ ngẫu nhiên vào văn bản"""
        return self.insert_aug.augment(text)[0]

    def back_translation(self, text):
        """Dịch ngược bằng mô hình Helsinki-NLP"""
        if isinstance(text, str):  # Đảm bảo text là danh sách
            text = [text]

        tokens = self.tokenizer_en_de(text, return_tensors="pt", padding=True, truncation=True)
        translated = self.translator_en_de.generate(**tokens)
        translated_text = self.tokenizer_en_de.batch_decode(translated, skip_special_tokens=True)

        tokens = self.tokenizer_de_en(translated_text, return_tensors="pt", padding=True, truncation=True)
        back_translated = self.translator_de_en.generate(**tokens)
        back_translated_text = self.tokenizer_de_en.batch_decode(back_translated, skip_special_tokens=True)

        return back_translated_text[0] if len(back_translated_text) > 0 else text[0]

# Khởi tạo augmenter
if 'augmenter' not in st.session_state:
    st.session_state.augmenter = DataAugmentation()

# Giao diện Streamlit
st.markdown("""
## Các kỹ thuật tăng cường dữ liệu văn bản

Chọn các phương pháp bạn muốn áp dụng cho văn bản:
""")

# Phần nhập liệu
st.header("1. Nhập liệu")
input_method = st.radio("Chọn cách nhập liệu:", ["Nhập trực tiếp", "Tải lên file văn bản"])

text_input = ""
if input_method == "Nhập trực tiếp":
    text_input = st.text_area("Nhập văn bản cần tăng cường:", 
                            "This paper will use DLmethods to improve the performance of sentiment analysis.")
else:
    uploaded_file = st.file_uploader("Tải lên file (.txt, .csv, .docx, .pdf, .json)", type=["txt", "csv", "docx", "pdf", "json"])
    if uploaded_file:
        texts = process_uploaded_file(uploaded_file)
        if texts:
            text_input = "\n".join(texts)
            st.text_area("Nội dung file:", text_input, height=150)

# Phần lựa chọn phương pháp
st.header("2. Lựa chọn phương pháp")
col1, col2 = st.columns(2)

with col1:
    use_synonym = st.checkbox("Thay thế từ đồng nghĩa", value=True)
    use_swap = st.checkbox("Đảo vị trí từ", value=True)
    use_insert = st.checkbox("Thêm từ ngẫu nhiên")

with col2:
    use_delete = st.checkbox("Xóa từ ngẫu nhiên")
    use_backtrans = st.checkbox("Dịch qua lại (back translation)")

# Phần thực thi
if st.button("🎯 Thực thi") and text_input:
    results = []
    original_text = text_input
    
    # Tạo bản gốc
    results.append({
        "Phương pháp": "Văn bản gốc",
        "Kết quả": original_text,
        "Độ dài": len(original_text.split())
    })
    
    # Áp dụng các phương pháp được chọn
    if use_synonym:
        augmented = st.session_state.augmenter.synonym_augmentation(original_text)
        results.append({
            "Phương pháp": "Thay từ đồng nghĩa",
            "Kết quả": augmented,
            "Độ dài": len(augmented.split())
        })
    
    if use_swap:
        augmented = st.session_state.augmenter.swap_words(original_text)
        results.append({
            "Phương pháp": "Đảo vị trí từ",
            "Kết quả": augmented,
            "Độ dài": len(augmented.split())
        })
    
    if use_insert:
        augmented = st.session_state.augmenter.insert_words(original_text)
        results.append({
            "Phương pháp": "Thêm từ ngẫu nhiên",
            "Kết quả": augmented,
            "Độ dài": len(augmented.split())
        })
    
    if use_delete:
        augmented = st.session_state.augmenter.delete_words(original_text)
        results.append({
            "Phương pháp": "Xóa từ ngẫu nhiên",
            "Kết quả": augmented,
            "Độ dài": len(augmented.split())
        })
    
    if use_backtrans:
        augmented = st.session_state.augmenter.back_translation(original_text)
        results.append({
            "Phương pháp": "Dịch qua lại",
            "Kết quả": augmented,
            "Độ dài": len(augmented.split())
        })
    
    # Hiển thị kết quả
    st.header("3. Kết quả Tăng cường")
    df_results = pd.DataFrame(results)
    st.dataframe(df_results, use_container_width=True)
    
    # Tải kết quả về
    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Tải kết quả về (CSV)",
        data=csv,
        file_name='augmented_results.csv',
        mime='text/csv'
    )
    
    # Hiển thị chi tiết từng kết quả
    st.header("Chi tiết kết quả")
    for result in results:
        with st.expander(f"🔍 {result['Phương pháp']}"):
            st.write(result["Kết quả"])
            st.caption(f"Độ dài: {result['Độ dài']} từ")

elif not text_input:
    st.warning("Vui lòng nhập hoặc tải lên văn bản trước khi thực thi")