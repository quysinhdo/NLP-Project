import streamlit as st
import pandas as pd
from docx import Document
import PyPDF2
import json
import io

def display_title_icon(icon, title):
    """Hiển thị tiêu đề với icon đồng nhất giữa các trang"""
    st.markdown(f"<h1 style='text-align: center;'>{icon} {title}</h1>", 
                unsafe_allow_html=True)

def display_code(code, language="python"):
    """Hiển thị code với syntax highlighting"""
    st.code(code, language=language)
    
def show_dataset_preview(df, n_rows=5):
    """Hiển thị preview dataset"""
    st.write(f"**Preview dataset (first {n_rows} rows):**")
    st.dataframe(df.head(n_rows))
    st.write(f"**Shape:** {df.shape}")

def process_uploaded_file(uploaded_file):
    """Xử lý các định dạng file khác nhau và trả về danh sách văn bản."""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        texts = []

        if file_extension == 'txt':
            texts = [line.strip() for line in uploaded_file.read().decode('utf-8').split('\n') if line.strip()]
        
        elif file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
            text_col = None
            for col in df.columns:
                if df[col].dtype == object and len(str(df[col].iloc[0]).split()) > 3:
                    text_col = col
                    break
            if text_col:
                texts = df[text_col].astype(str).tolist()
            else:
                st.error("Không tìm thấy cột văn bản trong file CSV")
                return []
        
        elif file_extension == 'docx':
            doc = Document(io.BytesIO(uploaded_file.read()))
            texts = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        
        elif file_extension == 'pdf':
            reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            texts = [page.extract_text().strip() for page in reader.pages if page.extract_text().strip()]
        
        elif file_extension == 'json':
            data = json.load(uploaded_file)
            if isinstance(data, list):
                texts = [str(item.get('text', '')) for item in data if 'text' in item]
            elif isinstance(data, dict):
                texts = [str(data.get('text', ''))] if 'text' in data else []
            else:
                st.error("Cấu trúc JSON không hỗ trợ")
                return []
        
        else:
            st.error(f"Định dạng file {file_extension} không được hỗ trợ")
            return []
        
        return texts if texts else ["Không tìm thấy nội dung hợp lệ trong file"]
    
    except Exception as e:
        st.error(f"Lỗi khi xử lý file: {str(e)}")
        return []