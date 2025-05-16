import streamlit as st
from utils.helpers import display_title_icon

# Cấu hình trang
st.set_page_config(
    page_title="NLP Project - Xử lý Ngôn ngữ Tự nhiên",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tiêu đề chính
display_title_icon("🤖", "Dự án Môn học Xử lý Ngôn ngữ Tự nhiên")
st.markdown("---")

# Giới thiệu tổng quan
st.markdown("""
## 🏠 Trang Chủ

Chào mừng bạn đến với dự án môn học Xử lý Ngôn ngữ Tự nhiên! Trang web này minh họa toàn bộ quy trình xử lý văn bản trong NLP.

**Các bước chính trong dự án:**
1. 📥 **Thu thập dữ liệu**: Crawl dữ liệu từ các nguồn khác nhau
2. 🛠️ **Tăng cường dữ liệu**: Augmentation dữ liệu văn bản
3. ✂️ **Tiền xử lý dữ liệu**: Làm sạch và chuẩn hóa văn bản
4. 🧮 **Biểu diễn dữ liệu**: Chuyển đổi văn bản thành vector
5. 🏷️ **Phân loại văn bản**: Áp dụng các mô hình machine learning
6. 🎥 **Hệ thống đề xuất**: Lọc nội dung phim
7. 💬 **ChatBox**: API và tự train model

**Hướng dẫn sử dụng:**
- Chọn các bước xử lý từ menu bên trái
- Mỗi trang sẽ có các chức năng tương tác để thực hành
- Code minh họa sẽ được hiển thị để tham khảo
""")

# Thông tin sinh viên
st.markdown("---")
with st.expander("ℹ️ Thông tin sinh viên"):
    st.write("""
    **Họ tên:** Đỗ Quý Sinh  
    **MSSV:** 22110222  
    **Lớp:** 22110CL_AI  
    **Giảng viên:** Phan Thị Huyền Trang
    """)



st.markdown("---")
st.success("👉 Chọn một bước xử lý từ sidebar để bắt đầu!")