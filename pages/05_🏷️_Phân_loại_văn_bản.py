import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from utils.helpers import display_title_icon, process_uploaded_file
import matplotlib.pyplot as plt

display_title_icon("🏷️", "Phân Loại Văn Bản")

class TextClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.models = {
            "Naive Bayes": MultinomialNB(),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "SVM": SVC(kernel='linear'),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier()
        }
        self.classes = None

    def preprocess_data(self, texts, labels=None):
        """Tiền xử lý và vector hóa dữ liệu"""
        if labels is None:
            return self.vectorizer.transform(texts)
        X = self.vectorizer.fit_transform(texts)
        self.classes = np.unique(labels)
        return X, labels

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Huấn luyện và đánh giá các mô hình"""
        results = {}
        for name, model in self.models.items():
            if name in st.session_state.selected_models:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                results[name] = accuracy_score(y_test, y_pred)
        return results

def detect_columns(df):
    """Tự động phát hiện cột văn bản và nhãn"""
    for col in df.columns:
        sample = str(df[col].iloc[0]) if len(df) > 0 else ""
        if len(sample.split()) > 3:
            text_col = col
            label_col = [c for c in df.columns if c != text_col][0]
            return text_col, label_col
    return None, None

def load_data():
    """Giao diện tải dữ liệu"""
    st.subheader("Nhập dữ liệu huấn luyện")
    
    input_type = st.radio("Chọn cách nhập dữ liệu:", 
                         ["Nhập tay", "Tải file"], 
                         horizontal=True)
    
    if input_type == "Nhập tay":
        text_input = st.text_area("Nhập văn bản (mỗi dòng một mẫu):", height=150)
        label_input = st.text_area("Nhập nhãn tương ứng (mỗi dòng một nhãn):", height=150)
        
        if text_input and label_input:
            texts = [line.strip() for line in text_input.split('\n') if line.strip()]
            labels = [line.strip() for line in label_input.split('\n') if line.strip()]
            return texts, labels
    else:
        uploaded_file = st.file_uploader("Tải file (.csv, .txt, .docx, .pdf, .json)", type=["csv", "txt", "docx", "pdf", "json"])
        if uploaded_file:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
                text_col, label_col = detect_columns(df)
                if text_col and label_col:
                    st.info(f"Đã nhận diện: Văn bản - '{text_col}', Nhãn - '{label_col}'")
                    return df[text_col].astype(str).tolist(), df[label_col].astype(str).tolist()
                st.error("Không thể tự động xác định cột văn bản và nhãn")
            else:
                texts = process_uploaded_file(uploaded_file)
                if texts:
                    st.warning("File không phải CSV. Nhãn cần được nhập riêng.")
                    label_input = st.text_area("Nhập nhãn (mỗi dòng một nhãn, tương ứng với văn bản):", height=150)
                    if label_input:
                        labels = [line.strip() for line in label_input.split('\n') if line.strip()]
                        if len(labels) == len(texts):
                            return texts, labels
                        st.error("Số lượng nhãn không khớp với số lượng văn bản")
    return None, None

def plot_algorithm_comparison(results, selected_models):
    """Vẽ biểu đồ cột so sánh accuracy của các thuật toán"""
    algorithms = selected_models
    accuracies = [results[algo] for algo in algorithms]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(algorithms, accuracies, color='#1f77b4')
    
    # Thêm giá trị trên đầu mỗi cột
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2%}', 
                ha='center', va='bottom', fontsize=10)
    
    plt.title('So sánh Accuracy của các Thuật toán Phân loại', fontsize=14, pad=20)
    plt.xlabel('Thuật toán', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Lưu biểu đồ vào file tạm thời
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'algorithm_comparison.png'

def main():
    st.markdown("""
    ## Công cụ Phân Loại Văn Bản
    
    Nhập dữ liệu huấn luyện và chọn thuật toán để phân loại
    """)
    
    # Tải dữ liệu
    texts, labels = load_data()
    if not texts or not labels:
        return

    # Chọn thuật toán
    selected_models = st.multiselect(
        "Chọn thuật toán phân loại:",
        ["Naive Bayes", "Logistic Regression", "SVM", "K-Nearest Neighbors", "Decision Tree"],
        default=["Naive Bayes"]
    )
    
    if not selected_models:
        st.warning("Vui lòng chọn ít nhất một thuật toán")
        return

    # Chia dữ liệu
    test_size = st.slider("Tỉ lệ dữ liệu kiểm tra:", 0.1, 0.5, 0.2, 0.05)

    if st.button("🏋️ Train"):
        classifier = TextClassifier()
        X, y = classifier.preprocess_data(texts, labels)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Lưu selected_models vào session state trước khi huấn luyện
        st.session_state.selected_models = selected_models
        
        results = classifier.train_and_evaluate(X_train, X_test, y_train, y_test)
        
        # Lưu classifier vào session state
        st.session_state.classifier = classifier
        
        # Hiển thị kết quả
        st.subheader("Kết quả đánh giá")
        results_df = pd.DataFrame({
            "Thuật toán": selected_models,
            "Accuracy": [results[name] for name in selected_models]
        })
        st.dataframe(results_df.style.format({"Accuracy": "{:.2%}"}))

        # Vẽ và hiển thị biểu đồ so sánh
        st.subheader("So sánh Accuracy các thuật toán")
        chart_path = plot_algorithm_comparison(results, selected_models)
        st.image(chart_path, caption="Biểu đồ so sánh Accuracy", use_container_width=True)

    if 'classifier' in st.session_state:
        st.subheader("Thử nghiệm với văn bản mới")
        new_text = st.text_input("Nhập văn bản cần phân loại:")
        
        if new_text and st.button("🔮 Dự đoán"):
            try:
                classifier = st.session_state.classifier
                selected_models = st.session_state.selected_models
                
                # In ra thông tin đầu vào
                st.write("Văn bản đầu vào:", new_text)
                
                # Tiền xử lý
                X_new = classifier.preprocess_data([new_text])
                st.write("Dữ liệu sau tiền xử lý:", X_new.shape)
                
                # Dự đoán
                predictions = {}
                for name in selected_models:
                    model = classifier.models[name]
                    predictions[name] = model.predict(X_new)[0]                
                
                st.success("Kết quả dự đoán:")
                st.write(pd.DataFrame.from_dict(predictions, orient='index', columns=['Nhãn dự đoán']))
                
            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {str(e)}")

if __name__ == "__main__":
    main()