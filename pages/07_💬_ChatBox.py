import streamlit as st
from openai import OpenAI
import time
import pandas as pd
from utils.helpers import display_title_icon

API_KEY_CHAT = "1eac33c811f948089fe9f52bb9065838" 
BASE_URL_CHAT = "https://api.aimlapi.com/v1"

client_api = None
if API_KEY_CHAT and API_KEY_CHAT != "YOUR_API_KEY_HERE":
    try:
        client_api = OpenAI(base_url=BASE_URL_CHAT, api_key=API_KEY_CHAT)
    except Exception as e:
        st.error(f"Lỗi khởi tạo OpenAI client cho API: {e}")
else:
    st.warning("API key cho AIMLAPI chưa được cấu hình. Chức năng chat API sẽ không hoạt động.")


def chat_with_api(messages_api):
    if not client_api:
        return "Lỗi: API client chưa được khởi tạo. Vui lòng kiểm tra API key."
    try:
        response = client_api.chat.completions.create(
            model="gpt-4o", 
            messages=messages_api,
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Lỗi API: {str(e)}"

@st.cache_data 
def load_faq_data():
    try:
        df_faq = pd.read_csv("data/faq_dataset.csv")
        faq_dict = pd.Series(df_faq.answer.values,index=df_faq.question.str.lower().str.strip()).to_dict()
        st.success("Tải dữ liệu FAQ cho model tự train thành công!")
        return faq_dict
    except FileNotFoundError:
        st.error("Lỗi: Không tìm thấy file data/faq_dataset.csv. Chức năng model tự train (FAQ) sẽ không hoạt động.")
        return {}
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu FAQ: {e}")
        return {}

faq_lookup = load_faq_data()

def chat_with_local_model_faq(user_input_local):
    if not faq_lookup:
        return "Dữ liệu FAQ chưa sẵn sàng."
    
    processed_input = user_input_local.lower().strip()
    if processed_input in faq_lookup:
        return faq_lookup[processed_input]
    
    return "Xin lỗi, tôi chưa được huấn luyện để trả lời câu hỏi đó. Bạn có thể thử hỏi khác được không?"

def display_chat_messages(chat_history_list):
    for msg in chat_history_list:
        avatar_icon = "👤" if msg['role'] == 'user' else "🤖"
        with st.chat_message(msg['role'], avatar=avatar_icon):
            st.write(msg['content'])

def main():
    display_title_icon("💬", "Hệ thống Chatbox")

    if 'api_chat_history' not in st.session_state:
        st.session_state.api_chat_history = [{"role": "system", "content": "You are a helpful AI assistant."}]
    
    if 'local_model_chat_history' not in st.session_state: 
        st.session_state.local_model_chat_history = []
    
    st.subheader("Chọn mô hình Chatbot:")
    model_choice = st.radio(
        "Bạn muốn chat với:",
        ("API (gpt-4o)", "Model FAQ"), 
        horizontal=True,
        key="chatbot_model_choice"
    )
    st.markdown("---")

    if model_choice == "API (gpt-4o)":
        st.markdown("### Chat với API")
        
        if not client_api: 
            st.error("Không thể kết nối với API. Vui lòng kiểm tra cấu hình API key.")
        else:
            current_api_chat_display = [msg for msg in st.session_state.api_chat_history if msg['role'] != 'system']
            display_chat_messages(current_api_chat_display)

            user_input_api = st.chat_input("Nhập tin nhắn của bạn cho API...", key="api_user_input")

            if user_input_api:
                st.session_state.api_chat_history.append({"role": "user", "content": user_input_api})
                
                with st.spinner("API đang trả lời..."):
                    messages_to_send_api = st.session_state.api_chat_history 
                    api_response = chat_with_api(messages_to_send_api)
                
                st.session_state.api_chat_history.append({"role": "assistant", "content": api_response})
                st.rerun()

    elif model_choice == "Model FAQ":
        st.markdown("### Chat với Model FAQ")
        
        display_chat_messages(st.session_state.local_model_chat_history)

        user_input_local = st.chat_input("Nhập tin nhắn của bạn cho model FAQ...", key="local_user_input")

        if user_input_local:
            st.session_state.local_model_chat_history.append({"role": "user", "content": user_input_local})
            
            with st.spinner("Model FAQ đang xử lý..."):
                response_text_local = chat_with_local_model_faq(user_input_local) 
            
            st.session_state.local_model_chat_history.append({"role": "assistant", "content": response_text_local})
            st.rerun()

if __name__ == "__main__":
    main()