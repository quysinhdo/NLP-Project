import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from utils.helpers import display_title_icon

def scrape_movie_data(url):
    try:
        # Gửi yêu cầu HTTP với header giả lập trình duyệt
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Phân tích HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        movie_items = soup.find_all('div', class_='rounded-lg border bg-card text-card-foreground shadow-sm overflow-hidden border-none')
        
        movies_data = []
        for item in movie_items:
            movie = {}
            
            # Tên phim
            title_tag = item.find('h3').find('a', class_='line-clamp-1 text-left text-sm font-normal text-foreground hover:text-primary')
            movie['title'] = title_tag.get_text(strip=True) if title_tag else 'N/A'
            
            # Tên tiếng Anh
            eng_title_tag = item.find('p', class_='text-muted-foreground mt-1 line-clamp-1 text-xs font-light')
            movie['english_title'] = eng_title_tag.get_text(strip=True) if eng_title_tag else 'N/A'
            
            # Năm phát hành và số tập
            info_div = item.find('div', class_='flex items-center gap-1.5 font-light')
            if info_div:
                spans = info_div.find_all('span', class_='inline-block text-xs text-muted-foreground')
                movie['year'] = spans[0].get_text(strip=True) if len(spans) > 0 else 'N/A'
                movie['episodes'] = spans[1].get_text(strip=True) if len(spans) > 1 else 'N/A'
            else:
                movie['year'] = 'N/A'
                movie['episodes'] = 'N/A'
            
            movies_data.append(movie)
        
        return movies_data
    
    except Exception as e:
        st.error(f"Lỗi khi cào dữ liệu: {str(e)}")
        return []

def main():
    display_title_icon("📥", "Thu thập Dữ liệu")
    
    st.markdown("""
    ## Công cụ Thu thập Dữ liệu
    
    Nhập URL trang web để cào thông tin phim (tên phim, tên tiếng Anh, năm phát hành, số tập).
    """)
    
    # Nhập URL
    st.subheader("Nhập URL trang web")
    default_url = "https://phimmoichillzz.com/"
    url = st.text_input("URL:", value=default_url)
    
    if st.button("🔍 Cào dữ liệu"):
        with st.spinner("Đang cào dữ liệu..."):
            movies_data = scrape_movie_data(url)
        
        if movies_data:
            # Chuyển thành DataFrame
            df = pd.DataFrame(movies_data)
            
            # Hiển thị dữ liệu
            st.subheader("Dữ liệu phim đã cào")
            st.dataframe(df, use_container_width=True)
            
            # Tùy chọn tải xuống CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="Tải xuống dữ liệu (CSV)",
                data=csv,
                file_name="movies_data.csv",
                mime="text/csv"
            )
            
            # Lưu vào session state
            st.session_state.movies_data = df
            
            st.success(f"Đã cào được {len(movies_data)} mục phim!")
        else:
            st.warning("Không cào được dữ liệu. Vui lòng kiểm tra URL hoặc kết nối.")

if __name__ == "__main__":
    main()