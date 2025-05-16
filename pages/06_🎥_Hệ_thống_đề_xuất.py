import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.helpers import display_title_icon

@st.cache_data
def load_movie_data():
    try:
        movies_df = pd.read_csv("data/movies.csv")
        ratings_df = pd.read_csv("data/ratings.csv")
    except FileNotFoundError:
        st.error("Lỗi: Không tìm thấy file movies.csv hoặc ratings.csv trong thư mục 'data'. Vui lòng tạo dữ liệu mẫu hoặc đặt file đúng vị trí.")
        movies_data = {'movieId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                       'title': ['Toy Story (1995)', 'Jumanji (1995)', 'Grumpier Old Men (1995)', 
                                 'Waiting to Exhale (1995)', 'Father of the Bride Part II (1995)', 
                                 'Heat (1995)', 'Sabrina (1995)', 'Tom and Huck (1995)', 
                                 'Sudden Death (1995)', 'GoldenEye (1995)', 'American President, The (1995)',
                                 'Dracula: Dead and Loving It (1995)', 'Balto (1995)', 
                                 'Nixon (1995)', 'Cutthroat Island (1995)'],
                       'genres': ['Adventure|Animation|Children|Comedy|Fantasy', 'Adventure|Children|Fantasy', 
                                  'Comedy|Romance', 'Comedy|Drama|Romance', 'Comedy', 
                                  'Action|Crime|Thriller', 'Comedy|Romance', 'Adventure|Children',
                                  'Action', 'Action|Adventure|Thriller', 'Comedy|Drama|Romance',
                                  'Comedy|Horror', 'Adventure|Animation|Children', 'Drama',
                                  'Action|Adventure|Romance']}
        movies_df = pd.DataFrame(movies_data)

        ratings_data = {'userId': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 1, 2, 6, 6],
                        'movieId': [1, 3, 6, 1, 2, 7, 2, 3, 4, 5, 6, 8, 1, 4, 5, 2, 5, 1, 10],
                        'rating': [4.0, 4.0, 4.0, 3.0, 3.0, 4.5, 2.0, 5.0, 1.0, 3.0, 2.0, 4.0, 5.0, 4.0, 4.0, 5.0, 2.5, 3.5, 4.0],
                        'timestamp': [964982703, 964981247, 964982224, 1000000000, 1000000001, 1000000002,
                                      1100000000, 1100000001, 1100000002, 1200000000, 1200000001, 1200000002,
                                      1300000000, 1300000001, 1300000002, 964980000, 1000000003, 1400000000, 1400000001]}
        ratings_df = pd.DataFrame(ratings_data)
        st.info("Đang sử dụng dữ liệu phim và rating mẫu do không tìm thấy file.")

    avg_ratings = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    avg_ratings.columns = ['movieId', 'average_rating', 'rating_count']
    movies_df = movies_df.merge(avg_ratings, on='movieId', how='left')
    movies_df['average_rating'] = movies_df['average_rating'].fillna(0).round(2)
    movies_df['rating_count'] = movies_df['rating_count'].fillna(0).astype(int)
    return movies_df, ratings_df

@st.cache_data
def create_content_matrix(movies_df_input):
    movies_df = movies_df_input.copy()
    movies_df['genres_str'] = movies_df['genres'].str.replace('|', ' ', regex=False)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
    content_matrix = tfidf_vectorizer.fit_transform(movies_df['genres_str'])
    return content_matrix, movies_df

@st.cache_data
def calculate_content_similarity(_content_matrix_input):
    content_similarity_matrix = cosine_similarity(_content_matrix_input)
    return content_similarity_matrix

def get_content_based_recommendations(movie_id, movies_df, content_similarity_matrix, top_n=10, sort_by_rating_after_similarity=True):
    if movie_id not in movies_df['movieId'].values:
        st.warning(f"Movie ID {movie_id} không tồn tại trong dữ liệu.")
        return pd.DataFrame()
    try:
        movie_idx = movies_df[movies_df['movieId'] == movie_id].index[0]
    except IndexError:
        st.warning(f"Không tìm thấy index cho Movie ID {movie_id}.")
        return pd.DataFrame()

    sim_scores = list(enumerate(content_similarity_matrix[movie_idx]))
    sim_scores_sorted_by_similarity = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    candidate_sim_scores = sim_scores_sorted_by_similarity[1 : (top_n * 2) + 1] 
    
    candidate_movie_indices = [i[0] for i in candidate_sim_scores]
    # Sử dụng .loc để tránh SettingWithCopyWarning và đảm bảo hoạt động đúng với index gốc
    candidate_movie_ids = movies_df.loc[candidate_movie_indices, 'movieId']
        
    recommendations_df = movies_df[movies_df['movieId'].isin(candidate_movie_ids)].copy()
    
    sim_map_from_original_idx = {idx: score for idx, score in candidate_sim_scores}

    # Tạo cột 'original_index' để map similarity score
    recommendations_df['original_index'] = recommendations_df.index
    recommendations_df['similarity_score'] = recommendations_df['original_index'].map(sim_map_from_original_idx)
    recommendations_df.drop(columns=['original_index'], inplace=True)


    if recommendations_df.empty:
        return pd.DataFrame()

    if sort_by_rating_after_similarity:
        recommendations_df = recommendations_df.sort_values(
            by=['average_rating', 'similarity_score'],
            ascending=[False, False]
        )
    else:
        recommendations_df = recommendations_df.sort_values(by='similarity_score', ascending=False)
    
    final_recommendations_df = recommendations_df.head(top_n)
    
    return final_recommendations_df[['movieId', 'title', 'genres', 'average_rating', 'rating_count', 'similarity_score']]

@st.cache_data
def create_user_item_matrix(ratings_df_input):
    if ratings_df_input.empty:
        return pd.DataFrame()
    user_item_matrix = ratings_df_input.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    return user_item_matrix

def get_collaborative_recommendations_for_new_user(new_user_ratings, user_item_matrix_orig, movies_df, top_n=10):
    if not new_user_ratings: 
        st.warning("Bạn chưa đánh giá phim nào để nhận đề xuất cộng tác.")
        return pd.DataFrame()
    
    if user_item_matrix_orig.empty:
        st.warning("Không đủ dữ liệu người dùng hiện có để đưa ra đề xuất cộng tác.")
        return pd.DataFrame()

    new_user_series = pd.Series(index=user_item_matrix_orig.columns, dtype=float).fillna(0)
    for movie_id, rating in new_user_ratings.items():
        if movie_id in new_user_series.index:
            new_user_series[movie_id] = rating
    
    new_user_vector = new_user_series.values.reshape(1, -1)
    similarities_to_new_user = cosine_similarity(new_user_vector, user_item_matrix_orig.values)[0]
    
    new_user_sim_series = pd.Series(similarities_to_new_user, index=user_item_matrix_orig.index)
    
    top_similar_users = new_user_sim_series[new_user_sim_series > 0].sort_values(ascending=False).head(5)

    if top_similar_users.empty:
        st.info("Không tìm thấy người dùng nào đủ tương đồng với đánh giá của bạn.")
        return pd.DataFrame()

    movie_scores = {}
    movies_rated_by_new_user = list(new_user_ratings.keys())

    for other_user_id, similarity_score in top_similar_users.items():
        movies_rated_by_other = user_item_matrix_orig.loc[other_user_id]
        for movie_id_other, rating_other in movies_rated_by_other[movies_rated_by_other > 3.0].items():
            if movie_id_other not in movies_rated_by_new_user:
                if movie_id_other not in movie_scores:
                    movie_scores[movie_id_other] = {'score_sum': 0, 'sim_sum': 0}
                movie_scores[movie_id_other]['score_sum'] += rating_other * similarity_score
                movie_scores[movie_id_other]['sim_sum'] += similarity_score
    
    predicted_ratings_for_new_user = []
    for movie_id_pred, scores_data in movie_scores.items():
        if scores_data['sim_sum'] > 0:
            predicted_rating = scores_data['score_sum'] / scores_data['sim_sum']
            predicted_ratings_for_new_user.append((movie_id_pred, predicted_rating))
            
    predicted_ratings_for_new_user.sort(key=lambda x: x[1], reverse=True)
    
    recommended_items = []
    for rec_movie_id, pred_rating in predicted_ratings_for_new_user[:top_n]:
        if rec_movie_id in movies_df['movieId'].values:
            movie_info = movies_df[movies_df['movieId'] == rec_movie_id].iloc[0]
            recommended_items.append({
                'movieId': rec_movie_id,
                'title': movie_info['title'],
                'genres': movie_info['genres'],
                'average_rating_overall': movie_info['average_rating'],
                'predicted_score_for_you': round(pred_rating, 2)
            })
            
    return pd.DataFrame(recommended_items)

def main():
    display_title_icon("🎥", "Hệ thống Đề xuất Phim")
    
    movies_df_orig, ratings_df_orig = load_movie_data()
    if movies_df_orig.empty:
        return

    with st.spinner("Đang chuẩn bị dữ liệu nền..."):
        content_matrix, movies_df_cb = create_content_matrix(movies_df_orig)
        content_similarity_matrix = calculate_content_similarity(content_matrix)
        user_item_matrix_existing = create_user_item_matrix(ratings_df_orig)

    recommendation_type = st.radio(
        "Chọn phương pháp đề xuất:",
        ("Lọc dựa trên Nội dung", "Lọc Cộng tác"),
        horizontal=True,
        key="main_rec_type"
    )
    st.markdown("---")

    if recommendation_type == "Lọc dựa trên Nội dung":
        st.subheader("🎬 Đề xuất dựa trên Nội dung Phim")        
        
        movie_titles_cb = movies_df_cb['title'].tolist()
        search_query_cb = st.text_input("Tìm kiếm phim bạn thích:", key="cb_movie_search")
        
        default_index_cb = 0 if movie_titles_cb else None

        if search_query_cb:
            filtered_titles_cb = [title for title in movie_titles_cb if search_query_cb.lower() in title.lower()]
            if not filtered_titles_cb:
                st.warning("Không tìm thấy phim nào khớp với tìm kiếm.")
                selected_movie_title_cb = None
            else:
                selected_movie_title_cb = st.selectbox("Chọn phim:", filtered_titles_cb, key="cb_movie_select_filtered", index=0 if filtered_titles_cb else None)
        else:
            selected_movie_title_cb = st.selectbox("Chọn phim từ danh sách:", movie_titles_cb, key="cb_movie_select_all", index=default_index_cb)

        if selected_movie_title_cb:
            selected_movie_id_cb = movies_df_cb[movies_df_cb['title'] == selected_movie_title_cb]['movieId'].iloc[0]
            
            if st.button("🔍 Tìm phim tương tự", key="cb_get_recs"):
                recommendations_cb = get_content_based_recommendations(
                    selected_movie_id_cb,
                    movies_df_cb,
                    content_similarity_matrix,
                    sort_by_rating_after_similarity=True 
                )
                if not recommendations_cb.empty:
                    st.write(f"Đề xuất cho phim: **{selected_movie_title_cb}**")
                    st.dataframe(
                        recommendations_cb[['title', 'genres', 'average_rating', 'rating_count', 'similarity_score']].rename(
                            columns={'similarity_score': 'Độ tương đồng nội dung', 'average_rating': 'ĐTB Chung'}
                        ).reset_index(drop=True), 
                        use_container_width=True
                    )
                else:
                    st.info("Không tìm thấy đề xuất nội dung nào.")
        else:
             st.info("Vui lòng chọn một bộ phim để nhận đề xuất dựa trên nội dung.")


    elif recommendation_type == "Lọc Cộng tác":
        st.subheader("🤝 Đề xuất Cộng tác dựa trên Đánh giá của Bạn")        

        if 'user_ratings_for_cf' not in st.session_state:
            st.session_state.user_ratings_for_cf = {}
        if 'cf_movie_index' not in st.session_state:
            st.session_state.cf_movie_index = 0
        if 'cf_movies_to_rate_ids' not in st.session_state or not st.session_state.cf_movies_to_rate_ids:
            all_movie_ids = movies_df_orig['movieId'].unique().tolist()
            np.random.shuffle(all_movie_ids) 
            st.session_state.cf_movies_to_rate_ids = all_movie_ids
        
        num_ratings_target = st.slider("Số phim tối thiểu cần đánh giá để nhận đề xuất:", 1, 10, 3, key="num_target_cf")

        current_movie_idx = st.session_state.cf_movie_index
        
        if current_movie_idx < len(st.session_state.cf_movies_to_rate_ids):
            current_movie_id = st.session_state.cf_movies_to_rate_ids[current_movie_idx]
            current_movie_info = movies_df_orig[movies_df_orig['movieId'] == current_movie_id].iloc[0]

            st.markdown(f"#### Phim #{len(st.session_state.user_ratings_for_cf) + 1} để đánh giá:") # Hiển thị số thứ tự dựa trên số phim đã rate
            st.markdown(f"**{current_movie_info['title']}**")
            st.write(f"Thể loại: `{current_movie_info['genres']}`")

            rating_cols_container = st.container()
            with rating_cols_container:
                rating_cols = st.columns(6) 
                rating_value = 0
                user_action_taken = False


                for i in range(1, 6): 
                    button_key = f"rate_{i}_movie_{current_movie_id}"
                    if rating_cols[i-1].button(f"{i} ★", key=button_key):
                        rating_value = i
                        user_action_taken = True
                
                skip_button_key = f"skip_movie_{current_movie_id}"
                if rating_cols[5].button("Bỏ qua / Tiếp", key=skip_button_key):
                    rating_value = -1
                    user_action_taken = True

            if user_action_taken:
                if rating_value > 0: 
                    st.session_state.user_ratings_for_cf[current_movie_id] = float(rating_value)
                    st.success(f"Bạn đã đánh giá {current_movie_info['title']} là {rating_value} sao.")
                elif rating_value == -1:
                    st.info(f"Đã bỏ qua phim {current_movie_info['title']}.")
                
                st.session_state.cf_movie_index += 1
                
                # Đảm bảo cf_movie_index không vượt quá giới hạn và xử lý lặp lại nếu cần
                if st.session_state.cf_movie_index >= len(st.session_state.cf_movies_to_rate_ids):
                    st.info("Đã hết danh sách phim ban đầu. Bạn có thể nhận đề xuất hoặc reset.")                    
                st.rerun() 
        else: 
            st.info("Bạn đã xem qua hết danh sách phim gợi ý để đánh giá. Hãy nhận đề xuất hoặc reset.")
        
        num_rated_so_far = len(st.session_state.user_ratings_for_cf)
        if num_rated_so_far > 0 :
             st.markdown("---")
             st.write(f"Bạn đã đánh giá **{num_rated_so_far}** phim.")
        
        if num_rated_so_far >= num_ratings_target:
            if st.button("🌟 Hoàn thành đánh giá & Nhận đề xuất", key="cf_get_recs_final"):
                if not st.session_state.user_ratings_for_cf: 
                    st.warning("Vui lòng đánh giá ít nhất một phim.")
                else:
                    with st.spinner("Đang tìm kiếm đề xuất cộng tác cho bạn..."):
                        recommendations_cf_final = get_collaborative_recommendations_for_new_user(
                            st.session_state.user_ratings_for_cf,
                            user_item_matrix_existing,
                            movies_df_orig,
                            top_n=10
                        )
                    if not recommendations_cf_final.empty:
                        st.write("Dựa trên đánh giá của bạn, đây là những phim bạn có thể thích:")
                        st.dataframe(
                            recommendations_cf_final[['title', 'genres', 'average_rating_overall', 'predicted_score_for_you']].rename(
                                columns={'predicted_score_for_you': 'Điểm dự đoán cho bạn', 'average_rating_overall': 'ĐTB Chung'}
                            ).reset_index(drop=True), 
                            use_container_width=True
                        )
                    else:
                        st.info("Không tìm thấy đề xuất cộng tác nào dựa trên đánh giá hiện tại của bạn. Hãy thử đánh giá thêm hoặc các phim khác.")
        elif num_rated_so_far < num_ratings_target and num_rated_so_far > 0:
             st.info(f"Vui lòng đánh giá thêm {num_ratings_target - num_rated_so_far} phim nữa để nhận đề xuất (hoặc nhiều hơn nếu muốn).")
        elif num_rated_so_far == 0 and st.session_state.cf_movie_index > 0 :
            pass

        if st.button("Xóa tất cả đánh giá & Đánh giá lại", key="cf_reset_ratings_all"):
            st.session_state.user_ratings_for_cf = {}
            st.session_state.cf_movie_index = 0
            all_movie_ids_reset = movies_df_orig['movieId'].unique().tolist()
            np.random.shuffle(all_movie_ids_reset)
            st.session_state.cf_movies_to_rate_ids = all_movie_ids_reset
            st.rerun()

if __name__ == "__main__":
    main()