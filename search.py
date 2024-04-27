import streamlit as st
import pandas as pd
import re
import nltk


st.set_page_config(page_title="Python Talks Search Engine", layout="wide")
st.title("Clothing Search")

text_search = st.text_input("Search your choice", value="")

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
text_search = re.sub('[^a-zA-Z]',' ', text_search)
text_search = text_search.lower()
text_search = text_search.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
text_search = [ps.stem(word) for word in text_search if not word in set(all_stopwords)]
text_search = ' '.join(text_search)
st.write(text_search)

df = pd.read_csv('styles.csv', usecols= ['id', 'productDisplayName'])

# m1 = df['productDisplayName'].contains(text_search)

# df_search = m1
# N_cards_per_row = 3
# if text_search:
#     for n_row, row in m1.reset_index().iterrows():
#         i = n_row%N_cards_per_row
#         if i==0:
#             st.write("---")
#             cols = st.columns(N_cards_per_row, gap="large")
#         with cols[n_row%N_cards_per_row]:
#             st.markdown(f"**{row['productDis  playName'].strip()}**")
#             st.write(id+'.jpg')



# def word_search(text_search, df):
#     # matched_ids = []
#     # for i in df['productDisplayName']:
#     #     words_list = df['productDisplayName'][i].split()
#     #     if text_search in words_list:
#     #         matched_ids.append(df['id'])
#     # return matched_ids
#     text_search = str(text_search.lower())
#     text_search = text_search.split()
#     matched_ids = []
#     for index, row in df.iterrows():
#         product_name = str(row['productDisplayName'])
#         product_name = product_name.lower()
#         # st.write(product_name)
#         words_list = product_name.split()
#         # st.write(words_list)
#         count = 0
#         if any(term in words_list for term in text_search):
#             count = count+1
#         if count >= 2:
#             matched_ids.append(row['id'])
#         # st.write(matched_ids)
#     return matched_ids

def word_search(text_search, df):
    text_search = str(text_search.lower())
    text_search = text_search.split()
    matched_ids = []
    for index, row in df.iterrows():
        product_name = str(row['productDisplayName'])
        product_name = product_name.lower()
        words_list = product_name.split()
        count = 0
        for term in text_search:
            if term in words_list:
                count += 1
                if count >= 3:  # At least two words match
                    matched_ids.append(row['id'])
                    break  # Exit the loop once two words match
    return matched_ids

if text_search:
    matched_ids = word_search(text_search, df.copy())  

    N_cards_per_row = 3
    for i, id in enumerate(matched_ids):
        if i % N_cards_per_row == 0:
            st.write("---")
            cols = st.columns(N_cards_per_row, gap="large")
        row_index = df[df['id'] == id].index[0]  

        with cols[i%N_cards_per_row]:
            st.markdown(f"**{df.loc[row_index, 'productDisplayName'].strip()}**")
            st.write(f"{id}.jpg")