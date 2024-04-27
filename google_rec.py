import google.generativeai as gai
import pandas as pd

gai.configure(api_key = 'AIzaSyAAVPWfim3yEzb3dactBVsA_vKHCKQdN0M')
model = gai.GenerativeModel('gemini-pro')

text_search = input('Enter text: ')

text = 'Filter the text' + text_search + 'and return the key words as a python list'

response = model.generate_content(text)
print(response.text)

df = pd.read_csv('styles1.csv', delimiter=';')


# def word_search(text_search, df):
#     text_search = str(text_search.lower())
#     text_search = text_search.split()
#     matched_ids = []
#     for index, row in df.iterrows():
#         product_name = str(row['productDisplayName'])
#         product_name = product_name.lower()
#         words_list = product_name.split()
#         count = 0
#         for term in text_search:
#             if term in words_list:
#                 count += 1
#                 if count >= 3:  # At least two words match
#                     matched_ids.append(row['id'])
#                     break  # Exit the loop once two words match
#     return matched_ids

# if text_search:
#     matched_ids = word_search(text_search, df.copy())  
#     print(matched_ids)
    # N_cards_per_row = 3
    # for i, id in enumerate(matched_ids):
    #     if i % N_cards_per_row == 0:
    #         # st.write("---")
    #         cols = st.columns(N_cards_per_row, gap="large")
    #     row_index = df[df['id'] == id].index[0]  

    #     with cols[i%N_cards_per_row]:
    #         print(f"**{df.loc[row_index, 'productDisplayName'].strip()}**")
    #         print(f"{id}.jpg")

def search_all_columns(text_search, df):
    matched_ids = []
    for index, row in df.iterrows():
        count = 0
        for column in df.columns:
            column_value = str(row[column]).lower()
            words_list = column_value.split()
            for term in text_search:
                if term in words_list:
                    count += 1
                    if count >= 3:  # At least three words match in any column
                        matched_ids.append(row['id'])
                        break  # Exit the loop once the condition is met
            if count >= 3:
                break  # Exit the column loop if the condition is met
    return matched_ids

if text_search:
    matched_ids = search_all_columns(text_search.split(), df.copy())  
    print(matched_ids)

