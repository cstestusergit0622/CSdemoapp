# Import libraries
import pandas as pd
import numpy as np
import openai
import spacy
from fuzzywuzzy import fuzz , process
import streamlit as st
import webbrowser

# Give the key
openai.api_key = "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"


# Import data
about_df = pd.read_excel("AboutCornerShop.xlsx")
subset_data = pd.read_excel('dummy_data.xlsx', sheet_name="Sheet1")


# Parameters

#List of Key words we're looking for: 
categories_keywords = ["categories", "category", "categorize"]
price_keywords = ["price", "cost", "£"]
stock_count_keywords = ["stock count", "inventory", "stock"]
availability_keywords = ["availability", "available", "in stock", "supermarkets", "in store"]
allergen_keywords = ["allergen", "allergies", "allergens", "allergic", "allergy"]

keywords = categories_keywords + price_keywords + stock_count_keywords + availability_keywords + allergen_keywords


# Models
def about_sol(df, new_question):

    prompt_beginning = "I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, \
    I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, \
    I will respond with 'Unknown'.\n\n"

    given_qnas = ""

    for q,a in zip(df.question, df.answer):
        string = "Q: " + q + "\n" + "A: " + a + "\n\n"
        given_qnas = given_qnas + string

    new_qna = "Q: " + new_question + "\n" + "A: "

    total_prompt = given_qnas + new_qna

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt = total_prompt,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"]
    )

    answer = response.choices[0].text.strip()    
    
    return answer

def product_sol(subset_data, user_qsn):
    """INPUT: subset_data, a pandas data frame drawn from a .xlsx in the same format as the dummydata file. 
              user_qsn, a string which represents a user query about product categories, price, stock count, availability or allergen information
    OUTPUT: output_message, a string which answeres the user query.
    """
    
    # openai.api_key = 'YOUR API KEY HERE' ##################################### INSERT API KEY HERE#################
    model_name = 'text-davinci-003'
    
    def _get_ans_from_response(response:openai.openai_object.OpenAIObject) -> str:
        first = dict(response)['choices']
        sec = dict(first[0])
        return sec['text']

    #Classify query topic.
    def classify_query_ai(model_engine: str = model_name, prompt: str = "") -> bool:
        """
        INPUT: prompt, a question in ther for of a string
        OUTPUT: L, list of indicators where 
                    L[0]=1 indicates that the question is about categories 
                    L[1]=1 indicates that the question is about price
                    L[2]=1 indicates that the question is about stock count
                    L[3]=1 indicates that the question is about availability
                    L[4]=1 indicates that the question is about allergen information
        """
        # Send the request to the Chat GPT API
        response = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=100
        )
    
        answer = _get_ans_from_response(response).lower() 
        prompt = prompt.lower()
        #### print(answer) ################################################TEST
        categories_keywords = ["categories", "category", "categorize"]
        price_keywords = ["price", "cost", "£"]
        stock_count_keywords = ["stock count", "inventory", "stock"]
        availability_keywords = ["availability", "available", "in stock", "where", "supermarkets", "in store"]
        allergen_keywords = ["allergen", "allergies", "allergens", "allergic", "allergy"]
        
        L = [0, 0, 0, 0, 0]
        
        if any(keyword in prompt for keyword in categories_keywords):
            L[0] = 1
        # Check if the answer contains "PRICE" or related keywords
        if any(word in answer for word in price_keywords):
            L[1]=1
        if any(keyword in answer for keyword in stock_count_keywords):
            L[2] = 1
        if any(keyword in answer for keyword in availability_keywords):
            L[3] = 1
        if any(word in answer for word in allergen_keywords):
            L[4] = 1
        return L

    def format_price(price_str): #This formats the price so it's no longer "GBP XXXX" but "£XX.XX instead"
        value = int(price_str.replace("GBP ", ""))

        # Convert from pennies to pounds
        value = value / 100

        # Format as a string with 2 decimal places, and add the £ symbol
        formatted_price = f"£{value:.2f}"
    
        return formatted_price
    
    def get_matching_products(query, products, score_cutoff=80):
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(query)
        phrases = [chunk.text for chunk in doc.noun_chunks]

        matching_products = set()  # convert to set to automatically avoid duplicates
        for phrase in phrases:
            highest = process.extractOne(phrase, products, scorer=fuzz.token_set_ratio)
            if highest[1] >= score_cutoff:
                matching_products.add(highest[0])  # use add method for sets instead of append
            
        return list(matching_products)  # convert back to list for return
    
    def get_product_info(user_qsn):
        """This asks for a user to input a string asking for a product's category, price, stock, 
                availability or allergen information and responds with details about that product"""
    
        # Creating a list of prodcuts, prices and allergies
        products = subset_data['Product Name'].tolist()
        categories = subset_data['categories'].tolist()
        prices = subset_data['prices'].tolist()
        stock = subset_data['size'].tolist()
        availabiltymode = subset_data['availabiltyMode'].tolist()
        allergens = subset_data['allergens'].tolist()
        
        matched_products = get_matching_products(user_qsn, products)  #match products
        
        if len(matched_products) > 0:
            L = classify_query_ai(prompt = user_qsn)
            output_message = []
            for product in matched_products: 
                product_index = products.index(product)
                category = categories[product_index]
                price = prices[product_index]
                stock_count = stock[product_index]
                availabilty = availabiltymode[product_index]
                allergen = allergens[product_index]
                response = ""
                if L[1]==1:
                    response = f"The price of {product} is {format_price(price)}."
                if L[0] == 1: 
                    response += f" Can be found in the {category} category."
                if L[2] == 1:
                    if type(stock_count) == type(1) :
                        response += f" There are {stock_count} in stock."
                    else : 
                        response += f" There are none in stock."
                if L[3] == 1: 
                    if availabilty == "Digital":
                        response += f" They are available online."
                    elif availabilty == "Physical":
                        response += f" They are available in store."
                    else :
                        response += " There is no availability information."
                if L[4] == 1:
                    if not(allergen) or allergen == 'NaN':    # This should be the path everytime it's "NaN" but it's not working
                        response += f" No allergen information found."
                    elif allergen == "none":
                        response += f" There are no allergens."
                    else:
                        response += f" The allergen it contains is/are {allergen}."
                output_message.append(response)
            return('\n'.join(output_message))
        else:
            return("No matching products found in the dataset.")
    final_message = get_product_info(user_qsn)
    return (final_message)


# Answer a question:
# def answer_q ():
#     while True:
#         user_prompt = input("Enter your prompt: ")
#         if user_prompt.lower() == 'exit':
#             break

#         common_words = set(keywords).intersection(set(user_prompt.split(" ")))
    
#         if len(common_words)>=1:
#             answer = product_sol(subset_data, user_prompt)
#         else:
#             answer = about_sol(about_df, user_prompt)
#         print(answer)

# answer_q()
def answer_q(user_prompt):
    common_words = set(keywords).intersection(set(user_prompt.split(" ")))

    if len(common_words) >= 1:
        answer = product_sol(subset_data, user_prompt)
    else:
        answer = about_sol(about_df, user_prompt)
    return answer

####################################################################################
# Streamlit app Section

# def main():
#     st.title("Question Answering Chatbot")
#     st.write("Enter your question below:")

#     end_chat = False  # Variable to track if the chat should end
#     user_prompt = st.text_input("Prompt")

#     if st.button("Ask your question!"):
#         answer = answer_q(user_prompt)
#         st.write("Answer:", answer)

#     if st.button("End Chat"):
#         end_chat = True
#         st.write("Thanks for getting in touch with me today. Hope I was able to help! Goodbye!")  # Display goodbye message

#     if end_chat:
#         if st.button("Do you want to start a new chat again? Click Here!"):
#             webbrowser.open_new_tab("http://localhost:8501")  # Replace the URL with the appropriate Streamlit app URL

# if __name__ == "__main__":
#     main()

def main():
    st.title("Welcome to your very own personal Cornershop Assistant!")
    st.write("Feel free to ask me questions about Cornershop, Product Prices, Quantities, Allergen information!!")
    st.write("You can try questions like 'What is Cornershop', 'What is the price of Coffee' etc.")

    end_chat = False  # Variable to track if the chat should end
    user_prompt = st.text_input("Please enter your question in the text box below", key="text_input")

    if st.button("Submit"):
        answer = answer_q(user_prompt)
        st.write("Answer:", answer)

    if st.button("End Chat"):
        end_chat = True
        st.write("Thanks for getting in touch with me today. Hope I was able to help! Goodbye!")  # Display goodbye message

    if end_chat:
        if st.button("Do you want to start a new chat again? Click Here!"):
            st.legacy_caching.clear_cache()
            st.experimental_rerun()
            # st.empty()
            # st.write("Enter your question below:")  # Display the initial message
            # user_prompt = st.text_input("Prompt", key="text_input")  # Display the input component

if __name__ == "__main__":
    main()
