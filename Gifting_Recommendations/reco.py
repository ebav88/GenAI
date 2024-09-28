# Import Libraries

import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

client = OpenAI()
 
def logic():
    st.header("Gifting Recommendations")
    st.subheader("Tell us about the recipient")
   
    # Age Input
    age = st.number_input("Age", min_value=0, max_value=100)
      
    # Create a radio button for gender selection
    gender = st.radio(
    "Select your gender:",
    ('Male', 'Female'))
    
    # Interests Input
    st.subheader("Interests")
    interests = []
    interest_options = ["reading", "traveling", "photography", "cooking", "sports", "gaming", "gardening", "music", "art", "writing", "dancing", "technology", "fashion", "fitness", "exploring nature", "movies", "theatre", "history", "science", "meditation", "yoga", "fishing", "volunteering", "languages", "crafts", "coding", "animal care", "social media", "wine tasting", "home improvement", "DIY projects", "collecting", "learning new skills", "astrology", "virtual reality", "genealogy", "sailing", "bird watching"]
    interests = st.multiselect("Select interests", interest_options)
    
    # Hobbies Input
    st.subheader("Hobbies")
    hobbies = []
    hobby_options = ["drawing", "painting", "knitting", "woodworking", "playing musical instruments", "bird watching", "hiking", "camping", "fishing", "collecting stamps", "model building", "baking", "pottery", "scrapbooking", "playing board games", "puzzles", "swimming", "skating", "doll making", "sewing", "flower arranging", "kite flying", "cycling", "lego building", "puzzles", "origami", "singing", "acting", "tennis", "basketball", "running", "martial arts", "sculpting", "pottery", "surfing", "gardening", "diy crafts", "robotics", "archery", "rock climbing"]
    hobbies = st.multiselect("Select hobbies", hobby_options)
   
    # Occasion Selection
    st.header("Select the Occasion")
    occasion_options = ["Birthday", "Graduation Party", "Wedding", "Anniversary", "Holiday", "Other"]
    occasion = st.selectbox("Occasion", occasion_options)
 
   
    if st.button("Get Gift Suggestions"):
        st.header("Here are some gift suggestions!")
        st.write(f"Suggestions for a {occasion} gift for a {age}-year-old {gender} who likes {', '.join(interests)} and enjoys {', '.join(hobbies)}:")
        
        question = f"Gifts for a {occasion} gift for a {age}-year-old {gender} who likes {', '.join(interests)} and enjoys {', '.join(hobbies)}:"
        
        #Ecommerce dataset - file read
        df_ecom = pd.read_csv(r'all_product_embedding.csv')
        df_ecom['embeddings'] = df_ecom['embeddings'].apply(lambda x: np.array(eval(x)))
        
        #Product description embedding
        
        def get_embedding(text,model="text-embedding-ada-002"):
            # Ensure the input is a string
            if isinstance(text, str):
                text = text.replace("\n", " ")
            else:
                text = ""  # Assign an empty string if text is None or not a string
            return client.embeddings.create(input=[text],model=model).data[0].embedding
        
        #question embedding
        question_embedding = get_embedding(question)
        
        #calculate distance between question & product through dot product
        df_ecom['distance'] = df_ecom['embeddings'].apply(lambda x: np.dot(x, question_embedding))
         
        #sort the distance in descending. User question that is closest to the product descriptions are displayed in that desc.
        
        df_ecom.sort_values(by=['distance'],ascending=False,inplace=True)
        
        context = df_ecom['Product_description'].iloc[0] + "\n" + df_ecom['Product_description'].iloc[1] + "\n" + df_ecom['Product_description'].iloc[2] + "\n" + df_ecom['Product_description'].iloc[3] + "\n" + df_ecom['Product_description'].iloc[4]  + "\n" + df_ecom['Product_description'].iloc[5]  + "\n" + df_ecom['Product_description'].iloc[6]  + "\n" + df_ecom['Product_description'].iloc[7]  + "\n" + df_ecom['Product_description'].iloc[8]  + "\n" + df_ecom['Product_description'].iloc[9] 
        
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role":"system","content": "you are an assistant helping my product recommendation system to answer user's question. From the Product description, extract Product name, Product description ane embed Product URL Embedded if only available in the Product description. Bold the column names"},
                {"role":"user","content": question},
                {"role":"assistant","content": f"use product informations from only this content - {context}  to answer the user query. Each product description is separated by a new line. I want 10 suggestions. Strictly no hallucination!"}   
                ])
        
        st.write(response.choices[0].message.content)
        
 
logic()