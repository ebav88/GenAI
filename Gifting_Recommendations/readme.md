**Gifting Recommendations**

This is a streamlit app - takes input such as age, gender, interests, hobbies and occassion as input and suggests Gifting options along with a URL

**Source data**:
Open sourced Amazon product data. It includes product details - id, desc, selling price and URL.

**Solution overview**:
It takes user input and matches to the most relevant products from the Product data through RAG (embeddings & dot vector distance calculation).

Then it sends to OpenAI model for formatting as per display needs.

**Note** 
This is a not real time app. It is built as part of GenAI learning. Feel free to replicate.
Install the required libraries from requirements. 
You would need OpenAI API key.

