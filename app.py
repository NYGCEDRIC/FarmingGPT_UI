import streamlit as st
import os
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models

# Write the service account key to a file from the secrets
with open("service_account.json", "w") as f:
    f.write(st.secrets["GOOGLE_APPLICATION_CREDENTIALS_CONTENT"])

# Set the environment variable for Google Application Credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account.json"

# Function to generate content from the model
def multiturn_generate_content():
    vertexai.init(project="765133488826", location="us-central1")
    model = GenerativeModel("projects/765133488826/locations/us-central1/endpoints/7733417232985751552")
    chat = model.start_chat()
    responses = []
    messages = [
        "Hi",
        "I have a question about it",
        "how can I start my laptop",
        text4_1,
        "asking about the control measure to control anthracnose disease in capsicum",
        "asking about the control measure to control anthracnose disease in chilli."
    ]
    for message in messages:
        response = chat.send_message([message], generation_config=generation_config, safety_settings=safety_settings)
        responses.append(response.result)
    return responses

text4_1 = "What are the the fertilizers dosages in terms of per tree and the procedure of application in the 11th year stage of coconut trees."

generation_config = {
    "max_output_tokens": 2048,
    "temperature": 1,
    "top_p": 1,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

# Streamlit UI
st.title("FarmingGPT Chatbot")
st.write("Ask your questions related to farming.")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

message = st.text_input("Enter your message:", "")
if st.button("Send"):
    st.session_state['messages
