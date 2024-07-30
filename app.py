import streamlit as st
import os
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models

# Write the service account key to a file
with open("service_account.json", "w") as f:
    f.write(st.secrets["GOOGLE_APPLICATION_CREDENTIALS_CONTENT"])

# Set the environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account.json"

# Initialize Vertex AI
project_id = "765133488826"
vertexai.init(project=project_id, location="us-central1")

# Initialize the model
model = GenerativeModel("projects/765133488826/locations/us-central1/endpoints/7733417232985751552")

# Define the safety settings
safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

generation_config = {
    "max_output_tokens": 2048,
    "temperature": 1,
    "top_p": 1,
}

# Function to generate content from the model
def generate_response(prompt):
    response = model.generate_content(
        prompt,
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    return response.text

# Streamlit UI
st.title("FarmingGPT Chatbot")
st.write("Ask your questions related to farming.")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

message = st.text_input("Enter your message:", "")
if st.button("Send"):
    st.session_state['messages'].append(f"You: {message}")
    response = generate_response(message)
    st.session_state['messages'].append(f"FarmingGPT: {response}")

# Display conversation
for msg in st.session_state['messages']:
    st.write(msg)
