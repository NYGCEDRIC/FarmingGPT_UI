import streamlit as st
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models


def multiturn_generate_content(messages):
    vertexai.init(project="765133488826", location="us-central1")
    model = GenerativeModel(
        "projects/765133488826/locations/us-central1/endpoints/7733417232985751552",
    )
    chat = model.start_chat()
    responses = []
    for message in messages:
        response = chat.send_message(
            [message],
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        responses.append(response)
    return responses


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

text4_1 = """What are the the fertilizers dosages in terms of per tree and the procedure of application in the 11th year stage of coconut trees."""

# Streamlit UI
st.title("FarmingGPT Chatbot")
st.write("Ask your questions related to farming.")

if 'messages' n
