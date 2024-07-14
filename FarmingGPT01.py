import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import numpy as np
import os
from gtts import gTTS
import speech_recognition as sr

# Load FarmingGPT model
tokenizer = AutoTokenizer.from_pretrained("Franklin01/Llama-2-7b-farmingGPT-finetune")
model = AutoModelForCausalLM.from_pretrained("Franklin01/Llama-2-7b-farmingGPT-finetune")

# Load SeamlessM4T model for translation
translation_model_name = "facebook/nllb-200-distilled-600M"
translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_name)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name)

def translate(text, src_lang="hi", tgt_lang="en"):
    inputs = translation_tokenizer(text, return_tensors="pt", src_lang=src_lang, tgt_lang=tgt_lang)
    translated_tokens = translation_model.generate(**inputs)
    translated_text = translation_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return translated_text

def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language="hi-IN")
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError:
            return "Error with the request"

def text_to_speech(text, lang="hi"):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    os.system("mpg321 output.mp3")

st.title("FarmingGPT Chatbot")

user_input = st.text_input("Enter your message (in Hindi):")
if st.button("Translate and Respond"):
    translated_text = translate(user_input, src_lang="hi", tgt_lang="en")
    inputs = tokenizer(translated_text, return_tensors="pt")
    response = model.generate(**inputs)
    response_text = tokenizer.decode(response[0], skip_special_tokens=True)
    st.write("Response in English:", response_text)
    response_in_hindi = translate(response_text, src_lang="en", tgt_lang="hi")
    st.write("Response in Hindi:", response_in_hindi)

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
if uploaded_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.write("File uploaded successfully. Processing...")
    user_speech = speech_to_text("temp_audio.wav")
    st.write("You said:", user_speech)
    translated_speech = translate(user_speech, src_lang="hi", tgt_lang="en")
    inputs = tokenizer(translated_speech, return_tensors="pt")
    response = model.generate(**inputs)
    response_text = tokenizer.decode(response[0], skip_special_tokens=True)
    response_in_hindi = translate(response_text, src_lang="en", tgt_lang="hi")
    st.write("Response in Hindi:", response_in_hindi)
    text_to_speech(response_in_hindi)
