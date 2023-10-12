import streamlit as st
import speech_recognition as sr
import numpy as np
from scipy.spatial import distance
import librosa
import librosa.display
import pyaudio

#Initialize the recognizer
r = sr.Recognizer()


st.title("Voice Recognition (Microphone)")


# Function to get or create the SessionState for audio1 and audio2
def get_session_state():
    if "audio1" not in st.session_state:
        st.session_state.audio1 = [0]
    if "audio2" not in st.session_state:
        st.session_state.audio2 = [0]
    

    return st.session_state

session_state = get_session_state()

audio_One = session_state.audio1
audio_Two = session_state.audio2


st.write("Click here Person1")
# Create a Streamlit button
if st.button("Person1"):
    with sr.Microphone() as source:
        st.write('Say Something (first person)...')
        audio1 = r.listen(source)   #say anything
        
#         #Updating nwe audio
#         a = audio1
#         audio1.append(a)
#         session_state.audio1 = audio1

    person1_result=r.recognize_google(audio1)
    st.write(person1_result)
    
    #Updaing new audio
    a = audio1
    audio_One.append(a)
    session_state.audio1 = audio_One
    

st.write("Click here Person2")
# Capture the second audio sample
if st.button("Person2"):
    with sr.Microphone() as source:
        st.write('Say Something (second person)...')
        audio2 = r.listen(source)  #say anything
        
#     #Updating new audio
#     b = audio2
#     audio2.append(b)
#     session_state.audio2 = audio2

    person2_result=r.recognize_google(audio2)
    st.write(person2_result)
    
    #Updating new audio
    b = audio2
    audio_Two.append(b)
    session_state.audio2 = audio_Two
    
    
    
    
    
st.write("Click here to see the Result")
if st.button("Check Result"):
    audio1=audio_One[-1]
    audio2=audio_Two[-1]
    # Convert the audio data to floating-point arrays
    audio_data1 = np.frombuffer(audio1.frame_data, dtype=np.int16).astype(np.float32)
    audio_data2 = np.frombuffer(audio2.frame_data, dtype=np.int16).astype(np.float32)
    
    # Extract MFCC features from the audio samples
    mfcc1 = librosa.feature.mfcc(y=audio_data1, sr=audio1.sample_rate, n_mfcc=13)
    mfcc2 = librosa.feature.mfcc(y=audio_data2, sr=audio2.sample_rate, n_mfcc=13)

    # Transpose the MFCC matrices to have the same shape for comparison
    mfcc1 = mfcc1.T
    mfcc2 = mfcc2.T


    # Compute the cosine similarity between the MFCC feature vectors
    similarity_score = 1 - distance.cosine(mfcc1.mean(axis=0), mfcc2.mean(axis=0))

    # Set a threshold for similarity (you may need to determine this empirically)
    threshold = 1

    # Compare the similarity score to the threshold
    if similarity_score >= threshold:
        st.write("The voices match.")
    else:
        st.write("The voices do not match.")
