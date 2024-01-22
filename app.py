import time
import streamlit as st
from dotenv import load_dotenv
import os
from clarifai.client.model import Model
import json
import base64
from moviepy.editor import ImageSequenceClip
from moviepy.editor import AudioFileClip
from moviepy.editor import vfx


load_dotenv()
clarifai_pat = os.getenv('CLARIFAI_PAT')


def generate_prompts(user_prompt):
        
    system_prompt = "You are an experienced social media growth and content maker. Your job is to make a short form video of 30 secs long based on the user's prompt. To do this you will generate 6 detailed prompts for feeding into DALLE to generate the images for the video and 1 tts prompt explaining the entire video with the help of the generated images. Return only the prompts as JSON response. The JSON should be in a format of image_prompts:[prompt:text] and tts_prompt:text" 
    inference_params = dict(temperature=0.9 ,top_p=1,  max_tokens=2048, system_prompt=system_prompt)
    # Model Predict
    model_prediction = Model("https://clarifai.com/openai/chat-completion/models/gpt-4-turbo").predict_by_bytes(user_prompt.encode(), input_type="text", inference_params=inference_params)
    testin = model_prediction.outputs[0].data.text.raw

    clean_output = testin.replace('json', '', 1).strip('`')
    valid = validate_json(clean_output)

    return valid


def validate_json(json_str):
    try:
        json.loads(json_str)
        return json.loads(json_str)
    except json.JSONDecodeError:
        return False


def generate_image(prompt, index):
    inference_params = dict(quality="standard", size='1024x1792')
    model_prediction = Model("https://clarifai.com/openai/dall-e/models/dall-e-3").predict_by_bytes(prompt.encode(), input_type="text", inference_params=inference_params)
    output_base64 = model_prediction.outputs[0].data.image.base64

    filename = f'image_{index:03d}.png'
    with open(filename, 'wb') as f:
        f.write(output_base64)
    return filename


def create_video(image_filenames, fps):
    # Create a video clip from images
    clip = ImageSequenceClip(image_filenames, fps=fps)
    audio = AudioFileClip("audio.mp3")
    clip = clip.set_audio(audio)
    clip.write_videofile("output.mp4", codec="libx264")


def understand_file(uploaded_file):
    system_prompt = "Explain what the image is about, including all specific details." 
    base64image = base64.b64encode(uploaded_file).decode('utf-8')

    inference_params = dict(temperature=0.2, max_tokens=100, image_base64=base64image)
    model_prediction = Model("https://clarifai.com/openai/chat-completion/models/gpt-4-vision").predict_by_bytes(system_prompt.encode(), input_type="text", inference_params=inference_params)
    vid_prompt = model_prediction.outputs[0].data.text.raw

    return vid_prompt

def generate_tts(prompt):
    inference_params = dict(voice="alloy", speed=1.0)
    model_prediction = Model("https://clarifai.com/openai/tts/models/openai-tts-1-hd").predict_by_bytes(prompt.encode(), input_type="text", inference_params=inference_params)
    output_base64 = model_prediction.outputs[0].data.audio.base64   
    with open("audio.mp3", "wb") as f:
        f.write(output_base64)
    return output_base64


def main(user_prompt):
    prompts = generate_prompts(user_prompt)
    image_filenames = []
    testing = generate_tts(prompts["tts_prompt"])

    for i in range(len(prompts["image_prompts"])):
        filename = generate_image(prompts["image_prompts"][i]["prompt"], i)
        image_filenames.append(filename)

    create_video(image_filenames, fps=0.4)
    for file_name in image_filenames:
        os.remove(file_name)
    os.remove("audio.mp3")

    video_file = open('output.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)


st.title("AI Reels Generator")
rad=st.sidebar.radio("Navigation",["Home","About Us"])
if rad=="Home":
    user_prompt = st.text_input("Enter the description of the video you want", "    ")

    # Generate image button
    if st.button("Generate Video from prompt"):
        try:
            st.text("Sit tight and wait for the magic!")
            main(user_prompt)
        
            with open("output.mp4", "rb") as file:
                btn = st.download_button(
            label="Download Video",
            data=file,
            file_name="output.mp4",
            mime="video/mp4"
          )

        except Exception as e:
            st.error(f"Error: {e}")

    uploaded_file=st.file_uploader("Upload a file to use it as the video script")
    if st.button("Generate Video from file") and uploaded_file is not None:
        bytes_data = uploaded_file.read()
        st.text("Sit tight and wait for the magic!")
        vid = understand_file(bytes_data)
        main(vid)