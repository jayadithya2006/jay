import os
import re
import sys
import math
import openai
import logging
import datetime
import requests
import traceback
import subprocess
import numpy as np

from gtts import gTTS
from pydub import AudioSegment
from PIL import Image
from skimage import img_as_ubyte
from typing import List, Optional, Tuple

# OpenAI API Key
openai.api_key = "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"

# Constants
story_filename = "story.txt"
num_ChatGPT_images = 1
size_ChatGPT_image = "1024x1024"
frame_rate = 30
output_dir = "output"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("script.log"), logging.StreamHandler()],
)


def load_story_from_file(filename: str) -> Optional[List[str]]:
    """Load text lines from filename."""
    if not os.path.isfile(filename):
        logging.error(f"Story file '{filename}' not found.")
        return None

    with open(filename, "r") as f:
        story = [line.strip() for line in f if line.strip()]
    return story


def generate_and_download_image(prompt: str, image_file: str) -> Optional[str]:
    """Generate an image using OpenAI API and download it."""
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=num_ChatGPT_images,
            size=size_ChatGPT_image,
            response_format="url",
        )
        image_url = response["data"][0]["url"]

        response = requests.get(image_url)
        if response.status_code != 200:
            logging.error(f"Failed to download image for prompt: {prompt}")
            return None

        with open(image_file, "wb") as f:
            f.write(response.content)
        
        logging.info(f"Image saved: {image_file}")
        return image_file

    except openai.error.InvalidRequestError as e:
        logging.error(f"OpenAI request failed for '{prompt}': {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error in image generation: {e}")
        return None


def generate_audio(text: str, out_file: str) -> Optional[str]:
    """Generate audio from text using gTTS."""
    try:
        tts = gTTS(text=text, lang="en")
        tts.save(out_file)
        return out_file
    except Exception as e:
        logging.error(f"Error generating audio for '{text}': {e}")
        return None


def combine_all_audio(audio_files: List[str], output_file: str) -> Optional[str]:
    """Combine multiple audio files into one."""
    try:
        combined = AudioSegment.empty()
        for file in audio_files:
            audio = AudioSegment.from_mp3(file)
            combined += audio
        combined.export(output_file, format="mp3")
        return output_file
    except Exception as e:
        logging.error(f"Error combining audio files: {e}")
        return None


def morph_images(image1: str, image2: str, output_file: str):
    """Perform image morphing."""
    try:
        if not os.path.exists(image1) or not os.path.exists(image2):
            logging.error(f"One or both images missing: {image1}, {image2}")
            return None

        img1 = Image.open(image1)
        img2 = Image.open(image2)

        # Convert images to same size
        img2 = img2.resize(img1.size)
        output_image = Image.blend(img1, img2, alpha=0.5)
        output_image.save(output_file)
        return output_file
    except Exception as e:
        logging.error(f"Error morphing images: {e}")
        return None


def main():
    # Load story
    story = load_story_from_file(story_filename)
    if not story:
        logging.error("Story file could not be loaded.")
        return

    image_files = []
    audio_files = []

    # Process each line of story
    for idx, line in enumerate(story):
        logging.info(f"Processing line {idx + 1}: {line}")

        image_file = os.path.join(output_dir, f"image_{idx:04d}.png")
        audio_file = os.path.join(output_dir, f"{idx:04d}_audio.mp3")

        img_path = generate_and_download_image(line, image_file)
        if img_path:
            image_files.append(img_path)

        audio_path = generate_audio(line, audio_file)
        if audio_path:
            audio_files.append(audio_path)

    # Morph images
    morphed_images = []
    for i in range(len(image_files) - 1):
        morph_file = os.path.join(output_dir, f"morph_{i:04d}.png")
        morph_path = morph_images(image_files[i], image_files[i + 1], morph_file)
        if morph_path:
            morphed_images.append(morph_path)

    # Combine audio
    final_audio = os.path.join(output_dir, "final_audio.mp3")
    combine_all_audio(audio_files, final_audio)

    logging.info("Processing complete!")


if __name__ == "__main__":
    main()