import os
import re
import srt
import sys
import math
import logging
import datetime
import requests
import traceback
import subprocess
import numpy as np
import openai
from srt import Subtitle
from typing import List, Optional
from gtts import gTTS
from pydub import AudioSegment
from PIL import Image
from skimage import img_as_ubyte

# 
openai.api_key = "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"

# Constants
STORY_FILENAME = "story.txt"
NUM_CHATGPT_IMAGES = 1
SIZE_CHATGPT_IMAGE = "1024x1024"
FRAME_RATE = 30

# Output directory
OUTPUT_DIR = 'output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def load_story_from_file(filename: str) -> Optional[List[str]]:
    """Load text lines from a story file."""
    if not os.path.isfile(filename):
        print(f"Story file '{filename}' not found. Please provide a valid story file.")
        return None

    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip()]

import openai

openai.api_key = "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"

response = openai.Image.create(
    prompt="A futuristic city with flying cars",
    n=1,
    size="1024x1024"
)

print(response)
def generate_and_download_image(prompt: str, image_file: str) -> Optional[str]:
    """Generate an image using OpenAI's DALL-E API and save it locally."""
    try:
        response = openai.images.generate(
    model="dall-e-3",  # Use "dall-e-2" if you don’t have access to 3
    prompt=prompt,
    n=1,
    size="1024x1024"
)
        image_url = response.data[0].url

        # Download the image
        response = requests.get(image_url)
        if not response:
            raise ValueError("No image created")

        with open(image_file, "wb") as f:
            f.write(response.content)

        return image_file

    except Exception as e:
        logging.error(f"Error generating image for '{prompt}': {e}\n{traceback.format_exc()}")
        return None


def generate_audio(text: str, out_file: str) -> tuple:
    """Generate an audio file from text."""
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(out_file)
        audio = AudioSegment.from_mp3(out_file)
        duration = math.ceil(audio.duration_seconds)
        return out_file, duration
    except Exception as e:
        logging.error(f"Error generating audio: {e}\n{traceback.format_exc()}")
        return None, 0


def combine_all_audio(audio_files: List[str], output_file: str) -> str:
    """Combine multiple audio files into one."""
    merged_audio = AudioSegment.empty()
    for audio_file in audio_files:
        audio_segment = AudioSegment.from_file(audio_file)
        merged_audio += audio_segment

    merged_audio.export(output_file, format="mp3")
    return output_file


def combine_all_subtitles(subtitle_entries: List[dict], output_file: str) -> str:
    """Create subtitle file from subtitle entries."""
    subtitles = []
    start_time = datetime.timedelta()

    for index, entry in enumerate(subtitle_entries):
        duration = datetime.timedelta(seconds=entry["duration"])
        end_time = start_time + duration
        subtitle = Subtitle(index, start_time, end_time, entry['text'])
        subtitles.append(subtitle)
        start_time = end_time

    with open(output_file, "w") as f:
        f.write(srt.compose(subtitles))

    return output_file


def create_video(video_files: str, audio_file: str, subtitle_file: str, output_file: str) -> bool:
    """Create a video using images, audio, and subtitles."""
    try:
        cmd = f"""
        ffmpeg -framerate {FRAME_RATE} -i {OUTPUT_DIR}/morphed_image_%04d.png -i {audio_file} -i {subtitle_file} \
            -map 0:v -c:v libx265 -preset slow -crf 25 \
            -map 1:a -c:a copy -metadata:s:a:0 language=eng \
            -map 2:s -c:s mov_text -metadata:s:s:0 language=eng -disposition:s:s:0 default+forced \
            -y {output_file}
        """
        subprocess.run(cmd, shell=True, check=True)
        return True
    except Exception as e:
        logging.error(f"Error creating video: {e}\n{traceback.format_exc()}")
        return False


def morph_images(image1: str, image2: str, steps: int, output_dir: str, start_index: int) -> List[str]:
    """Morph two images into multiple transition frames."""
    try:
        with Image.open(image1) as img1, Image.open(image2) as img2:
            img1_array = np.array(img1, dtype=np.float32) / 255.0
            img2_array = np.array(img2, dtype=np.float32) / 255.0

        morphed_images = []
        for cnt in range(steps + 1):
            alpha = cnt / steps
            morphed_image = (1 - alpha) * img1_array + alpha * img2_array

            filename = f"{output_dir}/morphed_image_{start_index + cnt:04d}.png"
            Image.fromarray(img_as_ubyte(morphed_image)).save(filename)
            morphed_images.append(filename)

        return morphed_images

    except Exception as e:
        logging.error(f"Error morphing images: {e}\n{traceback.format_exc()}")
        return []


def main(story_file: List[str]):
    images = []
    audio_files = []
    subtitle_entries = []
    start_time = 0
    start_index = 0

    for cnt, line in enumerate(story_file):
        image_file = f"{OUTPUT_DIR}/image_{cnt:04d}.png"
        print(f"\n{cnt} Line {line} -> {image_file}")

        # Generate and download image
        image_file = generate_and_download_image(line, image_file)
        images.append(image_file)

        # Generate audio and get duration
        audio_file_name = f"{OUTPUT_DIR}/{cnt:04d}_audio.mp3"
        audio_file, duration = generate_audio(line, audio_file_name)
        audio_files.append(audio_file)

        # Create subtitle entry
        end_time = start_time + duration
        subtitle_entries.append({
            "start": f"{start_time:.3f}",
            "end": f"{end_time:.3f}",
            "text": line,
            "duration": duration
        })
        start_time = end_time

        # Morph images
        if cnt > 0:
            morph_steps = duration * FRAME_RATE
            morphed_images = morph_images(images[cnt - 1], images[cnt], morph_steps, OUTPUT_DIR, start_index)
            start_index += len(morphed_images)

    # Merge audio files
    merged_audio = f"{OUTPUT_DIR}/merged_audio.mp3"
    combined_audio = combine_all_audio(audio_files, merged_audio)

    # Create subtitle file
    subtitle_file = f"{OUTPUT_DIR}/subtitles.srt"
    combined_subtitles = combine_all_subtitles(subtitle_entries, subtitle_file)

    # Create final video
    video_file = f"{OUTPUT_DIR}/final_video.mp4"
    success = create_video(f"{OUTPUT_DIR}/morphed_image_%04d.png", combined_audio, combined_subtitles, video_file)

    if success:
        print("Video created successfully!")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, filename="video_generation.log", filemode="w",
                        format="%(asctime)s - %(levelname)s - %(message)s")

    story_data = load_story_from_file(STORY_FILENAME)
    if story_data:
        main(story_data)