import os
import logging
import openai
import requests
from gtts import gTTS
from pydub import AudioSegment
from typing import List, Optional
import subprocess

# OpenAI API Key (Leave empty if using offline mode)
openai.api_key = ""

# Constants
story_filename = "story.txt"
num_ChatGPT_images = 1
size_ChatGPT_image = "1024x1024"
frame_rate = 1  # Set frame rate based on scene timing
output_dir = "output"
use_existing_images = False  # Automatically set if API fails

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("script.log"), logging.StreamHandler()],
)

def load_story_from_file(filename: str) -> Optional[List[str]]:
    """Load text lines from a story file."""
    if not os.path.isfile(filename):
        logging.error(f"Story file '{filename}' not found.")
        return None
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip()]

def check_openai_api():
    """Check if OpenAI API is available."""
    global use_existing_images
    try:
        openai.Image.create(prompt="Test", n=1, size="256x256")
        return True  # API works
    except Exception as e:
        logging.warning(f"OpenAI API unavailable: {e}")
        use_existing_images = True  # Use existing images instead
        return False

def generate_and_download_image(prompt: str, image_file: str) -> Optional[str]:
    """Generate an image using OpenAI API or use existing images."""
    if use_existing_images:
        if os.path.exists(image_file):
            logging.info(f"Using existing image: {image_file}")
            return image_file
        else:
            logging.error(f"Image '{image_file}' not found. Skipping...")
            return None
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=num_ChatGPT_images,
            size=size_ChatGPT_image,
            response_format="url",
        )
        image_url = response["data"][0]["url"]
        image_data = requests.get(image_url)
        if image_data.status_code != 200:
            logging.error(f"Failed to download image for prompt: {prompt}")
            return None
        with open(image_file, "wb") as f:
            f.write(image_data.content)
        logging.info(f"Image saved: {image_file}")
        return image_file
    except Exception as e:
        logging.error(f"Image generation failed for '{prompt}': {e}")
        return None

def generate_audio(text: str, out_file: str) -> Optional[str]:
    """Generate audio from text using gTTS."""
    try:
        tts = gTTS(text=text, lang="en")
        tts.save(out_file)
        logging.info(f"Audio saved: {out_file}")
        return out_file
    except Exception as e:
        logging.error(f"Error generating audio: {e}")
        return None

def combine_all_audio(audio_files: List[str], output_file: str) -> Optional[str]:
    """Combine multiple audio files into one."""
    try:
        combined = AudioSegment.empty()
        for file in audio_files:
            audio = AudioSegment.from_mp3(file)
            combined += audio
        combined.export(output_file, format="mp3")
        logging.info(f"Final audio saved: {output_file}")
        return output_file
    except Exception as e:
        logging.error(f"Error combining audio files: {e}")
        return None

def create_video_from_images(image_files: List[str], output_video: str):
    """Create a video from images using ffmpeg."""
    try:
        # Create a temporary text file for ffmpeg input
        with open("input.txt", "w") as f:
            for img in image_files:
                f.write(f"file '{img}'\n")

        # Use ffmpeg to create a video from images
        subprocess.run([
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", "input.txt",
            "-vsync", "vfr",
            "-pix_fmt", "yuv420p",
            "-r", str(frame_rate),
            output_video
        ], check=True)

        logging.info(f"Video created from images: {output_video}")

    except Exception as e:
        logging.error(f"Error creating video from images: {e}")

def main():
    global use_existing_images
    
    # Load story
    story = load_story_from_file(story_filename)
    if not story:
        logging.error("Story file could not be loaded.")
        return
    
    # Check API availability
    api_available = check_openai_api()
    
    image_files = []
    audio_files = []
    
    # Process each line in the story
    for idx, line in enumerate(story):
        logging.info(f"Processing line {idx + 1}: {line}")
        image_file = os.path.join(output_dir, f"image_{idx:04d}.png")
        audio_file = os.path.join(output_dir, f"{idx:04d}_audio.mp3")
        
        # Generate images if API is available, otherwise use existing images
        img_path = generate_and_download_image(line, image_file)
        if img_path:
            image_files.append(img_path)
        
        # Generate audio
        audio_path = generate_audio(line, audio_file)
        if audio_path:
            audio_files.append(audio_path)
    
    # Ensure images exist
    if not image_files:
        logging.error("No images available. Exiting...")
        return
    
    # Combine all audio into one file
    final_audio = os.path.join(output_dir, "final_audio.mp3")
    combine_all_audio(audio_files, final_audio)
    
    # Create video from images
    final_video = os.path.join(output_dir, "final_video.mp4")
    create_video_from_images(image_files, final_video)
    
    # Combine video and audio using ffmpeg
    final_output = os.path.join(output_dir, "final_output.mp4")
    subprocess.run([
        "ffmpeg",
        "-i", final_video,
        "-i", final_audio,
        "-c:v", "copy",
        "-c:a", "aac",
        "-strict", "experimental",
        final_output
    ], check=True)

    logging.info("✅ AI Video Creation Complete!")

if __name__ == "__main__":
    main()