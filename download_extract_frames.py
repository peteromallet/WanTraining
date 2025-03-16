#!/usr/bin/env python
import os
import sys
import random
import requests
import cv2
from dotenv import load_dotenv
import argparse
import base64
from openai import OpenAI
import time
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

load_dotenv()

def crop_frame(frame):
    """
    Crops the given frame down to the candidate resolution that it supports and 
    which is closest (minimizing the overall difference in dimensions) from the list:
      - 16:9 -> 832x480
      - 4x3 -> 704x544
      - 1x1 -> 624x624
      - 9:16 -> 480x832
      - 3x4 -> 544x704
    If no candidate fits (i.e. frame is smaller than all), returns the original frame.
    """
    h, w = frame.shape[:2]
    # List of candidate resolutions (width, height)
    candidates = [
        (832, 480),  # 16:9
        (704, 544),  # 4x3
        (624, 624),  # 1x1
        (480, 832),  # 9:16
        (544, 704)   # 3x4
    ]
    # Keep candidates which can fit into the current frame dimensions.
    valid_candidates = [(cw, ch) for (cw, ch) in candidates if w >= cw and h >= ch]
    if not valid_candidates:
        return frame  # No valid candidate; return original frame.
    # Choose the candidate that minimizes the difference sum in width and height.
    chosen = min(valid_candidates, key=lambda cand: abs(w - cand[0]) + abs(h - cand[1]))
    target_w, target_h = chosen
    start_x = (w - target_w) // 2
    start_y = (h - target_h) // 2
    cropped = frame[start_y:start_y + target_h, start_x:start_x + target_w]
    return cropped

def get_random_video(api_key, query):
    """
    Fetches up to 50 videos from Pexels based on the search query and returns one random video.
    """
    url = "https://api.pexels.com/videos/search"
    params = {
        "query": query,
        "per_page": 50
    }
    headers = {
        "Authorization": api_key
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        print("Error fetching videos:", response.status_code, response.text)
        sys.exit(1)
    
    data = response.json()
    videos = data.get("videos")
    if not videos:
        print("No videos found.")
        sys.exit(1)
    
    video = random.choice(videos)
    return video

def download_video(video):
    """
    Downloads the selected video file from the video dictionary.
    Returns the local file name where it is saved, or None if no suitable video found.
    """
    video_files = video.get("video_files", [])
    
    # List of candidate minimum resolutions (width, height)
    min_resolutions = [
        (832, 480),  # 16:9
        (704, 544),  # 4x3
        (624, 624),  # 1x1
        (480, 832),  # 9:16
        (544, 704)   # 3x4
    ]
    
    def qualifies(vf):
        if vf.get("file_type") != "video/mp4":
            return False
        if "width" not in vf or "height" not in vf:
            return False
        w = vf.get("width")
        h = vf.get("height")
        # Check if the video file meets any candidate minimum resolution.
        for (min_w, min_h) in min_resolutions:
            if w >= min_w and h >= min_h:
                return True
        return False

    # Filter video_files based on type and resolution
    qualified_video_files = [vf for vf in video_files if qualifies(vf)]
    if not qualified_video_files:
        print("No downloadable MP4 video file found meeting minimum resolution criteria.")
        return None
    
    file_url = None
    chosen_vf = None
    # Prioritize video files with "sd" quality if found
    for vf in qualified_video_files:
        if vf.get("quality") == "sd":
            chosen_vf = vf
            break
    if chosen_vf is None:
        chosen_vf = qualified_video_files[0]
    file_url = chosen_vf.get("link")
    
    if not file_url:
        print("Chosen video file does not have a link.")
        return None
    
    local_filename = f"{video['id']}.mp4"
    print("Downloading video from", file_url)
    response = requests.get(file_url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return local_filename

def encode_image(image_path):
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_prompt_from_images(control_path, target_path):
    """Get prompt from OpenAI API comparing control and target images"""
    try:
        client = OpenAI()
        
        # Encode both images
        control_base64 = encode_image(control_path)
        target_base64 = encode_image(target_path)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Describe the difference between these as if you're giving instructions to a videographer in a passive voice. 

The instructions should be very short - a short sentence ideally, and describe the change precisely.

Three examples:

i) if the first image shows a woman with no expression and the second is her laughing, it should be: "She smiles and laughs, looks to the side"
ii) If the first image shows a horse and the second shows a zoomed out image of that same horse, it should be: "The camera zooms out"
(iii) if the first image shows a man sitting down, and the second shows him standing up, it should be "He stands up, the camera pans upwards".
(iv) if there's no major difference, reply "no difference"

Reply with just the instructions:""",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{control_base64}",
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{target_base64}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=100,
        )
        
        return response.choices[0].message.content.strip('"')
    except Exception as e:
        print(f"Error getting prompt from OpenAI: {e}")
        return ""

def extract_frames(video_file, video_id):
    """
    Extracts the first frame and a random frame from 1-3 seconds into the video.
    Saves them as control/target pairs.
    Returns True if successful, False otherwise.
    """
    # Check if files already exist
    control_path = os.path.join("training_data/control", f"{video_id}.jpg")
    target_path = os.path.join("training_data/target", f"{video_id}.jpg")
    if os.path.exists(control_path) or os.path.exists(target_path):
        print(f"Frames for video {video_id} already exist, skipping...")
        return False, ""

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error opening video file.")
        return False, ""

    # Check the actual video resolution
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    min_resolutions = [
         (832, 480),  # 16:9
         (704, 544),  # 4x3
         (624, 624),  # 1x1
         (480, 832),  # 9:16
         (544, 704)   # 3x4
    ]
    if not any(actual_w >= cw and actual_h >= ch for cw, ch in min_resolutions):
         print(f"Video resolution too low: {actual_w}x{actual_h}")
         cap.release()
         return False, ""

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    if fps == 0:
        print("FPS is 0, cannot process video.")
        cap.release()
        return False, ""

    duration = total_frames / fps
    if duration < 3:
        print("Video too short for extraction. Duration:", duration)
        cap.release()
        return False, ""
    
    # Extract first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame0 = cap.read()
    if not ret:
        print("Failed to read the first frame.")
        cap.release()
        return False, ""
    
    # Pick a random offset between 3 and min(7, duration).
    lower_offset = 3
    upper_offset = 7 if duration >= 7 else duration
    offset = random.uniform(lower_offset, upper_offset)
    target_frame = int(offset * fps)
    if target_frame >= total_frames:
        target_frame = int(total_frames) - 1
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame1 = cap.read()
    if not ret:
        print("Failed to read the random frame.")
        cap.release()
        return False, ""
    
    # Add debug checks to verify frames are different
    frame_difference = cv2.absdiff(frame0, frame1)
    mean_diff = frame_difference.mean()
    print(f"Mean difference between frames: {mean_diff}")
    if mean_diff < 1.0:  # If frames are nearly identical
        print("Warning: Frames appear to be too similar")
        cap.release()
        return False, ""
    
    # Crop frames to one of the candidate resolutions.
    cropped_frame0 = crop_frame(frame0)
    cropped_frame1 = crop_frame(frame1)
    
    # Additional debug check after cropping
    cropped_difference = cv2.absdiff(cropped_frame0, cropped_frame1)
    cropped_mean_diff = cropped_difference.mean()
    print(f"Mean difference between cropped frames: {cropped_mean_diff}")
    
    # Create control and target directories if they don't exist
    os.makedirs("training_data/control", exist_ok=True)
    os.makedirs("training_data/target", exist_ok=True)

    # Save the first frame as control, second as target
    cv2.imwrite(control_path, cropped_frame0)
    cv2.imwrite(target_path, cropped_frame1)
    
    # Get prompt from OpenAI
    prompt = get_prompt_from_images(control_path, target_path)
    
    cap.release()
    print(f"Frames saved as control/{video_id}.jpg and target/{video_id}.jpg")
    print(f"Generated prompt: {prompt}")
    return True, prompt

@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(
        multiplier=600,  # 10 minutes in seconds (10 * 60)
        min=600,        # Minimum wait of 10 minutes
        max=18000       # Maximum wait of 5 hours (5 * 60 * 60)
    ),
    stop=stop_after_attempt(12)
)
def fetch_videos_with_retry(query):
    try:
        url = "https://api.pexels.com/videos/search"
        params = {
            "query": query,
            "per_page": 50
        }
        headers = {
            "Authorization": os.getenv("PEXELS_API_KEY")
        }
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 429:
            print(f"Rate limited. Waiting before retry...")
            raise Exception("Rate limit hit")
            
        if response.status_code != 200:
            print(f"Error fetching videos: {response.status_code} {response.text}")
            raise Exception(f"API error: {response.status_code}")
            
        data = response.json()
        videos = data.get("videos", [])
        if not videos:
            raise Exception("No videos found")
            
        return videos
    except Exception as e:
        print(f"Error occurred: {str(e)}. Retrying in a few moments...")
        raise

def main():
    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        print("PEXELS_API_KEY not found in environment.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Download random videos from Pexels and extract frames.")
    parser.add_argument("count", type=int, help="Number of videos to process")
    parser.add_argument("query", type=str, nargs="?", default="people",
                        help="Search query or category (e.g., people, peopledoing, things, creative)")
    parser.add_argument("--delete_current", action="store_true",
                        help="Delete all existing files in the training_data folder before processing")
    args = parser.parse_args()

    # If flag is set, delete current training data
    if args.delete_current:
        import shutil
        for path in ["training_data/control", "training_data/target", "training_data/prompts.json"]:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
        print("Existing training data deleted.")

    # Create training data structure
    os.makedirs("training_data/control", exist_ok=True)
    os.makedirs("training_data/target", exist_ok=True)

    # Load or create prompts.json
    import json
    prompts = {}
    if os.path.exists("training_data/prompts.json"):
        with open("training_data/prompts.json", "r") as f:
            prompts = json.load(f)
    else:
        with open("training_data/prompts.json", "w") as f:
            json.dump(prompts, f, indent=4)

    count = args.count
    raw_category = args.query
    
    # Hardcoded query categories
    query_categories = {
        "people": ["man", "woman", "boy", "girl", "lady", "gentleman", "child", "teen"],
        "peopledoing": ["eating", "dancing", "running", "studying", "working", "exercising", "playing", "cooking"],
        "thingshappening": ["explosion", "driving", "accident", "storm", "fire", "earthquake", "celebration"],
        "landscapes": ["mountain", "forest", "lake", "desert", "coast", "countryside", "island", "valley"]
    }

    # Split the provided query on commas in case of multiple categories (e.g.: "peopledoing,people")
    categories = [q.strip() for q in raw_category.split(",")]

    for i in range(count):
        while True:  # Keep trying until we get a valid video
            print(f"Processing video {i+1}/{count}")
            current_category = random.choice(categories) if len(categories) > 1 else categories[0]

            if current_category in query_categories:
                 current_query = random.choice(query_categories[current_category])
                 print(f"For iteration {i+1}, using hardcoded query for category '{current_category}': {current_query}")
            else:
                 current_query = current_category
                 print(f"For iteration {i+1}, using provided query: {current_query}")

            try:
                videos = fetch_videos_with_retry(current_query)
                video = random.choice(videos)
                video_id = video.get("id")
                if not video_id:
                    print("Video does not have an id, trying another...")
                    continue

                # Skip if video already exists in prompts
                if f"{video_id}.jpg" in prompts:
                    print(f"Video {video_id} already in dataset, trying another...")
                    continue

                print("Selected video id:", video_id)
                local_video_file = download_video(video)
                if local_video_file is None:
                    continue  # Try another video if download failed
                
                # Remove the first call to extract_frames
                success, prompt = extract_frames(local_video_file, video_id)
                if success:
                    prompts[f"{video_id}.jpg"] = {
                        "prompt": prompt,
                        "negative_prompt": ""
                    }
                    with open("training_data/prompts.json", "w") as f:
                        json.dump(prompts, f, indent=4)
                    
                    os.remove(local_video_file)
                    break  # Success! Move on to next video
                else:
                    print("Failed to process video, trying another...")
                    os.remove(local_video_file)
                    continue
            except Exception as e:
                print(f"Failed after all retries: {str(e)}")
                # Maybe log the failure and continue with next query

    print("Completed processing all videos.")

if __name__ == "__main__":
    main() 