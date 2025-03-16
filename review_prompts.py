import streamlit as st
import os
import json
import glob
from PIL import Image

def load_data():
    """Load prompts and get all image pairs"""
    with open("training_data/prompts.json", "r") as f:
        prompts = json.load(f)
    return prompts

def save_data(prompts):
    """Save updated prompts"""
    with open("training_data/prompts.json", "w") as f:
        json.dump(prompts, f, indent=4)

def delete_pair(image_id, prompts):
    """Delete an image pair and its prompt"""
    # Remove files
    control_path = os.path.join("training_data/control", image_id)
    target_path = os.path.join("training_data/target", image_id)
    
    if os.path.exists(control_path):
        os.remove(control_path)
    if os.path.exists(target_path):
        os.remove(target_path)
    
    # Remove from prompts
    if image_id in prompts:
        del prompts[image_id]
        save_data(prompts)

def main():
    st.title("Training Data Review")

    # Initialize session state for current index and prompts if not exists
    if 'current_idx' not in st.session_state:
        st.session_state.current_idx = 0
    if 'prompts' not in st.session_state:
        st.session_state.prompts = load_data()

    # Use prompts from session state
    prompts = st.session_state.prompts
    image_ids = list(prompts.keys())

    if not image_ids:
        st.write("No images to review!")
        return

    # Keep index in bounds
    st.session_state.current_idx = st.session_state.current_idx % len(image_ids)
    current_id = image_ids[st.session_state.current_idx]

    # Display progress
    st.write(f"Reviewing {st.session_state.current_idx + 1} of {len(image_ids)}")

    # Display images side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Control Image")
        control_img = Image.open(os.path.join("training_data/control", current_id))
        st.image(control_img)

    with col2:
        st.write("Target Image")
        target_img = Image.open(os.path.join("training_data/target", current_id))
        st.image(target_img)

    # Edit prompt
    new_prompt = st.text_area("Prompt", value=prompts[current_id]["prompt"])
    if new_prompt != prompts[current_id]["prompt"]:
        prompts[current_id]["prompt"] = new_prompt
        prompts[current_id]["negative_prompt"] = prompts[current_id].get("negative_prompt", "")
        save_data(prompts)  # Save to disk
        st.session_state.prompts = prompts  # Update session state
        st.rerun()  # Force a refresh after saving

    # Navigation buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Back"):
            st.session_state.current_idx = (st.session_state.current_idx - 1) % len(image_ids)
            st.rerun()

    with col2:
        if st.button("Next"):
            st.session_state.current_idx = (st.session_state.current_idx + 1) % len(image_ids)
            st.rerun()

    with col3:
        if st.button("Delete", type="primary", use_container_width=True):
            delete_pair(current_id, prompts)
            if len(image_ids) > 1:
                st.session_state.current_idx = st.session_state.current_idx % (len(image_ids) - 1)
            else:
                st.session_state.current_idx = 0
            st.rerun()

if __name__ == "__main__":
    main() 