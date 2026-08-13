import os
import json
import torch
import numpy as np
import cv2

MOTION_LIB_DIR = os.path.join(os.path.dirname(__file__), "motion_library")
METADATA_FILE = os.path.join(MOTION_LIB_DIR, "metadata.json")

class MiniMaxMotionLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_reference_id": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "frame_count")
    FUNCTION = "load_motion"
    CATEGORY = "MiniMax"

    def load_motion(self, motion_reference_id):
        if not os.path.exists(METADATA_FILE):
            print(f"[MiniMaxMotionLoader] metadata.json not found in {MOTION_LIB_DIR}")
            return (torch.zeros((1, 512, 512, 3)), 0)

        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if motion_reference_id not in metadata:
            print(f"[MiniMaxMotionLoader] motion_reference_id '{motion_reference_id}' not found in metadata.json")
            # fallback to a dummy sequence
            return (torch.zeros((16, 512, 512, 3), dtype=torch.float32), 16)

        filename = metadata[motion_reference_id].get("file", "")
        filepath = os.path.join(MOTION_LIB_DIR, filename)

        if not os.path.exists(filepath):
            print(f"[MiniMaxMotionLoader] Video file {filepath} not found. Creating dummy tensor for motion '{motion_reference_id}'.")
            # For testing without actual mp4s, we'll return a dummy batch of 16 frames
            dummy = torch.zeros((16, 512, 512, 3), dtype=torch.float32)
            return (dummy, 16)

        # Load real video using cv2
        cap = cv2.VideoCapture(filepath)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # OpenCV returns BGR, convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if not frames:
            return (torch.zeros((1, 512, 512, 3)), 0)

        # Convert to ComfyUI format (B, H, W, C) in [0, 1] range
        frames_np = np.array(frames).astype(np.float32) / 255.0
        out_tensor = torch.from_numpy(frames_np)

        return (out_tensor, len(frames))
