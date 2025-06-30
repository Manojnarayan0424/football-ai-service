import torch
import clip
from PIL import Image
import numpy as np
import cv2

class CLIPService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def compare_frame_to_prompts(self, frame, prompts):
        """
        Given a video frame and a list of prompts (e.g., ["kick", "run"]),
        returns the most similar label and similarity scores.
        """
        # Convert BGR (OpenCV) to RGB and PIL format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)

        image_input = self.preprocess(image_pil).unsqueeze(0).to(self.device)
        text_tokens = clip.tokenize(prompts).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_tokens)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        best_idx = similarity.argmax().item()
        best_label = prompts[best_idx]
        return best_label, similarity.squeeze().cpu().numpy()
