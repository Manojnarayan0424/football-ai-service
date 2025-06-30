import torch
import clip
from PIL import Image
import numpy as np
import cv2

class CLIPService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        # ✅ Default action prompts — you can expand this list
        self.labels = ["kick", "dribble", "pass", "run", "stand", "jump"]
        self.text_tokens = clip.tokenize(self.labels).to(self.device)

    def compare_frame_to_prompts(self, frame, prompts):
        """
        Given a video frame and a list of prompts (e.g., ["kick", "run"]),
        returns the most similar label and similarity scores.
        """
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

    def compare_frames(self, frame1, frame2):
        """
        Compare two video frames semantically using CLIP.
        Returns predicted label1, label2, and semantic similarity (0-100 scale).
        """
        image1 = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
        image2 = Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))

        image1_input = self.preprocess(image1).unsqueeze(0).to(self.device)
        image2_input = self.preprocess(image2).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features1 = self.model.encode_image(image1_input)
            image_features2 = self.model.encode_image(image2_input)
            text_features = self.model.encode_text(self.text_tokens)

            image_features1 /= image_features1.norm(dim=-1, keepdim=True)
            image_features2 /= image_features2.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Semantic similarity between images
            clip_sim = float(torch.nn.functional.cosine_similarity(image_features1, image_features2).item())

            sim1 = (100.0 * image_features1 @ text_features.T).softmax(dim=-1)
            sim2 = (100.0 * image_features2 @ text_features.T).softmax(dim=-1)

        label1 = self.labels[sim1.argmax().item()]
        label2 = self.labels[sim2.argmax().item()]

        return label1, label2, round(clip_sim * 100, 2)
