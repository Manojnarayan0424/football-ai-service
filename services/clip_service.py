import torch
import clip
from PIL import Image
import numpy as np
import cv2

class CLIPService:
    def __init__(self, device="cpu"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.labels = ["kicking", "dribbling", "standing", "running", "jumping"]

        # Preprocess labels
        with torch.no_grad():
            self.label_tokens = clip.tokenize(self.labels).to(self.device)
            self.label_features = self.model.encode_text(self.label_tokens)
            self.label_features /= self.label_features.norm(dim=-1, keepdim=True)

    def preprocess_frame(self, frame):
        # Convert frame (OpenCV) to PIL format and apply CLIP preprocessing
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        return self.preprocess(pil_img).unsqueeze(0).to(self.device)

    def compare_frames(self, frame1, frame2):
        with torch.no_grad():
            f1 = self.preprocess_frame(frame1)
            f2 = self.preprocess_frame(frame2)

            feat1 = self.model.encode_image(f1)
            feat2 = self.model.encode_image(f2)

            feat1 /= feat1.norm(dim=-1, keepdim=True)
            feat2 /= feat2.norm(dim=-1, keepdim=True)

            # Cosine similarity
            similarity = (feat1 @ feat2.T).item()

            # Predict action label
            label_logits1 = feat1 @ self.label_features.T
            label_logits2 = feat2 @ self.label_features.T

            label1 = self.labels[label_logits1.argmax().item()]
            label2 = self.labels[label_logits2.argmax().item()]

        return label1, label2, round(similarity, 3)
