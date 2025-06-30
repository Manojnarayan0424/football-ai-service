import cv2
import numpy as np
import tensorflow as tf

# MoveNet 17 body keypoints
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

class MoveNetService:
    def __init__(self, model_path):
        # Load TFLite model and allocate tensors
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_type = self.input_details[0]['dtype']

    def detect(self, frame):
        """
        Detect keypoints from input frame. Returns a dictionary
        with part names as keys and (x, y, confidence) as values.
        """
        h, w, _ = frame.shape
        input_shape = self.input_details[0]['shape']
        ih, iw = input_shape[1], input_shape[2]

        img = cv2.resize(frame, (iw, ih))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.input_type == np.uint8:
            input_tensor = np.expand_dims(img_rgb, axis=0).astype(np.uint8)
        else:
            input_tensor = np.expand_dims(img_rgb / 255.0, axis=0).astype(np.float32)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()

        keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]

        keypoints = {}
        for i, kp in enumerate(keypoints_with_scores):
            y, x, score = kp
            # Convert normalized coordinates to absolute pixel positions
            keypoints[KEYPOINT_NAMES[i]] = (int(x * w), int(y * h), score)

        return keypoints

    def draw_keypoints(self, frame, keypoints, threshold=0.3):
        """
        Draws visible keypoints on the frame with optional labels.
        """
        for part_name, (x, y, score) in keypoints.items():
            if score > threshold:
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                cv2.putText(frame, part_name, (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        return frame

    def compare_keypoints(self, keypoints1, keypoints2):
        """
        Compares two sets of keypoints and returns similarity %
        based on average distance of matched keypoints.
        """
        distances = []
        for part in KEYPOINT_NAMES:
            if part in keypoints1 and part in keypoints2:
                x1, y1, s1 = keypoints1[part]
                x2, y2, s2 = keypoints2[part]
                if s1 > 0.3 and s2 > 0.3:
                    dist = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
                    distances.append(dist)

        if not distances:
            return 0.0

        max_dim = max(640, 480)  # adjust if needed based on actual resolution
        similarity = 100 - (np.mean(distances) / max_dim * 100)
        return round(similarity, 2)
