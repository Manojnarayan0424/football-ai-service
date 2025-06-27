import numpy as np
import cv2

class MoveNetService:
    def __init__(self, model_path):
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def detect_keypoints(self, frame):
        # Resize and normalize frame
        input_shape = self.input_details[0]['shape']
        h, w = input_shape[1], input_shape[2]
        resized = cv2.resize(frame, (w, h))

        if self.input_details[0]['dtype'] == np.uint8:
            input_tensor = np.expand_dims(resized, axis=0).astype(np.uint8)
        else:
            input_tensor = np.expand_dims(resized / 255.0, axis=0).astype(np.float32)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()

        # Output: [1, 1, 17, 3] → squeeze to [17, 3]
        keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])
        keypoints = keypoints_with_scores[0][0]  # shape: [17, 3]

        return keypoints.tolist()  # [ [y, x, score], ..., ]
