import cv2

class VideoService:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"❌ Cannot open video file: {video_path}")

    def get_next_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def rewind(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def get_frame_count(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_resolution(self):
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def release(self):
        self.cap.release()


# ✅ Draw keypoints on a frame
def draw_keypoints(frame, keypoints, threshold=0.3):
    h, w, _ = frame.shape
    for kp in keypoints:
        y, x, confidence = kp
        if confidence > threshold:
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
    return frame
