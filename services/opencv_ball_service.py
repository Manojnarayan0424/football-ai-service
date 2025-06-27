import cv2
import numpy as np

class OpenCVBallService:
    def __init__(self):
        self.lower_color = (29, 86, 6)    # HSV lower bound for green ball
        self.upper_color = (64, 255, 255) # HSV upper bound for green ball

    def detect_ball(self, frame):
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        center = None
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if radius > 10:
                return {
                    "x": int(x),
                    "y": int(y),
                    "radius": int(radius),
                    "found": True
                }
        return {"found": False}

