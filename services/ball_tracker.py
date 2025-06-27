import cv2
import numpy as np

class BallTracker:
    def __init__(self):
        self.dp = 1.2
        self.min_dist = 40
        self.param1 = 100
        self.param2 = 30
        self.min_radius = 5
        self.max_radius = 60

    def track(self, frame, debug=False):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )

        ball_position = None
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                ball_position = (x, y)
                # Optional drawing inside processing pipeline
                break

        if debug:
            cv2.imshow("Ball Detection Debug", blurred)
            cv2.waitKey(1)

        return ball_position
