import cv2
import numpy as np
import os
import pandas as pd
from services.movenet_service import MoveNetService
from services.video_service import draw_keypoints
from services.clip_service import CLIPService
from services.ball_tracker import BallTracker
from utils.benchmark_logger import BenchmarkLogger

# ✅ Initialize services
movenet = MoveNetService("models/movenet_thunder_int8.tflite")
clip = CLIPService()
ball_tracker = BallTracker()

def process_drill(drill_id, coach_path, student_path):
    cap1 = cv2.VideoCapture(coach_path)
    cap2 = cv2.VideoCapture(student_path)

    logger = BenchmarkLogger(drill_id)
    frame_num = 0

    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # ✅ Detect keypoints
        coach_kps = movenet.detect(frame1)
        student_kps = movenet.detect(frame2)

        # ✅ Draw keypoints
        frame1 = draw_keypoints(frame1, coach_kps)
        frame2 = draw_keypoints(frame2, student_kps)

        # ✅ Pose similarity
        pose_sim = movenet.compare_keypoints(coach_kps, student_kps)

        # ✅ Ball tracking (fixed)
        frame1, ball1 = ball_tracker.track_ball(frame1)
        frame2, ball2 = ball_tracker.track_ball(frame2)

        # ✅ CLIP semantic similarity
        label1, label2, clip_sim = clip.compare_frames(frame1, frame2)

        # ✅ Store logs
        logger.log(frame_num, coach_kps, student_kps, pose_sim, ball1, ball2, label1, label2, clip_sim)

        frame_num += 1

    cap1.release()
    cap2.release()

    # ✅ Save outputs
    logger.save_overlay_video()
    logger.save_to_csv()
    logger.save_summary_charts()
