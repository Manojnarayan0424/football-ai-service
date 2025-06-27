import os
import cv2
from services.movenet_service import MoveNetService
from services.clip_service import CLIPService
from services.ball_tracker import BallTracker
from utils.benchmark_logger import BenchmarkLogger
from utils.db import DBLogger

# ✅ Initialize services
pose = MoveNetService()
clip_service = CLIPService()
benchmark = BenchmarkLogger()
db_logger = DBLogger()
ball_tracker = BallTracker()

# ✅ Config
video_dir = "videos"
drill_count = 3
player_id = "player_101"  # You can make this dynamic later

def pose_accuracy(keypoints):
    return sum(1 for k in keypoints if k[2] > 0.3) / len(keypoints) * 100

# ✅ Loop through drills
for idx in range(drill_count):
    drill_id = f"Drill {idx + 1}"
    coach_path = os.path.join(video_dir, f"coach_drill{idx + 1}.MP4")
    student_path = os.path.join(video_dir, f"student_drill{idx + 1}.MP4")

    if not os.path.exists(coach_path) or not os.path.exists(student_path):
        print(f"❌ {drill_id} videos not found.")
        continue

    print(f"\n🎥 Comparing {drill_id}: Coach vs Student")

    cap1 = cv2.VideoCapture(coach_path)
    cap2 = cv2.VideoCapture(student_path)

    frame_count = 0
    coach_acc_list, student_acc_list, pose_sim_list = [], [], []

    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break

        frame_count += 1

        # ✅ Pose detection
        coach_kpts = pose.detect(frame1)
        student_kpts = pose.detect(frame2)

        # ✅ Ball tracking
        frame1, ball1 = ball_tracker.track_ball(frame1)
        frame2, ball2 = ball_tracker.track_ball(frame2)

        # ✅ CLIP similarity
        label1, sim1 = clip_service.compare_frame_to_prompts(
            frame1, ["dribble", "kick", "run", "stand"]
        )
        sim1_score = float(sim1.max().item())

        # ✅ Pose comparison
        pose_sim = pose.compare_keypoints(coach_kpts, student_kpts)
        acc1 = pose_accuracy(coach_kpts)
        acc2 = pose_accuracy(student_kpts)

        coach_acc_list.append(acc1)
        student_acc_list.append(acc2)
        pose_sim_list.append(pose_sim)

        # ✅ CSV + DB Logging
        benchmark.log(
            drill_id=drill_id,
            frame_num=frame_count,
            coach_acc=acc1,
            student_acc=acc2,
            clip_label=label1,
            clip_sim=sim1_score * 100,
            pose_sim=pose_sim,
            ball1=ball1,
            ball2=ball2
        )

        db_logger.insert_performance(
            drill_id=drill_id,
            frame_num=frame_count,
            coach_acc=acc1,
            student_acc=acc2,
            clip_label=label1,
            clip_sim=sim1_score,
            pose_sim=pose_sim,
            ball1=ball1,
            ball2=ball2
        )

        # ✅ Show video side-by-side
        if frame1 is not None and frame2 is not None:
            if frame1.shape[:2] != frame2.shape[:2]:
                frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
            if frame1.dtype == frame2.dtype:
                combined = cv2.hconcat([frame1, frame2])
                cv2.imshow("Coach vs Student", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap1.release()
    cap2.release()

    # ✅ Insert aggregated player performance for this drill
    print(f"\n🧪 Loop finished for {drill_id}")
    print(f"coach_acc_list: {len(coach_acc_list)} | student_acc_list: {len(student_acc_list)} | pose_sim_list: {len(pose_sim_list)}")

    if coach_acc_list and student_acc_list and pose_sim_list:
        coach_avg = sum(coach_acc_list) / len(coach_acc_list)
        student_avg = sum(student_acc_list) / len(student_acc_list)
        pose_avg = sum(pose_sim_list) / len(pose_sim_list)

        print(f"📥 Inserting summary for {player_id} - {drill_id}")
        db_logger.insert_player_performance(
            player_id=player_id,
            drill_id=drill_id,
            coach_avg=coach_avg,
            student_avg=student_avg,
            pose_sim_avg=pose_avg
        )
    else:
        print(f"❌ Skipped insert for {drill_id} due to missing data.")

# ✅ Generate report visuals
plot_paths = benchmark.plot_summary()

print(f"\n✅ Benchmark CSV saved at: {benchmark.filepath}")
print(f"📊 Accuracy Chart saved at: {plot_paths[0]}")
print(f"📉 Histogram saved at: {plot_paths[1]}")
print(f"📌 Confusion Matrix saved at: {plot_paths[2]}")

# ✅ Clean up
cv2.destroyAllWindows()
db_logger.close()
