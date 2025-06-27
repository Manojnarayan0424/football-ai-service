import os
import csv
import cv2
import matplotlib.pyplot as plt
import pandas as pd

class BenchmarkLogger:
    def __init__(self, drill_id):
        self.drill_id = drill_id
        self.rows = []
        self.overlay_writer = None
        self.overlay_path = f"results/overlay_drill_{drill_id}.mp4"
        os.makedirs("results", exist_ok=True)

    def log(self, frame_num, coach_kps, student_kps, pose_sim, ball1, ball2, label1, label2, clip_sim):
        coach_acc = self._calculate_accuracy(coach_kps)
        student_acc = self._calculate_accuracy(student_kps)

        self.rows.append({
            "drill_id": self.drill_id,
            "frame_num": frame_num,
            "coach_acc": coach_acc,
            "student_acc": student_acc,
            "pose_sim": pose_sim,
            "ball1": str(ball1),
            "ball2": str(ball2),
            "clip_label": f"{label1} vs {label2}",
            "clip_sim": clip_sim
        })

    def _calculate_accuracy(self, keypoints, threshold=0.3):
        if not keypoints:
            return 0.0
        visible = [kp for kp in keypoints if kp[2] > threshold]
        return round(len(visible) / len(keypoints) * 100, 2)

    def save_to_csv(self):
        if not self.rows:
            print("⚠️ No data to save.")
            return
        csv_path = self._csv_path()
        keys = self.rows[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"✅ CSV saved: {csv_path}")

    def save_overlay_video(self):
        csv_path = self._csv_path()
        if not os.path.exists(csv_path):
            # Auto-save CSV if not already done
            print("ℹ️ CSV not found. Attempting to save it first...")
            self.save_to_csv()
            if not os.path.exists(csv_path):
                print(f"❌ Still missing: {csv_path}")
                return

        df = pd.read_csv(csv_path)

        # Construct video path using drill number
        drill_num = self.drill_id.split()[-1]
        video_path = f"videos/student_drill{drill_num}.MP4"
        if not os.path.exists(video_path):
            print(f"❌ Missing student video: {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Failed to open video: {video_path}")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(self.overlay_path, fourcc, fps, (width, height))

        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or idx >= len(df):
                break

            row = df.iloc[idx]
            text = f"Coach: {row['coach_acc']}%  Student: {row['student_acc']}%  Sim: {row['pose_sim']:.2f}  Action: {row['clip_label']}"

            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
            out.write(frame)
            idx += 1

        cap.release()
        out.release()
        print(f"🎥 Overlay video saved: {self.overlay_path}")

    def save_summary_charts(self):
        if not self.rows:
            print("⚠️ No rows available to generate charts.")
            return

        df = pd.DataFrame(self.rows)

        # Accuracy chart
        plt.figure()
        plt.plot(df["frame_num"], df["coach_acc"], label="Coach Accuracy")
        plt.plot(df["frame_num"], df["student_acc"], label="Student Accuracy")
        plt.xlabel("Frame")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.title("Pose Accuracy Over Time")
        plt.savefig("results/pose_accuracy_hist.png")
        plt.close()

        # Confusion chart
        plt.figure()
        label_diffs = df["clip_label"].apply(lambda x: x.split(" vs "))
        mismatch = [1 if a != b else 0 for a, b in label_diffs]
        plt.plot(df["frame_num"], mismatch, label="Action Mismatch (1 = mismatch)")
        plt.xlabel("Frame")
        plt.ylabel("Mismatch")
        plt.title("Semantic Action Mismatches")
        plt.savefig("results/pose_accuracy_confusion.png")
        plt.close()

        # CLIP similarity
        plt.figure()
        plt.plot(df["frame_num"], df["clip_sim"], label="CLIP Similarity")
        plt.xlabel("Frame")
        plt.ylabel("Cosine Similarity")
        plt.title("CLIP Semantic Similarity Over Time")
        plt.savefig("results/benchmark_plot.png")
        plt.close()

        print("📊 Charts saved in 'results/'.")

    def _csv_path(self):
        return f"results/log_drill_{self.drill_id}.csv"
