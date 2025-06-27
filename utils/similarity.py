import numpy as np

def calculate_pose_similarity(coach_kps, student_kps):
    def to_array(kps):
        return np.array([(x, y) for (x, y, c) in kps if c > 0.5])

    coach_arr = to_array(coach_kps)
    student_arr = to_array(student_kps)

    # Handle empty or mismatched keypoints
    if len(coach_arr) == 0 or len(student_arr) == 0 or len(coach_arr) != len(student_arr):
        return 0.0, 0.0, 0.0

    # Coach and student accuracy: confidence-averaged keypoints present
    coach_acc = sum(1 for (_, _, c) in coach_kps if c > 0.5) / len(coach_kps) * 100
    student_acc = sum(1 for (_, _, c) in student_kps if c > 0.5) / len(student_kps) * 100

    # Pose similarity: mean Euclidean distance similarity
    diffs = np.linalg.norm(coach_arr - student_arr, axis=1)
    similarity = max(0, 100 - np.mean(diffs))  # Scale to 0-100

    return coach_acc, student_acc, similarity
