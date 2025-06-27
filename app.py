from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import shutil
from utils.db import DBLogger
from drill_evaluator import process_drill  # ✅ Pose + ball tracking

app = FastAPI()

UPLOAD_DIR = "videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Serve static files and templates
app.mount("/static", StaticFiles(directory="results"), name="static")
templates = Jinja2Templates(directory="templates")


# ✅ ROOT route
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <head><title>Football AI Service</title></head>
        <body style="font-family: sans-serif; text-align: center;">
            <h1>🏆 Welcome to Football AI Backend</h1>
            <p>Access <a href='/docs'>API Documentation</a></p>
            <p>View <a href='/v1/performance/report'>Drill Performance Report</a></p>
        </body>
    </html>
    """


# 🎥 Upload Drill Videos
@app.post("/v1/upload-drill/")
async def upload_both_videos(
    drill_id: int = Form(...),
    coach_video: UploadFile = File(...),
    student_video: UploadFile = File(...)
):
    coach_filename = f"coach_drill{drill_id}.MP4"
    student_filename = f"student_drill{drill_id}.MP4"

    coach_path = os.path.join(UPLOAD_DIR, coach_filename)
    student_path = os.path.join(UPLOAD_DIR, student_filename)

    with open(coach_path, "wb") as f:
        shutil.copyfileobj(coach_video.file, f)

    with open(student_path, "wb") as f:
        shutil.copyfileobj(student_video.file, f)

    # ✅ Run evaluator
    process_drill(drill_id=f"Drill {drill_id}", coach_path=coach_path, student_path=student_path)

    return JSONResponse({
        "message": f"✅ Drill {drill_id} uploaded and processed.",
        "coach_video": coach_path,
        "student_video": student_path,
        "result_video": f"results/overlay_drill_{drill_id}.mp4"
    })


# 📊 Performance Report Page (Visual HTML Report)
@app.get("/v1/performance/report", response_class=HTMLResponse)
def generate_report(request: Request):
    db = DBLogger()
    db.cursor.execute("""
        SELECT drill_id, frame_num, coach_acc, student_acc, pose_sim, ball1, ball2, clip_label, clip_sim
        FROM performance ORDER BY frame_num ASC
    """)
    rows = db.cursor.fetchall()
    db.close()

    total = len(rows)
    coach_avg = round(sum(r[2] for r in rows) / total, 2) if total else 0
    student_avg = round(sum(r[3] for r in rows) / total, 2) if total else 0
    sim_avg = round(sum(r[4] for r in rows) / total, 2) if total else 0

    return templates.TemplateResponse("report.html", {
        "request": request,
        "rows": rows,
        "total": total,
        "coach_avg": coach_avg,
        "student_avg": student_avg,
        "sim_avg": sim_avg
    })


# 📎 Downloadable Report (e.g. benchmark_plot.png)
@app.get("/v1/performance/report/image", response_class=FileResponse)
def download_report_image():
    report_path = "results/benchmark_plot.png"
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report image not found.")

    return FileResponse(
        path=report_path,
        media_type="image/png",
        filename="benchmark_report.png"
    )


# 🧠 Save Performance Summary per Drill
@app.post("/v1/player/performance")
def store_player_performance(
    player_id: str = Form(...),
    drill_id: str = Form(...)
):
    db = DBLogger()
    db.cursor.execute("""
        SELECT coach_acc, student_acc, pose_sim
        FROM performance
        WHERE drill_id = %s
    """, (drill_id,))
    rows = db.cursor.fetchall()

    if not rows:
        db.close()
        return JSONResponse(
            status_code=404,
            content={"error": f"No frame data found for {drill_id}"}
        )

    coach_avg = round(sum(r[0] for r in rows) / len(rows), 2)
    student_avg = round(sum(r[1] for r in rows) / len(rows), 2)
    pose_avg = round(sum(r[2] for r in rows) / len(rows), 2)

    db.insert_player_performance(player_id, drill_id, coach_avg, student_avg, pose_avg)
    db.close()

    return {
        "message": "✅ Player performance summary saved.",
        "player_id": player_id,
        "drill_id": drill_id,
        "coach_avg": coach_avg,
        "student_avg": student_avg,
        "pose_avg": pose_avg
    }


# 🔍 Get Performance by Player ID
@app.get("/v1/player/{player_id}/performance")
def get_player_performance(player_id: str):
    db = DBLogger()
    try:
        db.cursor.execute("""
            SELECT drill_id, average_coach_accuracy, average_student_accuracy, average_pose_similarity
            FROM player_performance
            WHERE player_id = %s
            ORDER BY id DESC
        """, (player_id,))
        rows = db.cursor.fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail="No performance data found for this player.")

        return {
            "player_id": player_id,
            "drills": [
                {
                    "drill_id": row[0],
                    "coach_avg": round(row[1], 2),
                    "student_avg": round(row[2], 2),
                    "pose_avg": round(row[3], 2)
                } for row in rows
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
