import psycopg2
import os
from dotenv import load_dotenv

# ✅ Load environment variables from .env file
load_dotenv()

class DBLogger:
    def __init__(self):
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("❌ DATABASE_URL is not set in the environment.")

        try:
            self.conn = psycopg2.connect(db_url)
            self.cursor = self.conn.cursor()
            print("✅ Connected to PostgreSQL")

            # Create required tables
            self.create_table()
            self.create_player_performance_table()
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            raise

    def create_table(self):
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance (
                    id SERIAL PRIMARY KEY,
                    drill_id TEXT,
                    frame_num INT,
                    coach_acc FLOAT,
                    student_acc FLOAT,
                    clip_label TEXT,
                    clip_sim FLOAT,
                    pose_sim FLOAT,
                    ball1 TEXT,
                    ball2 TEXT
                )
            """)
            self.conn.commit()
        except Exception as e:
            print(f"❌ Failed to create 'performance' table: {e}")

    def create_player_performance_table(self):
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS player_performance (
                    id SERIAL PRIMARY KEY,
                    player_id TEXT,
                    drill_id TEXT,
                    average_coach_accuracy FLOAT,
                    average_student_accuracy FLOAT,
                    average_pose_similarity FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.commit()
        except Exception as e:
            print(f"❌ Failed to create 'player_performance' table: {e}")

    def insert_performance(self, drill_id, frame_num, coach_acc, student_acc,
                           clip_label, clip_sim, pose_sim, ball1, ball2):
        try:
            self.cursor.execute("""
                INSERT INTO performance (
                    drill_id, frame_num, coach_acc, student_acc,
                    clip_label, clip_sim, pose_sim, ball1, ball2
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                str(drill_id),
                int(frame_num),
                float(coach_acc),
                float(student_acc),
                str(clip_label),
                float(clip_sim),
                float(pose_sim),
                str(ball1),
                str(ball2)
            ))
            self.conn.commit()
        except Exception as e:
            print(f"❌ Failed to insert frame-level performance: {e}")
            self.conn.rollback()

    def insert_player_performance(self, player_id, drill_id, coach_avg, student_avg, pose_sim_avg):
        try:
            # ✅ Check if this player+drill already exists
            self.cursor.execute("""
                SELECT 1 FROM player_performance
                WHERE player_id = %s AND drill_id = %s
            """, (player_id, drill_id))
            if self.cursor.fetchone():
                print(f"⚠️ Duplicate found: skipping insert for {player_id} - {drill_id}")
                return

            # ✅ Proceed with insert
            print(f"📝 INSERTING: player_id='{player_id}', drill_id='{drill_id}', "
                  f"coach_avg={coach_avg}, student_avg={student_avg}, pose_sim_avg={pose_sim_avg}")

            self.cursor.execute("""
                INSERT INTO player_performance (
                    player_id, drill_id, average_coach_accuracy,
                    average_student_accuracy, average_pose_similarity
                ) VALUES (%s, %s, %s, %s, %s)
            """, (
                str(player_id),
                str(drill_id),
                float(coach_avg),
                float(student_avg),
                float(pose_sim_avg)
            ))
            self.conn.commit()
            print("✅ COMMIT OK ✅")

            # Debug: row count
            self.cursor.execute("SELECT COUNT(*) FROM player_performance")
            row_count = self.cursor.fetchone()[0]
            print(f"📊 Rows now in player_performance: {row_count}")

        except Exception as e:
            print(f"❌ Failed to insert player summary: {e}")
            self.conn.rollback()

    def close(self):
        self.cursor.close()
        self.conn.close()
        print("🔒 PostgreSQL connection closed")
