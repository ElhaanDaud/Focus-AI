import base64
import json
import cv2
import numpy as np
import os 
import asyncio
from datetime import datetime, timezone
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
# Configuration
load_dotenv()
# Import our detectors
from app.detectors.face_mesh import FaceAnalyzer
from app.detectors.posture import PostureAnalyzer
from app.detectors.yolo_detector import YOLOAnalyzer
from app.utils.scoring import calculate_attention_score

app = FastAPI()

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MongoDB Config ---
# REPLACE WITH YOUR ACTUAL ATLAS URI
MONGO_URI = os.getenv("MONGO_URI") 
client = AsyncIOMotorClient(MONGO_URI)
db = client.focus_ai_db
collection = db.attention_logs

# --- Initialize Detectors ---
face_analyzer = FaceAnalyzer()
posture_analyzer = PostureAnalyzer()
yolo_analyzer = YOLOAnalyzer()

@app.get("/")
def root():
    return {"status": "Focus AI Backend Running"}

@app.websocket("/ws/{user_id}/{session_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, session_id: str):
    await websocket.accept()
    
    try:
        while True:
            # 1. Receive Frame (Base64)
            data = await websocket.receive_text()
            
            # Decode Base64 to OpenCV Image
            try:
                image_bytes = base64.b64decode(data.split(',')[1])
                np_arr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception:
                continue # Skip bad frames

            if frame is None: 
                continue

            # 2. Run Detectors (Parallel execution could be added, but sequential is safer for resource locks)
            # A. Face & Head Pose
            face_data = face_analyzer.analyze(frame) # {detected, eyes_score, pitch, yaw}
            
            # B. Posture
            posture_score = posture_analyzer.detect(frame)
            
            # C. YOLO (Phone & Person Count)
            # Resize for YOLO speed (optional, handled inside detector usually)
            yolo_data = yolo_analyzer.detect(frame) # {phone_detected, phone_boxes, person_count, inference_ms}

            # 3. Calculate Attention Score
            attention_score = calculate_attention_score(
                eyes_score=face_data["eyes_score"],
                posture_score=posture_score,
                pitch=face_data["pitch"],
                yaw=face_data["yaw"],
                phone_detected=yolo_data["phone_detected"],
                person_count=yolo_data["person_count"],
                face_detected=face_data["detected"]
            )

            # 4. Construct JSON Payload (Exact requirements)
            payload = {
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "attention": float(attention_score),
                "posture": float(posture_score),
                "eyes": float(face_data["eyes_score"]),
                "head_angle_pitch": float(face_data["pitch"]),
                "head_angle_yaw": float(face_data["yaw"]),
                "phone": 1 if yolo_data["phone_detected"] else 0,
                "yolo_phone_boxes": yolo_data["phone_boxes"],
                "yolo_person_count": yolo_data["person_count"],
                "yolo_inference_ms": float(yolo_data["inference_ms"])
            }

            # 5. Send to Frontend
            await websocket.send_json(payload)

            # 6. Store in MongoDB (Fire and forget / async)
            # Note: For high throughput, you might buffer these or use a queue.
            try:
                await collection.insert_one(payload.copy())
            except Exception as e:
                print(f"DB Error: {e}")

    except WebSocketDisconnect:
        print(f"Client {user_id} disconnected")
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()