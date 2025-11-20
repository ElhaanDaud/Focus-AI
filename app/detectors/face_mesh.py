import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

class FaceAnalyzer:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )

    def analyze(self, frame):
        """
        Returns: {
            "eyes_score": float (0.0 - 1.0),
            "pitch": float,
            "yaw": float,
            "detected": bool
        }
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return {"detected": False, "eyes_score": 0.0, "pitch": 0.0, "yaw": 0.0}

        lm = results.multi_face_landmarks[0].landmark
        
        # --- 1. Eye Ratio Logic (From your eye_detector.py) ---
        LEFT_EYE_TOP = 386
        LEFT_EYE_BOTTOM = 374
        RIGHT_EYE_TOP = 159
        RIGHT_EYE_BOTTOM = 145

        def eye_ratio(top, bottom):
            return abs(lm[top].y - lm[bottom].y)

        left_open = eye_ratio(LEFT_EYE_TOP, LEFT_EYE_BOTTOM)
        right_open = eye_ratio(RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM)
        raw_eye = (left_open + right_open) / 2
        # Heuristic: Normalize typical eye openness (approx 0.01-0.03 range) to 0-1
        eyes_score = max(0.0, min(raw_eye * 15, 1.0))

        # --- 2. Head Pose (Pitch/Yaw) Logic ---
        # 3D model points
        face_3d = []
        face_2d = []
        
        # Landmarks: Nose(1), Chin(152), Left Eye(33), Right Eye(266), Mouth Left(61), Mouth Right(291)
        points_idx = [1, 152, 33, 266, 61, 291]

        for idx in points_idx:
            x, y = int(lm[idx].x * w), int(lm[idx].y * h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm[idx].z])       

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        # Camera matrix approximation
        focal_length = 1 * w
        cam_matrix = np.array([[focal_length, 0, w / 2],
                               [0, focal_length, h / 2],
                               [0, 0, 1]])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        if success:
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            
            # angles[0] = pitch, angles[1] = yaw
            pitch = angles[0] * 360
            yaw = angles[1] * 360
        else:
            pitch = 0.0
            yaw = 0.0

        return {
            "detected": True,
            "eyes_score": eyes_score,
            "pitch": pitch,
            "yaw": yaw
        }