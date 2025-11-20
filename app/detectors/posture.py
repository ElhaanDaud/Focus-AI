import cv2
import mediapipe as mp

# Wrapping your provided code into a class/function for cleaner import
mp_pose = mp.solutions.pose

class PostureAnalyzer:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False
        )

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        if not results.pose_landmarks:
            return 0.0

        lm = results.pose_landmarks.landmark

        # Your original logic
        left_sh = lm[11]
        right_sh = lm[12]
        nose = lm[1]
       
        shoulder_y_avg = (left_sh.y + right_sh.y) / 2
        vertical_drop = abs(nose.y - shoulder_y_avg)
        shoulder_width = abs(left_sh.x - right_sh.x) 
        
        if shoulder_width == 0: return 0.0

        normalized_drop = vertical_drop / shoulder_width
        
        # Tuning provided in original file
        raw_score = (normalized_drop - 0.425) * 5.0
        return max(0.0, min(1.0, raw_score))