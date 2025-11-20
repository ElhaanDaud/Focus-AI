def calculate_attention_score(
    eyes_score: float,
    posture_score: float,
    pitch: float,
    yaw: float,
    phone_detected: bool,
    person_count: int,
    face_detected: bool
) -> float:
    
    # Base score
    score = 1.0

    # RULE 1: If eyes not detected, severe penalty
    if not face_detected:
        return 0.0
    
    # RULE 2: Eyes closed/drowsy
    # eyes_score is 0.0 (closed) to 1.0 (open)
    if eyes_score < 0.4:
        score -= 0.4

    # RULE 3: Bad Posture
    # posture_score is 0.0 (slouch) to 1.0 (good)
    if posture_score < 0.5:
        score -= 0.2
    
    # RULE 4: Gaze (Pitch/Yaw)
    # Normal range approx -15 to +15 degrees. 
    # If looking far left/right/up/down, penalty.
    if abs(yaw) > 5 or abs(pitch) > 10:
        score -= 0.3
    
    # RULE 5: Phone Detected
    if phone_detected:
        score -= 0.5

    # RULE 6: Multiple People
    if person_count > 1:
        score -= 0.3

    # Clamp result
    return max(0.0, min(1.0, score))