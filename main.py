# main.py
import time
import cv2
import numpy as np
import os

# ===============================
# 🔥 공유 RUNNING 플래그
# ===============================
from modules.shared_flags import RUNNING

# ===============================
# 🔥 단일 카메라 스레드
# ===============================
from modules.camera.camera_manager import start_camera_thread

# ===============================
# 🔥 모듈별 스레드 & 큐
# ===============================
from modules.pose.pose_thread_example import start_pose_thread, result_queue as pose_result_queue
from modules.gaze.gaze_thread_example import start_gaze_thread, gaze_result_queue
from modules.expression.expression_thread_example import start_expression_thread, expression_result_queue
from modules.hands.hand_thread_example import start_hands_thread, hands_result_queue
from modules.voice.voice_thread_example import start_voice_thread, voice_result_queue


# 표정 모듈은 emotion_detector 필요 없음 → None 사용
# GazeTracker를 표정에 넘기면 detect_faces 없어 오류남 (사용 금지)


# ===============================
# 최신 값만 가져오기
# ===============================
def drain_queue(q):
    latest = None
    while not q.empty():
        latest = q.get()
    return latest


# ===============================
# 메인 실행부
# ===============================
def main():
    # 🔥 표정모듈은 감정모델이 없으므로 None 전달
    emotion_detector = None

    # 🔥 반드시 이 순서로 시작
    start_camera_thread()  # 1개 카메라만 공유
    start_pose_thread()
    gaze_thread = start_gaze_thread()
    start_expression_thread(emotion_detector)  # ← 수정됨
    start_hands_thread()
    start_voice_thread()

    print("\n🚀 AI Mock Interview — Main Started (q 또는 X로 종료)\n")

    latest_pose = None
    latest_gaze = None
    latest_expr = None
    latest_hands = None
    latest_voice = None

    window_name = "AI Mock Interview - Dashboard"

    while True:

        # ===== 최신 데이터 수신 =====
        pose_data = drain_queue(pose_result_queue)
        gaze_data = drain_queue(gaze_result_queue)
        expr_data = drain_queue(expression_result_queue)
        hands_data = drain_queue(hands_result_queue)
        voice_data = drain_queue(voice_result_queue)

        if pose_data is not None:
            latest_pose = pose_data
        if gaze_data is not None:
            latest_gaze = gaze_data
        if expr_data is not None:
            latest_expr = expr_data
        if hands_data is not None:
            latest_hands = hands_data
        if voice_data is not None:
            latest_voice = voice_data

        # ============================
        # 대시보드 화면
        # ============================
        dashboard = np.zeros((800, 1200, 3), dtype=np.uint8)

        cv2.putText(
            dashboard,
            "AI Mock Interview Dashboard",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        # ========== 자세 =============
        if latest_pose is not None:
            frame, motion, _ = latest_pose
            f = cv2.resize(frame, (350, 250))
            dashboard[80:330, 20:370] = f
            cv2.putText(dashboard, f"Movement: {motion:.2f}",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ========== 시선 =============
        if latest_gaze is not None:
            frame, g = latest_gaze
            f = cv2.resize(frame, (350, 250))
            dashboard[80:330, 400:750] = f
            cv2.putText(dashboard, f"Gaze: {g['left_right']} / {g['up_down']}",
                        (400, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ========== 손 =============
        if isinstance(latest_hands, np.ndarray):
            f = cv2.resize(latest_hands, (350, 250))
            dashboard[350:600, 400:750] = f

        # ========== 표정 =============
        if latest_expr is not None and isinstance(latest_expr[1], dict):
            emo = latest_expr[1]
            cv2.putText(dashboard, f"Expression: {emo['dominant']}",
                        (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # ========== 음성 =============
        if latest_voice is not None:
            text = latest_voice["text"]
            safe_text = text[:50] if text else "(음성 인식 실패)"

            cv2.putText(dashboard, f"Voice: {safe_text}",
                        (20, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ============================
        # 창 표시
        # ============================
        try:
            cv2.imshow(window_name, dashboard)
        except cv2.error:
            print("🔥 imshow error — window closed")
            break

        # X 닫기 확인
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("🔥 Window closed by user.")
            break

        # q 종료
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            from modules.gaze.gaze_thread_example import request_gaze_calibration
            request_gaze_calibration()
            print("✅ 'c' pressed → Gaze center calibration requested")

        if key == ord('q'):
            print("🔚 'q' pressed. Exiting.")
            breakgit remote -v

    # ============================
    # 🔥 전체 스레드 종료
    # ============================
    import modules.shared_flags as flags
    flags.RUNNING = False
    gaze_thread.join()
    cv2.destroyAllWindows()
    print("🧹 Threads stopped.")

    return


# ============================
# 실행 시작
# ============================
if __name__ == "__main__":
    main()
