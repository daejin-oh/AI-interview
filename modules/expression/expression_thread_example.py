# modules/expression/expression_thread_example.py

import cv2
import threading
import queue
import mediapipe as mp

from modules.expression.emotion_recorg import emotion_detect
from modules.expression.emotion_stabilizer import emo_stabilizer
from modules.shared_flags import RUNNING

# 🔥 공용 카메라 프레임
from modules.camera.camera_manager import shared_frame_queue

# 결과 → main.py
expression_result_queue = queue.Queue(maxsize=5)

# MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)


# =====================================================
# 🙂 표정 분석 스레드 (카메라 공유 버전)
# =====================================================
def expression_worker(emotion_detector=None, padding=20):
    print("🙂 Expression Thread Started")

    while RUNNING:

        # 카메라 프레임이 아직 없으면 패스
        if shared_frame_queue.empty():
            continue

        # 공용 프레임 사용
        frame = shared_frame_queue.get()

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 얼굴 탐지 (MediaPipe)
        result = mp_face_detection.process(rgb)

        result_data = None

        if result.detections:
            det = result.detections[0]
            bbox = det.location_data.relative_bounding_box

            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)

            # 패딩 적용 + 이미지 범위 체크
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)

            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]

                # 감정 분석이 파일 기반이므로 이미지 저장
                cv2.imwrite("exp_tmp.jpg", crop)

                emo_raw = None
                if emotion_detector is not None:
                    # 🔥 감정 모델이 있을 때만 실행
                    try:
                        emo_raw = emotion_detect("exp_tmp.jpg", emotion_detector)
                    except Exception as e:
                        print(f"❌ emotion_detect error: {e}")
                        emo_raw = None

                if emo_raw:
                    emo_smooth = emo_stabilizer(emo_raw)

                    result_data = {
                        "raw": emo_raw["emotions"],
                        "dominant": emo_raw["dominant"],
                        "smooth": emo_smooth["smoothed"] if emo_smooth else emo_raw["emotions"],
                    }

        # 최신 데이터만 유지
        if expression_result_queue.full():
            try:
                expression_result_queue.get_nowait()
            except:
                pass

        expression_result_queue.put((frame, result_data))

    print("🙂 Expression Thread Stopped")


# =====================================================
# ▶️ 스레드 시작 함수
# =====================================================
def start_expression_thread(emotion_detector=None):
    t_ex = threading.Thread(
        target=expression_worker,
        args=(emotion_detector,),
        daemon=True
    )
    t_ex.start()
    print("🚀 expression_thread_example 실행됨! (Camera 공유 버전)")
    return t_ex


# =====================================================
# 🔍 단독 테스트용 (원하면 나중에 사용)
# =====================================================
if __name__ == "__main__":
    from modules.camera.camera_manager import start_camera_thread
    import modules.shared_flags as flags

    flags.RUNNING = True
    start_camera_thread()

    # 지금은 감정모델 없음 → None
    start_expression_thread(None)

    while True:
        if not expression_result_queue.empty():
            frame, emo = expression_result_queue.get()

            if emo:
                print("dominant:", emo["dominant"])

            cv2.imshow("Expression Thread Debug", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            flags.RUNNING = False
            break

    cv2.destroyAllWindows()
