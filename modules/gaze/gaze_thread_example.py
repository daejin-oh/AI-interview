# modules/gaze/gaze_thread_example.py

import cv2
import threading
import queue
import time
from modules.gaze.gaze_module import GazeTracker
import modules.shared_flags as flags

# 🔥 camera_manager에서 공통 프레임 가져오기
from modules.camera.camera_manager import shared_frame_queue

# 분석 결과 → main.py
gaze_result_queue = queue.Queue(maxsize=5)

calibrate_event = threading.Event()  # 보정 요청 이벤트

# ✅ 종료 시 마지막 값 저장용 (원하면 main에서 참조 가능)
last_center_ratio = 0.0
last_center_time = 0.0
last_total_time = 0.0


def request_gaze_calibration():
    """main에서 'c' 눌렀을 때 호출"""
    calibrate_event.set()


def score_center_ratio(center_ratio: float) -> int:
    """
    80% 이상 → 100점
    30% 이하 → 0점
    중간은 선형 증가
    """
    if center_ratio <= 30:
        return 0
    elif center_ratio >= 80:
        return 100
    else:
        return int(round((center_ratio - 30) / 50 * 100))

def score_avg_deviation_time(avg_deviation_time: float) -> int:
    """
    평균 이탈시간 점수화
    - 1초 이하 → 100점
    - 3초 이상 → 0점
    - 그 사이는 선형 감소
    """
    if avg_deviation_time <= 1.0:
        return 100
    elif avg_deviation_time >= 3.0:
        return 0
    else:
        score = (3.0 - avg_deviation_time) / 2.0 * 100
        return int(round(score))


def gaze_worker():
    global last_center_ratio, last_center_time, last_total_time

    tracker = GazeTracker()
    print("👁 Gaze Thread Started")

    # ✅ "c키 이후"에만 측정 시작 (원치 않으면 True로 바꾸면 됨)
    measuring_started = False

    # ✅ 깜빡임 제외 시간 기반 누적
    total_gaze_time = 0.0
    center_gaze_time = 0.0

    last_ts = time.perf_counter()

    deviation_started = False
    deviation_start_ts = None
    deviation_durations = []

    # ✅ 이탈 방향 시간 누적(깜빡임 제외, measuring_started 이후)
    off_center_time = 0.0

    lr_off_time = 0.0
    ud_off_time = 0.0
    diag_off_time = 0.0

    left_time = 0.0
    right_time = 0.0
    up_time = 0.0
    down_time = 0.0

    while flags.RUNNING:

        if shared_frame_queue.empty():
            continue

        frame = shared_frame_queue.get()

        # 'c' 보정 요청이 있으면 calibrate + 리셋 + 측정 시작
        if calibrate_event.is_set():
            try:
                tracker.calibrate()
            except Exception:
                pass

            # ✅ 전체/정면 누적 리셋
            total_gaze_time = 0.0
            center_gaze_time = 0.0
            measuring_started = True
            last_ts = time.perf_counter()

            # ✅ 이탈 구간 리셋
            deviation_started = False
            deviation_start_ts = None
            deviation_durations = []

            # ✅ 이탈 방향 누적 리셋
            off_center_time = 0.0
            lr_off_time = 0.0
            ud_off_time = 0.0
            diag_off_time = 0.0
            left_time = 0.0
            right_time = 0.0
            up_time = 0.0
            down_time = 0.0

            calibrate_event.clear()
            print("🎯 [GAZE] Calibrated & measurement reset/start")

        processed = tracker.process_frame(frame)

        # ✅ 시간 누적(프레임 처리 기준)
        now = time.perf_counter()
        dt = now - last_ts
        last_ts = now

        # dt가 너무 작거나 이상하면 방어
        if dt <= 0 or dt > 1.0:
            dt = 0.0

        # =========================
        # 정면 판정 (대소문자 안전)
        # =========================
        is_center = (
                str(tracker.gaze_direction_x).upper() == "CENTER"
                and str(tracker.gaze_direction_y).upper() == "CENTER"
        )

        # =========================
        # 정면 유지 시간 / 이탈 평균시간 계산
        # (깜빡임 제외)
        # =========================
        if measuring_started and dt > 0 and not tracker.is_blinking:

            # 🔹 전체 측정 시간
            total_gaze_time += dt

            # 🔹 정면 유지 시간
            if is_center:
                center_gaze_time += dt

            # ✅ 이탈 방향 시간 누적(정면이 아닐 때만)
            gx = str(tracker.gaze_direction_x).upper()
            gy = str(tracker.gaze_direction_y).upper()

            # 🔹 이탈 평균시간 계산
            if not is_center:
                # CENTER → OFF (이탈 시작)
                if not deviation_started:
                    deviation_started = True
                    deviation_start_ts = time.perf_counter()
                off_center_time += dt

                x_off = (gx != "CENTER")
                y_off = (gy != "CENTER")

                # 상하냐/좌우냐/대각이냐(서로 겹치지 않게 분리)
                if x_off and not y_off:
                    lr_off_time += dt
                elif y_off and not x_off:
                    ud_off_time += dt
                elif x_off and y_off:
                    diag_off_time += dt

                # 상세 방향(좌/우/상/하) 시간 누적
                if gx == "LEFT":
                    left_time += dt
                elif gx == "RIGHT":
                    right_time += dt

                if gy == "UP":
                    up_time += dt
                elif gy == "DOWN":
                    down_time += dt
            else:
                # OFF → CENTER (이탈 종료)
                if deviation_started and deviation_start_ts is not None:
                    dur = time.perf_counter() - deviation_start_ts
                    if 0.05 <= dur <= 10.0:
                        deviation_durations.append(dur)
                    deviation_started = False
                    deviation_start_ts = None

        # =========================
        # 평균 이탈시간 + 점수 계산
        # =========================
        avg_deviation_time = (
            sum(deviation_durations) / len(deviation_durations)
            if deviation_durations else 0.0
        )

        avg_deviation_score = score_avg_deviation_time(avg_deviation_time)

        # =========================
        # 최대 이탈시간 계산
        # =========================
        max_deviation_time = max(deviation_durations) if deviation_durations else 0.0


        center_ratio = (center_gaze_time / total_gaze_time * 100.0) if total_gaze_time > 0 else 0.0
        center_score = score_center_ratio(center_ratio)

        deviation_count = len(deviation_durations)

        # ✅ 이탈 방향 비율 계산(이탈 시간 기준)
        if off_center_time > 0:
            lr_ratio = lr_off_time / off_center_time
            ud_ratio = ud_off_time / off_center_time
            diag_ratio = diag_off_time / off_center_time

            left_ratio = left_time / off_center_time
            right_ratio = right_time / off_center_time
            up_ratio = up_time / off_center_time
            down_ratio = down_time / off_center_time
        else:
            lr_ratio = ud_ratio = diag_ratio = 0.0
            left_ratio = right_ratio = up_ratio = down_ratio = 0.0

        # (선택) 좌/우 밸런스(좌우만 놓고 봤을 때)
        lr_total = left_time + right_time
        if lr_total > 0:
            left_ratio_lr = left_time / lr_total
            right_ratio_lr = right_time / lr_total
        else:
            left_ratio_lr = right_ratio_lr = 0.0

        # (선택) 상/하 밸런스(상하만 놓고 봤을 때)
        ud_total = up_time + down_time
        if ud_total > 0:
            up_ratio_ud = up_time / ud_total
            down_ratio_ud = down_time / ud_total
        else:
            up_ratio_ud = down_ratio_ud = 0.0

        result = {
            "left_right": tracker.gaze_direction_x,
            "up_down": tracker.gaze_direction_y,
            "is_blinking": tracker.is_blinking,
            "ear": tracker.current_avg_ear,

            # ✅ 정면유지비율 + 점수
            "measuring": measuring_started,
            "center_ratio": center_ratio,
            "center_score": center_score,
            "center_time": center_gaze_time,
            "total_time": total_gaze_time,

            # 🔽 이탈 평균시간 결과
            "avg_deviation_time": avg_deviation_time,
            "deviation_count": deviation_count,
            "avg_deviation_score": avg_deviation_score,
            "max_deviation_time": max_deviation_time,

            # 🔽 이탈 방향 비율(이탈 시간 기준)
            "off_center_time": off_center_time,

            "lr_ratio": lr_ratio,
            "ud_ratio": ud_ratio,
            "diag_ratio": diag_ratio,

            "left_ratio": left_ratio,
            "right_ratio": right_ratio,
            "up_ratio": up_ratio,
            "down_ratio": down_ratio,

            # (선택) 밸런스
            "left_ratio_lr": left_ratio_lr,
            "right_ratio_lr": right_ratio_lr,
            "up_ratio_ud": up_ratio_ud,
            "down_ratio_ud": down_ratio_ud,
        }

        # 최신 데이터만 유지
        if gaze_result_queue.full():
            try:
                gaze_result_queue.get_nowait()
            except:
                pass

        gaze_result_queue.put((processed, result))

    # ============================
    # 🔚 종료 직전: 진행 중 이탈 마감
    # ============================
    if deviation_started and deviation_start_ts is not None:
        dur = time.perf_counter() - deviation_start_ts
        if 0.05 <= dur <= 10.0:
            deviation_durations.append(dur)
        deviation_started = False
        deviation_start_ts = None

    # ============================
    # 📊 종료 시 이탈 시선 점수 출력
    # ============================
    final_avg_deviation_time = (
        sum(deviation_durations) / len(deviation_durations)
        if deviation_durations else 0.0
    )
    final_deviation_count = len(deviation_durations)
    final_max_deviation_time = max(deviation_durations) if deviation_durations else 0.0
    final_avg_deviation_score = score_avg_deviation_time(final_avg_deviation_time)

    if off_center_time > 0:
        print(
            f"📌 [GAZE] 이탈 방향 비율: "
            f"좌우 {lr_off_time / off_center_time * 100:.1f}% / "
            f"위아래 {ud_off_time / off_center_time * 100:.1f}% / "
            f"상하좌우 {diag_off_time / off_center_time * 100:.1f}% | "
            f"좌 {left_time / off_center_time * 100:.1f}% 우 {right_time / off_center_time * 100:.1f}% "
            f"위 {up_time / off_center_time * 100:.1f}% 아래 {down_time / off_center_time * 100:.1f}%"
        )
    else:
        print("📌 [GAZE] 이탈 방향 비율: 측정된 이탈 없음")

    print(
        f"📌 [GAZE] 이탈 시선 점수: {final_avg_deviation_score}점 "
        f"(평균 {final_avg_deviation_time:.2f}s / "
        f"횟수 {final_deviation_count} / "
        f"최대 {final_max_deviation_time:.2f}s)"
    )

    # 종료 시 최종 계산
    final_center_ratio = (center_gaze_time / total_gaze_time * 100.0) if total_gaze_time > 0 else 0.0
    final_center_score = score_center_ratio(final_center_ratio)

    # 외부 참조용 저장
    last_center_ratio = final_center_ratio
    last_center_time = center_gaze_time
    last_total_time = total_gaze_time

    # 출력
    print(f"📌 [GAZE] 정면 시선 점수: {final_center_score}점 (정면 유지 {final_center_ratio:.1f}%)\n")


def start_gaze_thread():
    t_gaze = threading.Thread(target=gaze_worker, daemon=True)
    t_gaze.start()
    print("🚀 gaze_thread_example 실행됨! (Camera 공유 버전)")
    return t_gaze


if __name__ == "__main__":
    from modules.camera.camera_manager import start_camera_thread

    start_camera_thread()
    flags.RUNNING = True

    start_gaze_thread()

    while True:
        if not gaze_result_queue.empty():
            frame, data = gaze_result_queue.get()

            print(f"시선: {data['left_right']}, {data['up_down']} / "
                  f"깜빡임: {data['is_blinking']} / EAR={data['ear']:.3f} / "
                  f"정면유지: {data.get('center_ratio', 0):.1f}% / 점수: {data.get('center_score', 0)}")

            cv2.imshow("Gaze Debug", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            request_gaze_calibration()
        if key == ord('q'):
            break

    flags.RUNNING = False
    cv2.destroyAllWindows()
