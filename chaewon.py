import cv2
import numpy as np
import time
import threading
import os
import json
import zmq

from flask import Flask, Response, render_template_string
from pinkylib import Camera, Motor
from ultralytics import YOLO


# ============================================================
# Raspberry Pi / Pinky Pro
# ============================================================
app = Flask(__name__)

motor = Motor()
camera = Camera()

motor.enable_motor()
camera.start()


# ============================================================
# Network
# ============================================================
PORT = 5000

# 노트북 ZMQ 서버 IP
LAPTOP_ZMQ_IP = "172.20.10.14" # 노트북 아이피
LAPTOP_ZMQ_PORT = 6000


# ============================================================
# 주행 설정
# ============================================================
BASE_SPEED = 20
KP = 0.12
MAX_SPEED = 32

# 커브 구간 감속 및 회전 강화 설정
CURVE_SLOWDOWN = 0.05   # 중심선 오차가 커질수록 기본 속도 감소
MIN_CURVE_SPEED = 12    # 커브 최소 속도
INNER_MIN_SPEED = -10   # 안쪽 바퀴 역회전 허용 속도

SEARCH_SPEED = 16
SEARCH_INNER_SPEED = -6 # 중심선 이탈 시 안쪽 바퀴 역회전 탐색

TEXT_FOLLOW_SPEED = 16
TEXT_FOLLOW_KP = 0.10
TEXT_FOLLOW_MAX_CORR = 10

TURN_SPEED = 18
ALIGN_SPEED = 6
CENTER_TOL = 30


# ============================================================
# ROI
# ============================================================
ROI_START_RATIO = 0.50


# ============================================================
# HSV
# ============================================================
LOWER_WHITE = np.array([0, 0, 175], dtype=np.uint8)
UPPER_WHITE = np.array([180, 75, 255], dtype=np.uint8)

LOWER_BLACK = np.array([0, 0, 0], dtype=np.uint8)
UPPER_BLACK = np.array([180, 130, 115], dtype=np.uint8)


# ============================================================
# 중심선 검출 설정
# ============================================================
MIN_AREA = 500
STEP = 15
MIN_WIDTH = 25
MAX_JUMP = 60
MAX_POINTS = 8

INTERNAL_GAP_RATIO = 0.30

# ============================================================
# 시작 후 25초 동안 중심선 추종
# ============================================================
CENTERLINE_RUN_SEC = 25.0
auto_start_time = None
initial_25s_done = False


# ============================================================
# YOLO STOP / STATION
# ============================================================
MODEL_PATH = "best_ncnn_model"
YOLO_CONF = 0.45
YOLO_IMGSZ = 320
TEXT_CLASSES = {"STOP", "STATION", "GOAL"}

# 현재 코스에서 찾아야 하는 표지 순서
# STOP1 -> STATION -> STOP2 -> GOAL
route_stage = "STOP1"

# STATION: 박스 하단이 화면 밑에 닿으면 회전 없이 2초 직진 -> 3초 정지
STATION_FORWARD_SEC = 2.0

# ============================================================
# STATION 이후 화살표 처리
# 화살표를 흰 도로(road_mask)와 똑같이 취급해서
# 화살표-도로선 사이로 중심선이 잡히는 문제를 막는다
# ============================================================
ARROW_FILL_STAGES = {"STOP2"}   # 화살표 채우기를 적용할 route_stage
ARROW_GAP_RATIO = 0.80          # 이 구간에서는 흰 영역 사이 간격을 화면폭 80%까지 메움
ARROW_CLOSE_KERNEL = 15         # 화살표 주변 틈을 메우는 닫힘 연산 커널 크기

# 마지막 STOP2 이후 GOAL 구간
FINAL_STOP_FORWARD_SEC = 3.0
FINAL_RIGHT_TURN_SEC = 0.90   # 90도 자체를 직접 측정할 수 없으므로 시간으로 근사. 현장에서 0.1초 단위로 튜닝
GOAL_NEAR_RATIO = 0.70
GOAL_MISS_TIMEOUT = 0.50
goal_seen_near = False
goal_missing_since = None

TEXT_FULL_MARGIN = 35
TEXT_FULL_STABLE_FRAMES = 3

TEXT_ALIGN_TRIGGER_RATIO = 0.60

TEXT_BOTTOM_TRIGGER_RATIO = 0.98

FORWARD_AFTER_TEXT_SEC = 3.0
STOP_AFTER_TEXT_SEC = 3.0

text_full_count = 0
text_candidate_type = None
current_text_type = None

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"{MODEL_PATH} 파일이 없습니다. 이 코드와 같은 폴더에 넣어주세요."
    )

print("Loading YOLO model...")
yolo_model = YOLO(MODEL_PATH)

# YOLO 첫 추론 지연 방지용 워밍업
try:
    warmup_img = np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8)
    yolo_model.predict(
        warmup_img,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        verbose=False
    )
    print("YOLO warm-up done.")
except Exception as e:
    print("YOLO warm-up failed:", e)

print("YOLO model loaded.")
print("YOLO classes:", yolo_model.names)


# ============================================================
# 상태
# ============================================================
drive_state = "CENTERLINE"
auto_mode = False
stop_event = threading.Event()

last_error = 0
state_start_time = 0.0

latest_jpeg = None
jpeg_lock = threading.Lock()

manual_until = 0.0
manual_cmd = None

MANUAL_SPEED = 24
MANUAL_PULSE = 0.30
JPEG_QUALITY = 55


# ============================================================
# ZMQ 통신 클라이언트 대기 스레드
# ============================================================
def zmq_wait_for_start():
    global auto_mode, drive_state

    try:
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(
            f"tcp://{LAPTOP_ZMQ_IP}:{LAPTOP_ZMQ_PORT}"
        )

        print("[핑키봇] 노트북 ZMQ 서버에 접속합니다...")
        socket.send_string("핑키봇 준비 완료!")

        response_raw = socket.recv_string()
        response = json.loads(response_raw)

        if response.get("status") == "START_AUTONAV":
            print("[핑키봇] START_AUTONAV 수신 -> AUTO 시작")
            auto_mode = True
            drive_state = "CENTERLINE"

    except Exception as e:
        print("[ZMQ] 대기 스레드 오류:", e)


# ============================================================
# Motor helpers
# ============================================================
def clamp(v, lo=-100, hi=100):
    return max(lo, min(hi, int(v)))


def drive(left, right):
    motor.move(clamp(left), clamp(right))


def stop_robot():
    motor.move(0, 0)


# ============================================================
# 작은 흰색 노이즈 제거
# ============================================================
def remove_small_components(binary_mask, min_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8
    )

    cleaned = np.zeros_like(binary_mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned


# ============================================================
# 중심선용 내부 간격 메우기
# ============================================================
def fill_road_internal_gaps(white_mask, gap_ratio=INTERNAL_GAP_RATIO):
    filled = white_mask.copy()
    h, w = white_mask.shape
    max_internal_gap = int(w * gap_ratio)

    for y in range(h):
        xs = np.where(white_mask[y] == 255)[0]
        if len(xs) == 0:
            continue

        groups = np.split(xs, np.where(np.diff(xs) > 1)[0] + 1)
        groups = [g for g in groups if len(g) >= MIN_WIDTH]

        if len(groups) < 2:
            continue

        for i in range(len(groups) - 1):
            left_group = groups[i]
            right_group = groups[i + 1]

            left_end = int(left_group[-1])
            right_start = int(right_group[0])

            gap = right_start - left_end - 1

            if 0 < gap <= max_internal_gap:
                filled[y, left_end:right_start + 1] = 255

    return filled


# ============================================================
# STATION 이후: 화살표를 흰 도로로 채우기
# 1) 큰 커널 닫힘 연산으로 화살표 테두리 틈 메우기
# 2) 흰 도로에 완전히 둘러싸인 구멍(화살표) 채우기
# 3) 좌우 흰 영역 사이 간격을 넓게 메워서 화살표가 도로 가장자리에
#    닿아 있어도 하나의 도로로 합치기
# ============================================================
def fill_arrow_as_road(white_mask):
    k = np.ones((ARROW_CLOSE_KERNEL, ARROW_CLOSE_KERNEL), np.uint8)
    mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, k, iterations=1)

    # 외곽 윤곽선 내부를 모두 채움 -> 둘러싸인 화살표 구멍 제거
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    # 열린 모양(가장자리에 닿은) 화살표까지 행 단위로 메움
    filled = fill_road_internal_gaps(filled, gap_ratio=ARROW_GAP_RATIO)
    return filled


# ============================================================
# YOLO STOP / STATION 검출
# ============================================================
def detect_text_yolo(frame, y_offset=0, expected_class=None):
    results = yolo_model(
        frame,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        verbose=False
    )

    if not results:
        return None

    r = results[0]
    if r.boxes is None:
        return None

    candidates = []

    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = str(yolo_model.names[cls_id]).upper().strip()

        if name not in TEXT_CLASSES:
            continue

        if expected_class is not None and name != expected_class:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        y1_full = y1 + y_offset
        y2_full = y2 + y_offset

        candidates.append({
            "type": name,
            "conf": conf,
            "bbox": (x1, y1_full, x2, y2_full),
            "center": ((x1 + x2) // 2, (y1_full + y2_full) // 2)
        })

    if not candidates:
        return None

    return max(candidates, key=lambda z: z["bbox"][3])


# ============================================================
# Main control loop
# ============================================================
def control_loop():
    global drive_state
    global auto_mode
    global last_error
    global state_start_time
    global latest_jpeg
    global manual_until
    global manual_cmd

    global text_full_count
    global text_candidate_type
    global current_text_type

    global auto_start_time
    global initial_25s_done
    global route_stage
    global goal_seen_near
    global goal_missing_since

    def expected_class():
        if route_stage == "STOP1":
            return "STOP"
        if route_stage == "STATION":
            return "STATION"
        if route_stage == "STOP2":
            return "STOP"
        if route_stage == "GOAL":
            return "GOAL"
        return None

    while not stop_event.is_set():

        frame = camera.get_frame()

        if frame is None:
            time.sleep(0.02)
            continue

        frame = frame.copy()
        h, w = frame.shape[:2]

        # ----------------------------------------------------
        # ROI
        # ----------------------------------------------------
        roi_start = int(h * ROI_START_RATIO)
        roi = frame[roi_start:h, :]
        roi_h, roi_w = roi.shape[:2]

        # ----------------------------------------------------
        # HSV / Masks
        # ----------------------------------------------------
        roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
        hsv = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)

        white_mask = cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)
        black_mask = cv2.inRange(hsv, LOWER_BLACK, UPPER_BLACK)

        kernel_open = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

        kernel_close = np.ones((7, 7), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        white_mask = remove_small_components(white_mask, MIN_AREA)

        # STATION 이후 구간에서는 화살표를 흰 도로와 똑같이 취급
        if route_stage in ARROW_FILL_STAGES:
            road_mask = fill_arrow_as_road(white_mask)
        else:
            road_mask = fill_road_internal_gaps(white_mask)

        # ----------------------------------------------------
        # 중심선 계산
        # ----------------------------------------------------
        center_points = []
        prev_center = None

        for y in range(roi_h - 1, 0, -STEP):
            xs = np.where(road_mask[y] == 255)[0]

            if len(xs) == 0:
                continue

            groups = np.split(xs, np.where(np.diff(xs) > 1)[0] + 1)
            groups = [g for g in groups if len(g) >= MIN_WIDTH]

            if len(groups) == 0:
                continue

            if prev_center is None:
                chosen = min(
                    groups,
                    key=lambda g: abs((int(g[0]) + int(g[-1])) // 2 - roi_w // 2)
                )
            else:
                predicted_x = prev_center

                if len(center_points) >= 2:
                    x1 = center_points[-2][0]
                    x2 = center_points[-1][0]
                    dx = x2 - x1
                    dx = int(np.clip(dx, -35, 35))
                    predicted_x = x2 + dx

                chosen = min(
                    groups,
                    key=lambda g: abs((int(g[0]) + int(g[-1])) // 2 - predicted_x)
                )

            x_left = int(chosen[0])
            x_right = int(chosen[-1])
            x_center = (x_left + x_right) // 2

            if prev_center is not None:
                if abs(x_center - prev_center) > MAX_JUMP:
                    continue

            original_y = y + roi_start
            center_points.append((x_center, original_y))
            prev_center = x_center

            if len(center_points) >= MAX_POINTS:
                break

        target_x = None
        target_y = None
        error = None

        if len(center_points) >= 3:
            target_x, target_y = center_points[2]

        if target_x is not None:
            error = target_x - w // 2
            last_error = error

        # ----------------------------------------------------
        # YOLO (25초 이후부터 활성화)
        # ----------------------------------------------------
        text_target = None

        if (
            initial_25s_done
            and drive_state in (
                "CENTERLINE",
                "SEARCH_TEXT",
                "APPROACH_TEXT",
                "DRIVE_TEXT",
                "GOAL_APPROACH",
                "GOAL_PASSING"
            )
        ):
            try:
                text_target = detect_text_yolo(
                    roi,
                    y_offset=roi_start,
                    expected_class=expected_class()
                )
            except Exception as e:
                print("YOLO ERROR:", e)

        # ----------------------------------------------------
        # Manual override
        # ----------------------------------------------------
        if time.time() < manual_until and manual_cmd is not None:
            l, r = manual_cmd
            drive(l, r)

        elif not auto_mode:
            stop_robot()

        else:
            # ================================================
            # CENTERLINE (통합 구조)
            # ================================================
            if drive_state == "CENTERLINE":

                if auto_start_time is None:
                    auto_start_time = time.time()

                elapsed_centerline = time.time() - auto_start_time

                # [이벤트 1] 최초 시작 후 25초 경과 시 정지 후 SEARCH_TEXT로 전환
                if not initial_25s_done and elapsed_centerline >= CENTERLINE_RUN_SEC:
                    stop_robot()
                    initial_25s_done = True
                    drive_state = "SEARCH_TEXT"
                    state_start_time = time.time()

                    text_full_count = 0
                    text_candidate_type = None

                    print("[TIMER] first 25s centerline done -> SEARCH_TEXT")

                # [이벤트 2] 25초 이후 중심선 주행 중 STOP/STATION 글씨 감지 시
                elif initial_25s_done and text_target is not None:
                    current_text_type = text_target["type"]

                    if current_text_type == "GOAL":
                        drive_state = "GOAL_APPROACH"
                        goal_seen_near = False
                        goal_missing_since = None
                        print("[YOLO] GOAL detected -> GOAL_APPROACH")
                    else:
                        drive_state = "APPROACH_TEXT"
                        print(f"[YOLO] {text_target['type']} detected -> APPROACH_TEXT")

                # [기본 상태] 별도 이벤트가 없으면 무조건 '중심선 추종(PID/검색)'
                else:
                    if error is not None:
                        # 커브 감속 + 안쪽 바퀴 역회전 허용
                        base = max(
                            MIN_CURVE_SPEED,
                            BASE_SPEED - CURVE_SLOWDOWN * abs(error)
                        )

                        correction = KP * error

                        left_speed = int(np.clip(
                            base + correction,
                            INNER_MIN_SPEED,
                            MAX_SPEED
                        ))
                        right_speed = int(np.clip(
                            base - correction,
                            INNER_MIN_SPEED,
                            MAX_SPEED
                        ))
                        drive(left_speed, right_speed)

                    else:
                        if last_error < 0:
                            drive(SEARCH_INNER_SPEED, SEARCH_SPEED)
                        elif last_error > 0:
                            drive(SEARCH_SPEED, SEARCH_INNER_SPEED)
                        else:
                            stop_robot()

            # ================================================
            # SEARCH_TEXT
            # ================================================
            elif drive_state == "SEARCH_TEXT":

                full_text_target = None

                if text_target is not None:
                    x1, y1, x2, y2 = text_target["bbox"]
                    text_fully_inside = (x1 >= TEXT_FULL_MARGIN and x2 <= (w - TEXT_FULL_MARGIN))
                    current_type = text_target["type"]

                    if text_fully_inside:
                        if text_candidate_type == current_type:
                            text_full_count += 1
                        else:
                            text_candidate_type = current_type
                            text_full_count = 1

                        if text_full_count >= TEXT_FULL_STABLE_FRAMES:
                            full_text_target = text_target
                    else:
                        text_full_count = 0
                        text_candidate_type = None
                else:
                    text_full_count = 0
                    text_candidate_type = None

                if full_text_target is not None:
                    stop_robot()
                    current_text_type = full_text_target["type"]

                    if current_text_type == "GOAL":
                        drive_state = "GOAL_APPROACH"
                        goal_seen_near = False
                        goal_missing_since = None
                        print("[YOLO] GOAL full -> GOAL_APPROACH")
                    else:
                        drive_state = "APPROACH_TEXT"
                        print(f"[YOLO] {full_text_target['type']} full -> APPROACH_TEXT")
                else:
                    drive(TURN_SPEED, -TURN_SPEED)

            # ================================================
            # APPROACH_TEXT
            # ================================================
            elif drive_state == "APPROACH_TEXT":

                if text_target is None:
                    drive(TEXT_FOLLOW_SPEED, TEXT_FOLLOW_SPEED)
                else:
                    text_x = text_target["center"][0]
                    _, _, _, text_y2 = text_target["bbox"]
                    text_error = text_x - w // 2

                    corr = np.clip(
                        TEXT_FOLLOW_KP * text_error,
                        -TEXT_FOLLOW_MAX_CORR,
                        TEXT_FOLLOW_MAX_CORR
                    )

                    drive(TEXT_FOLLOW_SPEED + corr, TEXT_FOLLOW_SPEED - corr)

                    if text_y2 >= int(h * TEXT_ALIGN_TRIGGER_RATIO):
                        drive_state = "DRIVE_TEXT"
                        print(f"[YOLO] {text_target['type']} reached 60% -> DRIVE_TEXT")

            # ================================================
            # DRIVE_TEXT
            # ================================================
            elif drive_state == "DRIVE_TEXT":

                if text_target is not None:
                    text_x = text_target["center"][0]
                    _, _, _, text_y2 = text_target["bbox"]
                    text_error = text_x - w // 2

                    if text_y2 >= int(h * TEXT_BOTTOM_TRIGGER_RATIO):
                        stop_robot()
                        state_start_time = time.time()

                        if current_text_type == "STATION":
                            # 회전 없이 바로 2초 직진 -> 3초 정지
                            drive_state = "FORWARD_2SEC"
                            print(f"[STATION] bbox bottom reached -> forward {STATION_FORWARD_SEC:.1f}s (no turn)")
                        elif current_text_type == "GOAL":
                            drive_state = "GOAL_PASSING"
                            goal_seen_near = True
                            goal_missing_since = None
                            print("[YOLO] GOAL near -> GOAL_PASSING")
                        else:
                            drive_state = "FORWARD_2SEC"
                            print(f"[YOLO] {current_text_type} bbox bottom reached -> forward 3.0s")

                    else:
                        corr = np.clip(
                            TEXT_FOLLOW_KP * text_error,
                            -TEXT_FOLLOW_MAX_CORR,
                            TEXT_FOLLOW_MAX_CORR
                        )
                        drive(TEXT_FOLLOW_SPEED + corr, TEXT_FOLLOW_SPEED - corr)

                else:
                    drive(TEXT_FOLLOW_SPEED, TEXT_FOLLOW_SPEED)

            # ================================================
            # FORWARD_2SEC
            # ================================================
            elif drive_state == "FORWARD_2SEC":

                forward_duration = (
                    STATION_FORWARD_SEC
                    if current_text_type == "STATION"
                    else FORWARD_AFTER_TEXT_SEC
                )

                if time.time() - state_start_time < forward_duration:
                    drive(TEXT_FOLLOW_SPEED, TEXT_FOLLOW_SPEED)
                else:
                    stop_robot()
                    state_start_time = time.time()
                    drive_state = "STOP_3SEC"
                    print(f"[TEXT] forward {forward_duration:.1f}s done -> stop 3.0s")

            # ================================================
            # STOP_3SEC
            # ================================================
            elif drive_state == "STOP_3SEC":

                stop_robot()

                if time.time() - state_start_time >= STOP_AFTER_TEXT_SEC:

                    # 숫자 stop_count를 사용하지 않고 현재 코스 단계로 구분
                    if route_stage == "STOP1":
                        route_stage = "STATION"
                        current_text_type = None
                        drive_state = "CENTERLINE"
                        print("[ROUTE] STOP1 done -> SEARCH STATION")

                    elif route_stage == "STATION":
                        route_stage = "STOP2"
                        current_text_type = None
                        drive_state = "CENTERLINE"
                        print("[ROUTE] STATION done -> CENTERLINE (arrow filled) / SEARCH STOP2")

                    elif route_stage == "STOP2":
                        current_text_type = None
                        state_start_time = time.time()
                        drive_state = "FINAL_FORWARD"
                        print("[ROUTE] STOP2 done -> FINAL_FORWARD 3.0s")

                    else:
                        current_text_type = None
                        drive_state = "CENTERLINE"

            # ================================================
            # FINAL_FORWARD
            # STOP2에서 마지막 우회전 위치까지 직진
            # ================================================
            elif drive_state == "FINAL_FORWARD":

                if time.time() - state_start_time < FINAL_STOP_FORWARD_SEC:
                    drive(TEXT_FOLLOW_SPEED, TEXT_FOLLOW_SPEED)
                else:
                    stop_robot()
                    state_start_time = time.time()
                    drive_state = "FINAL_RIGHT_TURN"
                    print(f"[ROUTE] final forward {FINAL_STOP_FORWARD_SEC:.1f}s done -> big right turn")

            # ================================================
            # FINAL_RIGHT_TURN
            # 큰 오른쪽 회전. 90도는 시간으로 근사
            # ================================================
            elif drive_state == "FINAL_RIGHT_TURN":

                if time.time() - state_start_time < FINAL_RIGHT_TURN_SEC:
                    drive(TURN_SPEED, -TURN_SPEED)
                else:
                    stop_robot()
                    state_start_time = time.time()
                    route_stage = "GOAL"
                    drive_state = "CENTERLINE"
                    print("[ROUTE] final right turn done -> CENTERLINE / SEARCH GOAL")

            # ================================================
            # GOAL_APPROACH
            # GOAL 중심을 따라 접근
            # ================================================
            elif drive_state == "GOAL_APPROACH":

                if text_target is None:
                    # GOAL이 잠깐 놓치면 바로 정지하지 않고 마지막 방향으로 저속 직진
                    drive(TEXT_FOLLOW_SPEED * 0.75, TEXT_FOLLOW_SPEED * 0.75)
                else:
                    text_x = text_target["center"][0]
                    _, _, _, text_y2 = text_target["bbox"]
                    text_error = text_x - w // 2

                    corr = np.clip(
                        TEXT_FOLLOW_KP * text_error,
                        -TEXT_FOLLOW_MAX_CORR,
                        TEXT_FOLLOW_MAX_CORR
                    )
                    drive(
                        TEXT_FOLLOW_SPEED + corr,
                        TEXT_FOLLOW_SPEED - corr
                    )

                    if text_y2 >= int(h * GOAL_NEAR_RATIO):
                        goal_seen_near = True
                        goal_missing_since = None
                        drive_state = "GOAL_PASSING"
                        print("[GOAL] near -> GOAL_PASSING")

            # ================================================
            # GOAL_PASSING
            # GOAL이 화면에서 사라질 때까지 직진
            # ================================================
            elif drive_state == "GOAL_PASSING":

                if text_target is not None:
                    goal_missing_since = None

                    text_x = text_target["center"][0]
                    text_error = text_x - w // 2

                    corr = np.clip(
                        TEXT_FOLLOW_KP * text_error,
                        -TEXT_FOLLOW_MAX_CORR,
                        TEXT_FOLLOW_MAX_CORR
                    )
                    drive(
                        TEXT_FOLLOW_SPEED + corr,
                        TEXT_FOLLOW_SPEED - corr
                    )
                else:
                    if goal_missing_since is None:
                        goal_missing_since = time.time()

                    drive(TEXT_FOLLOW_SPEED * 0.65, TEXT_FOLLOW_SPEED * 0.65)

                    if (
                        goal_seen_near
                        and time.time() - goal_missing_since >= GOAL_MISS_TIMEOUT
                    ):
                        stop_robot()
                        route_stage = "DONE"
                        drive_state = "DONE"
                        auto_mode = False
                        print("[GOAL] disappeared -> ROBOT STOP / DONE")

            # ================================================
            # DONE
            # ================================================
            elif drive_state == "DONE":
                stop_robot()

        # ----------------------------------------------------
        # 화면 표시
        # ----------------------------------------------------
        result = frame.copy()

        cv2.line(result, (0, roi_start), (w, roi_start), (0, 255, 255), 2)
        cv2.line(result, (w // 2, roi_start), (w // 2, h), (0, 255, 0), 2)

        for x, y in center_points:
            cv2.circle(result, (x, y), 4, (0, 0, 255), -1)

        for i in range(len(center_points) - 1):
            cv2.line(result, center_points[i], center_points[i + 1], (255, 0, 0), 3)

        if target_x is not None:
            cv2.circle(result, (int(target_x), int(target_y)), 9, (0, 255, 255), -1)

        if text_target is not None:
            x1, y1, x2, y2 = text_target["bbox"]
            cv2.rectangle(result, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(
                result,
                f"{text_target['type']} {text_target['conf']:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 0, 255),
                2
            )
            cv2.circle(result, text_target["center"], 7, (0, 0, 255), -1)

        shown_state = drive_state if auto_mode else f"PAUSED / {drive_state}"

        cv2.putText(
            result,
            f"STATE: {shown_state}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 255),
            2
        )

        cv2.putText(
            result,
            f"ROUTE: {route_stage}",
            (12, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 0),
            2
        )

        if (
            auto_mode
            and drive_state == "CENTERLINE"
            and not initial_25s_done
            and auto_start_time is not None
        ):
            elapsed_show = min(CENTERLINE_RUN_SEC, time.time() - auto_start_time)
            remain_show = max(0.0, CENTERLINE_RUN_SEC - elapsed_show)
            cv2.putText(
                result,
                f"CENTERLINE TIMER: {remain_show:.1f}s",
                (12, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 255),
                2
            )

        road_bgr = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
        road_bgr = cv2.resize(road_bgr, (result.shape[1], result.shape[0]), interpolation=cv2.INTER_NEAREST)

        combo = np.hstack([result, road_bgr])

        ok, jpg = cv2.imencode(".jpg", combo, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

        if ok:
            with jpeg_lock:
                latest_jpeg = jpg.tobytes()

        time.sleep(0.03)


# ============================================================
# Flask
# ============================================================
HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Pinky Centerline + YOLO Route</title>
<style>
body{
    background:#111;
    color:white;
    font-family:Arial;
    text-align:center
}
img{
    width:95%;
    max-width:1200px;
    border:2px solid #555
}
button{
    font-size:24px;
    margin:5px;
    padding:10px 20px
}
</style>
</head>
<body>

<h2>Pinky Centerline + YOLO STOP / STATION / GOAL</h2>

<img src="/video_feed">

<div>
<button onclick="cmd('w')">W</button>
<button onclick="cmd('s')">S</button>
<button onclick="cmd('a')">A</button>
<button onclick="cmd('d')">D</button>
<button onclick="cmd('p')">AUTO</button>
<button onclick="cmd('r')">RESET</button>
<button onclick="cmd('space')">STOP</button>
</div>

<script>
function cmd(k){
    fetch('/cmd/'+k);
}

document.addEventListener(
    'keydown',
    function(e){
        if(e.repeat) return;

        let k=e.key.toLowerCase();

        if(e.code==='Space'){
            e.preventDefault();
            cmd('space');
        }
        else if(
            ['w','a','s','d','p','r'].includes(k)
        ){
            cmd(k);
        }
    }
);
</script>

</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


def mjpeg():
    while True:
        with jpeg_lock:
            data = latest_jpeg

        if data is not None:
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + data
                + b'\r\n'
            )

        time.sleep(0.04)


@app.route("/video_feed")
def video_feed():
    return Response(
        mjpeg(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route("/cmd/<key>")
def command(key):
    global auto_mode
    global drive_state
    global manual_until
    global manual_cmd

    global text_full_count
    global text_candidate_type
    global current_text_type

    global auto_start_time
    global initial_25s_done
    global route_stage
    global goal_seen_near
    global goal_missing_since

    global last_error
    global state_start_time

    if key == "p":
        auto_mode = not auto_mode

        if auto_mode:
            drive_state = "CENTERLINE"

            if not initial_25s_done:
                auto_start_time = time.time()
                print("AUTO ON -> FIRST 25s CENTERLINE")
            else:
                print("AUTO ON -> NORMAL CENTERLINE")
        else:
            stop_robot()
            print("AUTO OFF")

    elif key == "r":
        auto_mode = False
        drive_state = "CENTERLINE"

        last_error = 0
        state_start_time = 0.0

        auto_start_time = None
        initial_25s_done = False
        route_stage = "STOP1"
        goal_seen_near = False
        goal_missing_since = None

        text_full_count = 0
        text_candidate_type = None
        current_text_type = None

        stop_robot()

        print("RESET")

    elif key == "space":
        auto_mode = False
        stop_robot()

    elif key == "w":
        manual_cmd = (MANUAL_SPEED, MANUAL_SPEED)
        manual_until = time.time() + MANUAL_PULSE

    elif key == "s":
        manual_cmd = (-MANUAL_SPEED, -MANUAL_SPEED)
        manual_until = time.time() + MANUAL_PULSE

    elif key == "a":
        manual_cmd = (-MANUAL_SPEED, MANUAL_SPEED)
        manual_until = time.time() + MANUAL_PULSE

    elif key == "d":
        manual_cmd = (MANUAL_SPEED, -MANUAL_SPEED)
        manual_until = time.time() + MANUAL_PULSE

    return "OK"


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    threading.Thread(
        target=control_loop,
        daemon=True
    ).start()

    print("Pinky server started")
    print("http://ROBOT_IP:5000")

    try:
        app.run(
            host="0.0.0.0",
            port=PORT,
            threaded=True,
            debug=False
        )
    finally:
        stop_event.set()
        stop_robot()
