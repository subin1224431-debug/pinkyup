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
# 기존 Raspberry Pi 코드에서 사용하던 주소
LAPTOP_ZMQ_IP = "172.20.10.14"
LAPTOP_ZMQ_PORT = 6000


# ============================================================
# 주행 설정
# ============================================================
BASE_SPEED = 20
KP = 0.12
MAX_SPEED = 32
SEARCH_SPEED = 14

TEXT_FOLLOW_SPEED = 16
TEXT_FOLLOW_KP = 0.10
TEXT_FOLLOW_MAX_CORR = 10

TURN_SPEED = 24
ALIGN_SPEED = 8
CENTER_TOL = 30


# ============================================================
# ROI
# ============================================================
ROI_START_RATIO = 0.68


# ============================================================
# 흰색 / 검은색 HSV
#
# 중심선용 road_mask:
# - 흰색 도로를 기준으로 잡음
# - 도로 내부의 검은 화살표/황토색 박스는 흰색으로 메움
# - 도로 바깥 황토색은 도로로 직접 등록하지 않음
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
# 카메라 화살표 카운팅
#
# 카메라에서 새 화살표를 발견할 때 1,2,3까지만 센다.
# ============================================================
ARROW_MIN_AREA = 350
ARROW_STABLE_FRAMES = 3
ARROW_GONE_FRAMES = 8

arrow_count = 0
arrow_stable_count = 0
arrow_missing_count = 0
arrow_count_armed = True


# ============================================================
# 3번째 화살표 거리 기준
# ============================================================
third_arrow_trigger_count = 0


# ============================================================
# YOLO STOP / STATION
# ============================================================
MODEL_PATH = "best_ncnn_model"
YOLO_CONF = 0.45
YOLO_IMGSZ = 320
TEXT_CLASSES = {"STOP", "STATION"}

TEXT_FULL_MARGIN = 35
TEXT_FULL_STABLE_FRAMES = 3

text_full_count = 0
text_candidate_type = None

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"{MODEL_PATH} 파일이 없습니다. 이 코드와 같은 폴더에 넣어주세요."
    )

print("Loading YOLO model...")
yolo_model = YOLO(MODEL_PATH)
print("YOLO model loaded.")
print("YOLO classes:", yolo_model.names)


# ============================================================
#
# 핵심:
# YOLO가 STOP/STATION을 인식한 뒤에만
# ============================================================


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
    return max(
        lo,
        min(
            hi,
            int(v)
        )
    )


def drive(left, right):
    motor.move(
        clamp(left),
        clamp(right)
    )


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
        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        if area >= min_area:
            cleaned[
                labels == i
            ] = 255

    return cleaned


# ============================================================
# 중심선용 내부 간격 메우기
#
# 흰색 도로 사이의 작은 내부 간격은
# 색과 상관없이 흰색으로 메운다.
#
# 따라서:
# - 검은 화살표 -> 중심선용 road_mask에서 제거
# - 황토색 박스 -> 흰 도로 내부라면 road_mask에서 흰길처럼 처리
#
# 단, 황토색을 도로색 자체로 등록하지 않으므로
# 도로 바깥 황토색 전체가 길로 잡히지는 않는다.
# ============================================================
def fill_road_internal_gaps(white_mask):
    filled = white_mask.copy()

    h, w = white_mask.shape
    max_internal_gap = int(
        w * INTERNAL_GAP_RATIO
    )

    for y in range(h):
        xs = np.where(
            white_mask[y] == 255
        )[0]

        if len(xs) == 0:
            continue

        groups = np.split(
            xs,
            np.where(
                np.diff(xs) > 1
            )[0] + 1
        )

        groups = [
            g
            for g in groups
            if len(g) >= MIN_WIDTH
        ]

        if len(groups) < 2:
            continue

        for i in range(
            len(groups) - 1
        ):
            left_group = groups[i]
            right_group = groups[i + 1]

            left_end = int(
                left_group[-1]
            )

            right_start = int(
                right_group[0]
            )

            gap = (
                right_start
                - left_end
                - 1
            )

            if (
                gap > 0
                and gap <= max_internal_gap
            ):
                filled[
                    y,
                    left_end:right_start + 1
                ] = 255

    return filled


# ============================================================
# 카메라 화살표 검출
# ============================================================
def detect_arrow_bbox(black_mask):
    contours, _ = cv2.findContours(
        black_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    h, w = black_mask.shape[:2]

    for c in contours:
        area = cv2.contourArea(c)

        if area < ARROW_MIN_AREA:
            continue

        x, y, bw, bh = cv2.boundingRect(c)

        if bw <= 0 or bh <= 0:
            continue

        aspect = bw / max(
            bh,
            1
        )

        # 너무 가로로 긴 글씨/선은 제외
        if aspect > 2.8:
            continue

        if (
            bw > int(w * 0.80)
            or bh > int(h * 0.95)
        ):
            continue

        candidates.append({
            "bbox": (
                x,
                y,
                bw,
                bh
            ),
            "area": float(area),
            "center": (
                x + bw // 2,
                y + bh // 2
            )
        })

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda z: (
            z["bbox"][1]
            + z["bbox"][3],
            z["area"]
        )
    )


# ============================================================
# YOLO STOP / STATION 검출
# ============================================================
def detect_text_yolo(frame):
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
        cls_id = int(
            box.cls[0]
        )

        conf = float(
            box.conf[0]
        )

        name = str(
            yolo_model.names[cls_id]
        ).upper().strip()

        if name not in TEXT_CLASSES:
            continue

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        candidates.append({
            "type": name,
            "conf": conf,
            "bbox": (
                x1,
                y1,
                x2,
                y2
            ),
            "center": (
                (x1 + x2) // 2,
                (y1 + y2) // 2
            )
        })

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda z: z["bbox"][3]
    )


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

    global arrow_count
    global arrow_stable_count
    global arrow_missing_count
    global arrow_count_armed
    global third_arrow_trigger_count

    global text_full_count
    global text_candidate_type
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
        roi_start = int(
            h * ROI_START_RATIO
        )

        roi = frame[
            roi_start:h,
            :
        ]

        roi_h, roi_w = roi.shape[:2]

        # ----------------------------------------------------
        # HSV / Masks
        # ----------------------------------------------------
        roi_blur = cv2.GaussianBlur(
            roi,
            (5, 5),
            0
        )

        hsv = cv2.cvtColor(
            roi_blur,
            cv2.COLOR_BGR2HSV
        )

        white_mask = cv2.inRange(
            hsv,
            LOWER_WHITE,
            UPPER_WHITE
        )

        black_mask = cv2.inRange(
            hsv,
            LOWER_BLACK,
            UPPER_BLACK
        )

        kernel_open = np.ones(
            (3, 3),
            np.uint8
        )

        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_OPEN,
            kernel_open,
            iterations=1
        )

        black_mask = cv2.morphologyEx(
            black_mask,
            cv2.MORPH_OPEN,
            kernel_open,
            iterations=1
        )

        kernel_close = np.ones(
            (7, 7),
            np.uint8
        )

        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_CLOSE,
            kernel_close,
            iterations=1
        )

        white_mask = remove_small_components(
            white_mask,
            MIN_AREA
        )

        # 중심선용: 내부 화살표/박스 제거
        road_mask = fill_road_internal_gaps(
            white_mask
        )

        # ----------------------------------------------------
        # 중심선 계산
        # ----------------------------------------------------
        center_points = []
        prev_center = None

        for y in range(
            roi_h - 1,
            0,
            -STEP
        ):
            xs = np.where(
                road_mask[y] == 255
            )[0]

            if len(xs) == 0:
                continue

            groups = np.split(
                xs,
                np.where(
                    np.diff(xs) > 1
                )[0] + 1
            )

            groups = [
                g
                for g in groups
                if len(g) >= MIN_WIDTH
            ]

            if len(groups) == 0:
                continue

            if prev_center is None:
                chosen = min(
                    groups,
                    key=lambda g:
                    abs(
                        (
                            int(g[0])
                            + int(g[-1])
                        ) // 2
                        - roi_w // 2
                    )
                )

            else:
                predicted_x = prev_center

                if len(center_points) >= 2:
                    x1 = center_points[-2][0]
                    x2 = center_points[-1][0]

                    dx = x2 - x1

                    dx = int(
                        np.clip(
                            dx,
                            -35,
                            35
                        )
                    )

                    predicted_x = x2 + dx

                chosen = min(
                    groups,
                    key=lambda g:
                    abs(
                        (
                            int(g[0])
                            + int(g[-1])
                        ) // 2
                        - predicted_x
                    )
                )

            x_left = int(
                chosen[0]
            )

            x_right = int(
                chosen[-1]
            )

            x_center = (
                x_left
                + x_right
            ) // 2

            if prev_center is not None:
                if abs(
                    x_center
                    - prev_center
                ) > MAX_JUMP:
                    continue

            original_y = (
                y
                + roi_start
            )

            center_points.append(
                (
                    x_center,
                    original_y
                )
            )

            prev_center = x_center

            if len(center_points) >= MAX_POINTS:
                break

        target_x = None
        target_y = None
        error = None

        if len(center_points) >= 3:
            target_x, target_y = (
                center_points[2]
            )

        if target_x is not None:
            error = (
                target_x
                - w // 2
            )

            last_error = error

        # ----------------------------------------------------
        # 카메라 화살표 검출 / 1,2,3 카운팅
        # ----------------------------------------------------
        arrow_target = detect_arrow_bbox(
            black_mask
        )

        if (
            arrow_count < 3
            and drive_state == "CENTERLINE"
        ):
            if arrow_target is not None:
                arrow_missing_count = 0

                if arrow_count_armed:
                    arrow_stable_count += 1

                    if (
                        arrow_stable_count
                        >= ARROW_STABLE_FRAMES
                    ):
                        arrow_count += 1
                        arrow_count_armed = False
                        arrow_stable_count = 0

                        print(
                            f"[CAMERA ARROW] {arrow_count}"
                        )

            else:
                arrow_stable_count = 0

                if not arrow_count_armed:
                    arrow_missing_count += 1

                    if (
                        arrow_missing_count
                        >= ARROW_GONE_FRAMES
                    ):
                        arrow_count_armed = True
                        arrow_missing_count = 0

        else:
            arrow_stable_count = 0

        # ----------------------------------------------------
        # 3번째 화살표 거리 기준선
        # ----------------------------------------------------
        if (
            drive_state == "CENTERLINE"
            and arrow_count == 3
            and arrow_target is not None
        ):
            ax, ay, aw, ah = (
                arrow_target["bbox"]
            )

            arrow_bottom = (
                ay + ah
            )

            trigger_y = int(
                roi_h
            )

            if arrow_bottom >= trigger_y:
                third_arrow_trigger_count += 1
            else:
                third_arrow_trigger_count = 0

            if (
                third_arrow_trigger_count
            ):
                stop_robot()

                drive_state = "SEARCH_TEXT"
                state_start_time = time.time()

                third_arrow_trigger_count = 0
                text_full_count = 0
                text_candidate_type = None

                print(
                    "[ARROW 3] distance line reached -> SEARCH_TEXT"
                )

        # ----------------------------------------------------
        # YOLO
        #
        # 첫 번째 3번째 화살표 이후부터는
        # 중심선 주행 중에도 다음 STOP/STATION을 계속 찾는다.
        # ----------------------------------------------------
        text_target = None

        if (
            arrow_count >= 3
            and drive_state in (
                "CENTERLINE",
                "SEARCH_TEXT",
                "ALIGN_TEXT",
                "DRIVE_TEXT"
            )
        ):
            try:
                text_target = detect_text_yolo(
                    frame
                )
            except Exception as e:
                print(
                    "YOLO ERROR:",
                    e
                )


        # ----------------------------------------------------
        # Manual override
        # ----------------------------------------------------
        if (
            time.time() < manual_until
            and manual_cmd is not None
        ):
            l, r = manual_cmd
            drive(l, r)

        elif not auto_mode:
            stop_robot()

        else:
            # ================================================
            # CENTERLINE
            # ================================================
            if drive_state == "CENTERLINE":

                # 3번째 화살표 이후에는
                # 중심선 추종 중 다음 STOP/STATION이 다시 보이면
                if (
                    arrow_count >= 3
                    and text_target is not None
                    
                ):
                    stop_robot()
                    drive_state = "ALIGN_TEXT"

                    print(
                        f"[YOLO] next {text_target['type']} detected "
                    )

                elif error is not None:
                    correction = (
                        KP * error
                    )

                    left_speed = (
                        BASE_SPEED
                        + correction
                    )

                    right_speed = (
                        BASE_SPEED
                        - correction
                    )

                    left_speed = int(
                        np.clip(
                            left_speed,
                            0,
                            MAX_SPEED
                        )
                    )

                    right_speed = int(
                        np.clip(
                            right_speed,
                            0,
                            MAX_SPEED
                        )
                    )

                    drive(
                        left_speed,
                        right_speed
                    )

                else:
                    if last_error < 0:
                        drive(
                            0,
                            SEARCH_SPEED
                        )

                    elif last_error > 0:
                        drive(
                            SEARCH_SPEED,
                            0
                        )

                    else:
                        stop_robot()

            # ================================================
            # SEARCH_TEXT
            # ================================================
            elif drive_state == "SEARCH_TEXT":

                full_text_target = None

                if text_target is not None:
                    x1, y1, x2, y2 = (
                        text_target["bbox"]
                    )

                    text_fully_inside = (
                        x1 >= TEXT_FULL_MARGIN
                        and x2 <= (
                            w
                            - TEXT_FULL_MARGIN
                        )
                    )

                    current_type = (
                        text_target["type"]
                    )

                    if text_fully_inside:
                        if (
                            text_candidate_type
                            == current_type
                        ):
                            text_full_count += 1
                        else:
                            text_candidate_type = current_type
                            text_full_count = 1

                        if (
                            text_full_count
                            >= TEXT_FULL_STABLE_FRAMES
                        ):
                            full_text_target = text_target

                    else:
                        text_full_count = 0
                        text_candidate_type = None

                else:
                    text_full_count = 0
                    text_candidate_type = None

                if full_text_target is not None:
                    stop_robot()

                    drive_state = "ALIGN_TEXT"

                    print(
                        f"[YOLO] {full_text_target['type']} "
                        "full -> ALIGN_TEXT"
                    )

                else:
                    drive(
                        TURN_SPEED,
                        -TURN_SPEED
                    )

            # ================================================
            # ALIGN_TEXT
            # ================================================
            elif drive_state == "ALIGN_TEXT":

                if text_target is None:
                    drive_state = "SEARCH_TEXT"

                else:
                    text_x = (
                        text_target["center"][0]
                    )

                    text_error = (
                        text_x
                        - w // 2
                    )

                    if abs(text_error) <= CENTER_TOL:
                        stop_robot()
                        drive_state = "DRIVE_TEXT"

                        print(
                            f"[YOLO] {text_target['type']} "
                            "centered -> DRIVE_TEXT"
                        )

                    else:
                        if text_error > 0:
                            drive(
                                ALIGN_SPEED,
                                -ALIGN_SPEED
                            )
                        else:
                            drive(
                                -ALIGN_SPEED,
                                ALIGN_SPEED
                            )

            # ================================================
            # DRIVE_TEXT
            # ================================================
            elif drive_state == "DRIVE_TEXT":



                elif text_target is not None:
                    text_x = (
                        text_target["center"][0]
                    )

                    text_error = (
                        text_x
                        - w // 2
                    )

                    corr = np.clip(
                        TEXT_FOLLOW_KP
                        * text_error,
                        -TEXT_FOLLOW_MAX_CORR,
                        TEXT_FOLLOW_MAX_CORR
                    )

                    drive(
                        TEXT_FOLLOW_SPEED + corr,
                        TEXT_FOLLOW_SPEED - corr
                    )

                else:
                    # 글씨가 아래로 빠진 후에도
                    drive(
                        TEXT_FOLLOW_SPEED,
                        TEXT_FOLLOW_SPEED
                    )


            # ================================================
            # STOP_3SEC
            # ================================================
            elif drive_state == "STOP_3SEC":

                stop_robot()

                if (
                    time.time()
                    - state_start_time
                ):
                    # 다음 YOLO 글씨가 나왔을 때

                    drive_state = "CENTERLINE"

                    print(
                        "ready for next YOLO"
                    )

        # ----------------------------------------------------
        # 화면 표시
        # ----------------------------------------------------
        result = frame.copy()

        cv2.line(
            result,
            (0, roi_start),
            (w, roi_start),
            (0, 255, 255),
            2
        )

        cv2.line(
            result,
            (w // 2, roi_start),
            (w // 2, h),
            (0, 255, 0),
            2
        )

        for x, y in center_points:
            cv2.circle(
                result,
                (x, y),
                4,
                (0, 0, 255),
                -1
            )

        for i in range(
            len(center_points) - 1
        ):
            cv2.line(
                result,
                center_points[i],
                center_points[i + 1],
                (255, 0, 0),
                3
            )

        if target_x is not None:
            cv2.circle(
                result,
                (
                    int(target_x),
                    int(target_y)
                ),
                9,
                (0, 255, 255),
                -1
            )

        if arrow_target is not None:
            ax, ay, aw, ah = (
                arrow_target["bbox"]
            )

            cv2.rectangle(
                result,
                (
                    ax,
                    ay + roi_start
                ),
                (
                    ax + aw,
                    ay + ah + roi_start
                ),
                (255, 0, 0),
                2
            )

        if (
            arrow_count == 3
            and drive_state == "CENTERLINE"
        ):
            third_line_y = (
                roi_start
                + int(
                    roi_h
                )
            )

            cv2.line(
                result,
                (0, third_line_y),
                (w, third_line_y),
                (0, 0, 255),
                2
            )

            cv2.putText(
                result,
                (
                    10,
                    max(
                        20,
                        third_line_y - 8
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 255),
                2
            )

        if text_target is not None:
            x1, y1, x2, y2 = (
                text_target["bbox"]
            )

            cv2.rectangle(
                result,
                (x1, y1),
                (x2, y2),
                (255, 0, 255),
                2
            )

            cv2.putText(
                result,
                f"{text_target['type']} {text_target['conf']:.2f}",
                (
                    x1,
                    max(
                        20,
                        y1 - 8
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 0, 255),
                2
            )

            cv2.circle(
                result,
                text_target["center"],
                7,
                (0, 0, 255),
                -1
            )

        shown_state = (
            drive_state
            if auto_mode
            else f"PAUSED / {drive_state}"
        )

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
            f"CAM ARROW: {arrow_count}/3",
            (12, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 255),
            2
        )

        # 오른쪽에 중심선용 road_mask 표시
        road_bgr = cv2.cvtColor(
            road_mask,
            cv2.COLOR_GRAY2BGR
        )

        road_bgr = cv2.resize(
            road_bgr,
            (
                result.shape[1],
                result.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )

        combo = np.hstack(
            [
                result,
                road_bgr
            ]
        )

        ok, jpg = cv2.imencode(
            ".jpg",
            combo,
            [
                int(
                    cv2.IMWRITE_JPEG_QUALITY
                ),
                JPEG_QUALITY
            ]
        )

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
<title>Pinky Centerline + YOLO</title>
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
    return render_template_string(
        HTML
    )


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
        mimetype=(
            'multipart/x-mixed-replace; '
            'boundary=frame'
        )
    )


@app.route("/cmd/<key>")
def command(key):
    global auto_mode
    global drive_state
    global manual_until
    global manual_cmd

    global arrow_count
    global arrow_stable_count
    global arrow_missing_count
    global arrow_count_armed
    global third_arrow_trigger_count

    global text_full_count
    global text_candidate_type
    global last_error
    global state_start_time

    if key == "p":
        auto_mode = not auto_mode

        if auto_mode:
            drive_state = "CENTERLINE"
            print("AUTO ON")
        else:
            stop_robot()
            print("AUTO OFF")

    elif key == "r":
        auto_mode = False
        drive_state = "CENTERLINE"

        last_error = 0
        state_start_time = 0.0

        arrow_count = 0
        arrow_stable_count = 0
        arrow_missing_count = 0
        arrow_count_armed = True

        third_arrow_trigger_count = 0

        text_full_count = 0
        text_candidate_type = None

        stop_robot()

        print("RESET")

    elif key == "space":
        auto_mode = False
        stop_robot()

    elif key == "w":
        manual_cmd = (
            MANUAL_SPEED,
            MANUAL_SPEED
        )
        manual_until = (
            time.time()
            + MANUAL_PULSE
        )

    elif key == "s":
        manual_cmd = (
            -MANUAL_SPEED,
            -MANUAL_SPEED
        )
        manual_until = (
            time.time()
            + MANUAL_PULSE
        )

    elif key == "a":
        manual_cmd = (
            -MANUAL_SPEED,
            MANUAL_SPEED
        )
        manual_until = (
            time.time()
            + MANUAL_PULSE
        )

    elif key == "d":
        manual_cmd = (
            MANUAL_SPEED,
            -MANUAL_SPEED
        )
        manual_until = (
            time.time()
            + MANUAL_PULSE
        )

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
