import cv2
import numpy as np
import time
import threading
from collections import deque
from flask import Flask, Response, render_template_string
from pinkylib import Camera, Motor, IR
from ultralytics import YOLO
import os

# ============================================================
# Pinky Pro - 화살표(OpenCV) + STOP/STATION(YOLO) 통합 주행
#
# 동작
# 1) 시작: 도로 이진화만 하면서 직진
# 2) 화살표는 OpenCV, STOP/STATION은 YOLO로 검출 -> 화면 중앙 정렬
# 3) 화살표: 내부 skeleton 중심선 추종
#    STOP/STATION: YOLO bbox 중심 추종
# 4) IR 센서가 검은색 감지 -> 정지
# 5) 다음 목표가 안 보이면 오른쪽 제자리 회전하며 탐색
# 6) 목표 발견 -> 중앙 정렬 -> 다시 직진/추종
# ============================================================

app = Flask(__name__)

# ----------------------------
# Hardware
# ----------------------------
motor = Motor()
camera = Camera()
ir = IR()

motor.enable_motor()
camera.start()

# ----------------------------
# Parameters
# ----------------------------
PORT = 5000

# ----------------------------
# YOLO text recognition
# ----------------------------
MODEL_PATH = "best.pt"
YOLO_CONF = 0.45
YOLO_IMGSZ = 320
YOLO_EVERY = 3          # Raspberry Pi 부하 감소: 3프레임마다 추론
TEXT_CLASSES = {"STOP", "STATION"}

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"{MODEL_PATH} 파일이 없습니다. road_test.py와 best.pt를 같은 폴더에 넣어주세요."
    )

print("Loading YOLO model...")
yolo_model = YOLO(MODEL_PATH)
print("YOLO model loaded.")
print("YOLO classes:", yolo_model.names)


# Motor speed
STRAIGHT_SPEED = 22
FOLLOW_SPEED = 19
SEARCH_SPEED = 10
ALIGN_SPEED = 10

# Manual pulse
MANUAL_SPEED = 24
MANUAL_PULSE = 0.30

# IR threshold
IR_THRESHOLD = 2600

# Camera ROI: 아래 50%, 좌우 10% 제외
ROI_TOP_RATIO = 0.50
ROI_SIDE_RATIO = 0.10

# HSV thresholds
WHITE_LO = np.array([0, 0, 175], dtype=np.uint8)
WHITE_HI = np.array([180, 75, 255], dtype=np.uint8)

BLACK_LO = np.array([0, 0, 0], dtype=np.uint8)
BLACK_HI = np.array([180, 130, 115], dtype=np.uint8)

# Detection
MIN_BLACK_AREA = 350
MIN_ARROW_AREA = 900
CENTER_TOL = 30

# Control
KP_ALIGN = 0.11
KP_FOLLOW = 0.10
MAX_CORR = 10

# Search behavior
IR_STOP_SEC = 0.45
IR_AFTER_HIT_DRIVE_SEC = 1.3   # IR 감지 후 1.3초 더 직진한 뒤 정지
IR_CLEAR_FRAMES = 4   # 검은 표식에서 벗어난 것이 연속 4프레임 확인되면 IR 재활성화
LOST_TARGET_FRAMES = 5

# Stream
JPEG_QUALITY = 55

# ----------------------------
# State
# ----------------------------
state = "START"
auto_mode = False
stop_event = threading.Event()

latest_jpeg = None
jpeg_lock = threading.Lock()

manual_until = 0.0
manual_cmd = None

lost_count = 0
last_target = None
ir_armed = True
ir_clear_count = 0

yolo_frame_count = 0
last_yolo_text = None
yolo_miss_count = 0
last_stable_yolo_text = None
YOLO_EDGE_MARGIN = 18
YOLO_STABLE_HOLD = 4
yolo_partial_count = 0

# ============================================================
# Motor helpers
# ============================================================

def clamp(v, lo=-100, hi=100):
    return max(lo, min(hi, int(v)))

def drive(left, right):
    left = clamp(left)
    right = clamp(right)
    motor.move(left, right)

def stop_robot():
    motor.move(0, 0)

def p_drive(error, base, kp, max_corr):
    corr = np.clip(kp * error, -max_corr, max_corr)
    left = base + corr
    right = base - corr
    drive(left, right)

# ============================================================
# ROI / Masks
# ============================================================

def get_roi(frame):
    h, w = frame.shape[:2]

    y1 = int(h * ROI_TOP_RATIO)
    x1 = int(w * ROI_SIDE_RATIO)
    x2 = int(w * (1.0 - ROI_SIDE_RATIO))

    roi = frame[y1:h, x1:x2]
    return roi, x1, y1

def make_masks(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    white = cv2.inRange(hsv, WHITE_LO, WHITE_HI)
    black = cv2.inRange(hsv, BLACK_LO, BLACK_HI)

    k3 = np.ones((3, 3), np.uint8)
    k7 = np.ones((7, 7), np.uint8)

    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, k3)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k7)

    black = cv2.morphologyEx(black, cv2.MORPH_OPEN, k3)
    black = cv2.morphologyEx(black, cv2.MORPH_CLOSE, k3)

    # --------------------------------------------------------
    # 중요:
    # 예전처럼 white를 단순 dilate해서 black과 AND하면
    # 화살표 내부가 잘리고 테두리만 남는 문제가 생김.
    #
    # 그래서 흰 도로의 가장 큰 외곽 contour를 "채워서"
    # 화살표/글씨가 들어있는 도로 전체 영역을 만든다.
    # --------------------------------------------------------
    road_region = np.zeros_like(white)

    contours, _ = cv2.findContours(
        white,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) > 1000:
            cv2.drawContours(
                road_region,
                [largest],
                -1,
                255,
                -1
            )

            # 도로 경계가 조금 끊겨도 내부를 유지
            road_region = cv2.dilate(
                road_region,
                np.ones((9,9), np.uint8),
                iterations=1
            )
        else:
            road_region[:] = 255
    else:
        road_region[:] = 255

    black = cv2.bitwise_and(black, road_region)

    return white, black, road_region

# ============================================================
# Skeleton
# ============================================================

def skeletonize(mask):
    img = mask.copy()
    skel = np.zeros_like(img)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded

        if cv2.countNonZero(img) == 0:
            break

    return skel

def largest_component(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    if n <= 1:
        return None

    best = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    area = stats[best, cv2.CC_STAT_AREA]

    if area < MIN_BLACK_AREA:
        return None

    out = np.zeros_like(mask)
    out[labels == best] = 255

    return out

def skeleton_centerline(mask):
    """
    화살표 내부를 1픽셀 skeleton으로 만든다.
    화면 표시에서는 skeleton 자체를 그대로 그린다.

    주행용 중심점은 화살표 bbox의 아래쪽~중간 구간에서
    skeleton 좌표를 찾아 사용한다.
    """
    skel = skeletonize(mask)

    ys, xs = np.where(skel > 0)

    if len(xs) < 10:
        return skel, [], None

    # skeleton 전체 표시용
    points = list(zip(xs.astype(int), ys.astype(int)))

    # 화살표의 실제 bbox
    py, px = np.where(mask > 0)

    if len(px) == 0:
        return skel, points, None

    x0, x1 = int(px.min()), int(px.max())
    y0, y1 = int(py.min()), int(py.max())
    hh = max(1, y1-y0+1)

    # 아래쪽에서 약 55~72% 높이의 중심선을 주행 look-ahead로 사용
    # (화살촉 가지가 갈라지는 맨 위쪽은 피함)
    band_top = int(y0 + 0.55*hh)
    band_bottom = int(y0 + 0.72*hh)

    sel = (ys >= band_top) & (ys <= band_bottom)

    if np.any(sel):
        tx = int(np.median(xs[sel]))
        ty = int(np.median(ys[sel]))
        follow_point = (tx,ty)
    else:
        # fallback: skeleton 전체 중앙
        follow_point = (
            int(np.median(xs)),
            int(np.median(ys))
        )

    return skel, points, follow_point

# ============================================================
# Target detection
# ============================================================

def find_black_groups(black_mask):
    contours, _ = cv2.findContours(
        black_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    items = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < 80:
            continue

        x,y,w,h = cv2.boundingRect(c)
        items.append({
            "contour": c,
            "area": area,
            "x": x,
            "y": y,
            "w": w,
            "h": h
        })

    return items

def detect_arrow_target(black_mask):
    """
    기존 OpenCV 방식은 화살표만 담당.
    STOP / STATION 글씨는 YOLO(best.pt)가 담당한다.
    """
    items = find_black_groups(black_mask)

    if not items:
        return None

    biggest = max(items, key=lambda z: z["area"])

    if biggest["area"] < MIN_ARROW_AREA:
        return None

    x,y,w,h = (
        biggest["x"],
        biggest["y"],
        biggest["w"],
        biggest["h"]
    )

    if h < 22 or w < 18:
        return None

    single = np.zeros_like(black_mask)

    cv2.drawContours(
        single,
        [biggest["contour"]],
        -1,
        255,
        -1
    )

    skel, path, follow_point = skeleton_centerline(single)

    if follow_point is None:
        return None

    return {
        "type": "ARROW",
        "bbox": (x,y,w,h),
        "center": follow_point,
        "mask": single,
        "skeleton": skel,
        "path": path,
        "follow_point": follow_point
    }


def detect_text_yolo(frame, ox, oy):
    """
    STOP / STATION은 오직 YOLO(best.pt)로만 검출한다.

    중요:
    글씨가 화면 가장자리에서 잘린 상태라면 bbox 중심이 실제 글씨 중심이 아니므로
    그 중심값으로 조향하지 않는다.

    반환:
    - bbox / center : 기존 ROI 기준 주행 제어용
    - frame_bbox    : 전체 프레임 기준
    - clipped       : 화면 가장자리에서 잘린 검출인지 여부
    """
    results = yolo_model(
        frame,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        verbose=False
    )

    candidates = []

    if not results:
        return None

    r = results[0]

    if r.boxes is None:
        return None

    fh, fw = frame.shape[:2]

    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = str(yolo_model.names[cls_id]).upper()

        if name not in TEXT_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # 화면 가장자리에 붙으면 글씨 일부가 잘린 것으로 판단
        clipped = (
            x1 <= YOLO_EDGE_MARGIN or
            y1 <= YOLO_EDGE_MARGIN or
            x2 >= fw - YOLO_EDGE_MARGIN or
            y2 >= fh - YOLO_EDGE_MARGIN
        )

        # 전체 프레임 -> ROI 상대 좌표
        rx1 = x1 - ox
        ry1 = y1 - oy
        rw = x2 - x1
        rh = y2 - y1

        cx = ((x1 + x2) // 2) - ox
        cy = ((y1 + y2) // 2) - oy

        candidates.append({
            "type": name,
            "bbox": (rx1, ry1, rw, rh),
            "frame_bbox": (x1, y1, x2, y2),
            "center": (cx, cy),
            "mask": None,
            "skeleton": None,
            "path": [],
            "follow_point": (cx, cy),
            "conf": conf,
            "clipped": clipped
        })

    if not candidates:
        return None

    # 화면 아래쪽 = 로봇에 가까운 글씨 우선
    return max(candidates, key=lambda z: z["center"][1])

def remove_yolo_text_from_black_mask(black_mask, text_target, ox, oy):
    """
    YOLO가 STOP/STATION으로 잡은 영역을 black_mask에서 완전히 제거한다.

    따라서 OpenCV 이진화는 글씨를 절대 목표로 쓰지 않고,
    화살표 검출에만 사용된다.
    """
    if text_target is None:
        return black_mask

    clean = black_mask.copy()

    x1, y1, x2, y2 = text_target["frame_bbox"]

    # 전체 프레임 좌표 -> ROI 좌표
    rx1 = max(0, x1 - ox)
    ry1 = max(0, y1 - oy)
    rx2 = min(clean.shape[1], x2 - ox)
    ry2 = min(clean.shape[0], y2 - oy)

    if rx2 > rx1 and ry2 > ry1:
        # bbox 주변에 약간의 여유를 두어 글자 조각까지 제거
        pad = 8
        rx1 = max(0, rx1 - pad)
        ry1 = max(0, ry1 - pad)
        rx2 = min(clean.shape[1], rx2 + pad)
        ry2 = min(clean.shape[0], ry2 + pad)

        clean[ry1:ry2, rx1:rx2] = 0

    return clean

# ============================================================
# Visualization
# ============================================================

def draw_target(vis, target, ox, oy):
    if target is None:
        return

    x,y,w,h = target["bbox"]
    cx,cy = target["center"]

    if target["type"] == "ARROW":
        color = (255,0,0)

        # 실제 1픽셀 skeleton을 화면에 굵게 표시
        skel = target["skeleton"]

        if skel is not None:
            sy, sx = np.where(skel > 0)

            for px, py in zip(sx, sy):
                xx = int(px + ox)
                yy = int(py + oy)

                if (
                    1 <= yy < vis.shape[0]-1
                    and 1 <= xx < vis.shape[1]-1
                ):
                    cv2.circle(
                        vis,
                        (xx,yy),
                        1,
                        (255,0,0),
                        -1
                    )

        # 실제 주행에 사용하는 중심점
        cv2.circle(
            vis,
            (cx+ox,cy+oy),
            7,
            (0,0,255),
            -1
        )

    else:
        color = (255,0,255)

        cv2.circle(
            vis,
            (cx+ox,cy+oy),
            7,
            (0,0,255),
            -1
        )

    cv2.rectangle(
        vis,
        (x+ox,y+oy),
        (x+w+ox,y+h+oy),
        color,
        2
    )

    label = target["type"]

    if "conf" in target:
        label = f'{label} {target["conf"]:.2f}'

    if target.get("clipped", False):
        label += " PARTIAL"

    cv2.putText(
        vis,
        label,
        (x+ox,max(20,y+oy-8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        .55,
        color,
        2
    )

# ============================================================
# State machine
# ============================================================

def control_loop():
    global state, auto_mode, latest_jpeg
    global manual_until, manual_cmd
    global ir_armed, ir_clear_count
    global lost_count, last_target
    global ir_armed, ir_clear_count
    global yolo_frame_count, last_yolo_text, yolo_miss_count
    global last_stable_yolo_text, yolo_partial_count

    ir_stop_time = 0.0

    while not stop_event.is_set():

        frame = camera.get_frame()

        if frame is None:
            time.sleep(0.02)
            continue

        frame = cv2.flip(frame, -1) if False else frame.copy()

        h, w = frame.shape[:2]
        roi, ox, oy = get_roi(frame)
        white_mask, black_mask, road_region = make_masks(roi)

        # ----------------------------------------------------
        # STOP/STATION: YOLO만 사용
        # ARROW       : OpenCV 이진화 + skeleton만 사용
        #
        # 중요:
        # YOLO가 잡은 글씨 영역은 black_mask에서 지운 뒤
        # 화살표 검출을 수행한다.
        # 따라서 글씨 이진화와 YOLO가 서로 충돌하지 않는다.
        # ----------------------------------------------------
        yolo_frame_count += 1

        if yolo_frame_count % YOLO_EVERY == 0:
            try:
                detected_text = detect_text_yolo(frame, ox, oy)

                if detected_text is not None:
                    yolo_miss_count = 0

                    if not detected_text.get("clipped", False):
                        # 글씨 전체가 화면 안에 들어온 정상 검출만
                        # 실제 조향 중심으로 저장
                        last_stable_yolo_text = detected_text
                        last_yolo_text = detected_text
                        yolo_partial_count = 0

                    else:
                        # 글씨가 잘린 상태:
                        # 보이는 조각의 중심으로 새로 조향하지 않는다.
                        yolo_partial_count += 1

                        if last_stable_yolo_text is not None:
                            # 직전 정상 bbox 중심 유지
                            last_yolo_text = last_stable_yolo_text
                        else:
                            # 아직 정상 bbox를 한 번도 못 봤으면
                            # 글씨를 목표로 쓰지 않고 계속 현재 주행 유지
                            last_yolo_text = None

                else:
                    yolo_miss_count += 1

                    # 잠깐의 YOLO 미검출은 이전 정상 bbox 유지
                    if yolo_miss_count >= YOLO_STABLE_HOLD:
                        last_yolo_text = None
                        last_stable_yolo_text = None
                        yolo_partial_count = 0

            except Exception as e:
                print("YOLO ERROR:", e)

        text_target = last_yolo_text

        # YOLO 글씨 영역을 이진화 마스크에서 제거
        arrow_black_mask = remove_yolo_text_from_black_mask(
            black_mask,
            text_target,
            ox,
            oy
        )

        # OpenCV 이진화는 이제 화살표만 검출
        arrow_target = detect_arrow_target(arrow_black_mask)

        # STOP/STATION이 보이면 YOLO 글씨를 우선 목표로 사용.
        # 글씨가 없을 때만 화살표를 사용.
        if text_target is not None:
            target = text_target
        elif arrow_target is not None:
            target = arrow_target
        else:
            target = None

        ir_l, ir_c, ir_r = ir.read_ir()
        ir_hit = (
            ir_l >= IR_THRESHOLD or
            ir_c >= IR_THRESHOLD or
            ir_r >= IR_THRESHOLD
        )

        # ----------------------------------------------------
        # IR 재활성화 로직
        # 한 번 검은 표식에서 멈춘 직후에는 IR을 잠시 무시한다.
        # 로봇이 그 검은 표식에서 완전히 벗어나 센서가 연속 몇 프레임
        # "흰 바닥"을 본 뒤에만 다시 IR 감지를 활성화한다.
        # ----------------------------------------------------
        if not ir_armed:
            if not ir_hit:
                ir_clear_count += 1
            else:
                ir_clear_count = 0

            if ir_clear_count >= IR_CLEAR_FRAMES:
                ir_armed = True
                ir_clear_count = 0

        # --------------------------------
        # Manual override
        # --------------------------------
        if time.time() < manual_until and manual_cmd is not None:
            l, r = manual_cmd
            drive(l, r)

        elif not auto_mode:
            stop_robot()

        else:
            # ====================================================
            # START
            # 검은 목표가 보일 때까지 직진
            # ====================================================
            if state == "START":
                if target is None:
                    drive(STRAIGHT_SPEED, STRAIGHT_SPEED)
                else:
                    last_target = target
                    state = "ALIGN"

            # ====================================================
            # ALIGN
            # 화살표 또는 YOLO STOP/STATION 중심을 화면 가운데로 정렬
            # ====================================================
            elif state == "ALIGN":
                if target is None:
                    lost_count += 1

                    if lost_count >= LOST_TARGET_FRAMES:
                        state = "SEARCH_RIGHT"
                        lost_count = 0
                else:
                    lost_count = 0
                    last_target = target

                    target_x = target["center"][0] + ox
                    error = target_x - (w/2)

                    if abs(error) <= CENTER_TOL:
                        stop_robot()
                        state = "FOLLOW"
                    else:
                        # 정렬은 제자리 회전
                        turn = int(np.clip(KP_ALIGN * error, -ALIGN_SPEED, ALIGN_SPEED))

                        if turn > 0:
                            drive(ALIGN_SPEED, -ALIGN_SPEED)
                        else:
                            drive(-ALIGN_SPEED, ALIGN_SPEED)

            # ====================================================
            # FOLLOW
            # 화살표는 skeleton 중심선 / STOP·STATION은 YOLO bbox 중심 추종
            # ====================================================
            elif state == "FOLLOW":
                if ir_armed and ir_hit:
                    # IR이 검은 표식을 처음 감지하면 바로 멈추지 않고
                    # 1.3초간 더 직진하여 표식을 완전히 통과
                    ir_armed = False
                    ir_clear_count = 0
                    ir_stop_time = time.time()
                    state = "IR_CONTINUE"

                elif target is not None:
                    last_target = target
                    target_x = target["center"][0] + ox
                    error = target_x - (w/2)

                    p_drive(
                        error,
                        FOLLOW_SPEED,
                        KP_FOLLOW,
                        MAX_CORR
                    )

                else:
                    drive(FOLLOW_SPEED, FOLLOW_SPEED)

            elif state == "IR_CONTINUE":
                # 기존 검은 글씨/화살표를 벗어나기 위해
                # IR 감지 후 1.3초간 추가 직진
                if time.time() - ir_stop_time < IR_AFTER_HIT_DRIVE_SEC:
                    drive(FOLLOW_SPEED, FOLLOW_SPEED)
                else:
                    stop_robot()
                    ir_stop_time = time.time()
                    state = "IR_STOP"

            # ====================================================
            # IR STOP
            # 실제 목표 위 도착
            # ====================================================
            elif state == "IR_STOP":
                stop_robot()

                if time.time() - ir_stop_time >= IR_STOP_SEC:
                    # 1.3초 더 주행해서 이전 표식을 벗어난 뒤이므로
                    # 다음 표식은 오른쪽 회전하면서 다시 탐색
                    state = "SEARCH_RIGHT"

            # ====================================================
            # SEARCH_RIGHT
            # 다음 목표가 안 보이면 오른쪽 제자리 회전
            # ====================================================
            elif state == "SEARCH_RIGHT":
                # 오른쪽으로 회전하면서 화살표 또는 YOLO STOP/STATION 탐색
                if target is not None:
                    stop_robot()
                    last_target = target
                    lost_count = 0
                    state = "ALIGN"
                else:
                    drive(SEARCH_SPEED, -SEARCH_SPEED)

        # ========================================================
        # Draw
        # ========================================================
        vis = frame.copy()

        # ROI
        cv2.rectangle(
            vis,
            (ox,oy),
            (w-ox,h-1),
            (0,255,0),
            2
        )

        # screen center
        cv2.line(
            vis,
            (w//2,0),
            (w//2,h),
            (255,255,255),
            1
        )

        draw_target(vis, target, ox, oy)

        shown_state = state if auto_mode else f"PAUSED / {state}"

        cv2.putText(
            vis,
            f"STATE: {shown_state}",
            (12,28),
            cv2.FONT_HERSHEY_SIMPLEX,
            .65,
            (0,255,255),
            2
        )

        cv2.putText(
            vis,
            f"AUTO: {auto_mode}",
            (12,55),
            cv2.FONT_HERSHEY_SIMPLEX,
            .55,
            (0,255,255),
            2
        )

        cv2.putText(
            vis,
            f"IR L:{ir_l} C:{ir_c} R:{ir_r}",
            (12,82),
            cv2.FONT_HERSHEY_SIMPLEX,
            .50,
            (0,255,255),
            2
        )

        cv2.putText(
            vis,
            f"IR_ARMED: {ir_armed}",
            (12,108),
            cv2.FONT_HERSHEY_SIMPLEX,
            .48,
            (0,255,255),
            2
        )

        # binary preview
        binary_bgr = cv2.cvtColor(arrow_black_mask, cv2.COLOR_GRAY2BGR)
        binary_bgr = cv2.resize(
            binary_bgr,
            (vis.shape[1], vis.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        combo = np.hstack([vis, binary_bgr])

        ok, jpg = cv2.imencode(
            ".jpg",
            combo,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
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
<title>Pinky Vision</title>
<style>
body{background:#111;color:white;font-family:Arial;text-align:center}
img{width:95%;max-width:1200px;border:2px solid #555}
button{font-size:24px;margin:5px;padding:10px 20px}
</style>
</head>
<body>
<h2>Pinky Arrow(OpenCV) + STOP/STATION(YOLO)</h2>
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
document.addEventListener('keydown',function(e){
    if(e.repeat) return;

    let k=e.key.toLowerCase();

    if(e.code==='Space'){
        e.preventDefault();
        cmd('space');
    }
    else if(['w','a','s','d','p','r'].includes(k)){
        cmd(k);
    }
});
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
                b'Content-Type: image/jpeg\r\n\r\n' +
                data +
                b'\r\n'
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
    global auto_mode, state
    global manual_until, manual_cmd
    global ir_armed, ir_clear_count

    if key == "p":
        auto_mode = not auto_mode

        if auto_mode:
            state = "START"
        else:
            stop_robot()

    elif key == "r":
        auto_mode = False
        state = "START"
        ir_armed = True
        ir_clear_count = 0
        stop_robot()

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
