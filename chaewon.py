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
# Pinky Pro - 화살표 Blob 중심(OpenCV) + STOP/STATION(YOLO) 통합 주행
#
# 동작
# 1) 시작: 도로 이진화만 하면서 직진
# 2) 화살표는 OpenCV, STOP/STATION은 YOLO로 검출 -> 화면 중앙 정렬
# 3) 갈라진 검은 조각을 그룹화한 뒤 solid Blob으로 뭉개고 무게중심 추종
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
YOLO_CONF = 0.45          # 실제 STOP/STATION 주행 목표로 사용할 신뢰도
YOLO_MASK_CONF = 0.20     # 글씨를 Blob에서 제외하기 위한 낮은 마스킹 신뢰도
YOLO_IMGSZ = 320
YOLO_EVERY = 1          # 글씨 bbox를 매 프레임 갱신
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
PARTIAL_BLOB_MIN_AREA = 420   # 화면에 약 2/3 정도만 보여도 허용
BLOB_GROUP_GAP = 35           # 반사로 갈라진 조각을 같은 화살표로 묶는 최대 간격(px)
BLOB_EDGE_MARGIN = 5          # 화면 경계에 닿으면 잘린 Blob으로 판단
BLOB_GONE_FRAMES = 10         # 같은 화살표 재카운트 방지: 충분히 사라진 뒤 다음 카운트 허용
CENTER_TOL = 30

# Control
KP_ALIGN = 0.11
KP_FOLLOW = 0.10
MAX_CORR = 10

# Search behavior
IR_STOP_SEC = 0.45
IR_AFTER_HIT_DRIVE_SEC = 1.0   # STOP/STATION 이후 기존 동작용
IR_CLEAR_FRAMES = 4            # 검은 표식에서 벗어난 것이 연속 4프레임 확인되면 IR 재활성화
LOST_TARGET_FRAMES = 5

# ------------------------------------------------------------
# 3번째 화살표 기반 우회전 타이밍
# ------------------------------------------------------------
# 3번째 화살표는 IR로 밟지 않는다.
# 화살표 bbox의 아래쪽 끝(y+h)이 ROI 높이의 이 비율까지 내려오면 우회전을 시작한다.
# 더 일찍 돌고 싶으면 값을 낮추고(예: 0.68), 더 늦게 돌고 싶으면 높인다(예: 0.80).
THIRD_ARROW_TURN_RATIO = 0.75

# bbox 흔들림 때문에 한 프레임만 잘못 내려오는 것을 막기 위해
# 연속 N프레임 기준선을 넘었을 때만 실제 우회전한다.
THIRD_ARROW_TRIGGER_FRAMES = 2

# 2번째 화살표를 IR로 밟은 직후 화면 아래쪽에 남아 있던 2번 화살표가 사라지고
# 더 먼 3번 화살표로 대상이 바뀌면 bbox 하단이 위쪽으로 크게 점프한다.
# ROI 높이의 이 비율만큼 위로 점프하면 새 3번 화살표로 인정한다.
THIRD_ARROW_SWITCH_JUMP_RATIO = 0.10

# 3번째 화살표를 향해 접근할 때는 너무 빠르게 가지 않도록 별도 속도 사용
THIRD_ARROW_SPEED = 17
THIRD_ARROW_KP = 0.08
THIRD_ARROW_MAX_CORR = 7

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
YOLO_STABLE_HOLD = 1
yolo_partial_count = 0
last_text_mask_targets = []

# STOP / STATION bbox 흔들림 보정
TEXT_BBOX_ALPHA = 0.60       # 현재 검출값을 더 많이 반영해 주행 중 bbox가 따라오게 함
TEXT_CENTER_DEADBAND = 3     # 아주 작은 흔들림만 무시
TEXT_MAX_JUMP = 180          # 실제 이동은 허용하고 비정상적인 큰 점프만 차단

# ----------------------------
# 화살표 Blob 카운트
# ----------------------------
blob_count = 0

# 같은 화살표를 여러 번 세지 않기 위한 상태
# count_armed=True일 때만 새 Blob을 카운트한다.
blob_count_armed = True

# 한 번 카운트한 Blob은 IR을 실제로 밟은 뒤,
# 그 Blob이 화면에서 충분히 사라져야 다음 카운트를 허용한다.
blob_ir_passed = False
blob_missing_frames = 0

ir_blob_count = 0

# 오른쪽 회전 허용 플래그
search_right_allowed = False

# 3번째 화살표 하단이 기준선을 넘은 연속 프레임 수
third_arrow_trigger_count = 0

# 2번째 화살표 IR 순간의 bbox 하단 위치.
# 2번 -> 3번 화살표 전환을 구분하는 보조 기준으로 사용.
second_ir_arrow_bottom = None

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

def _bbox_gap(a, b):
    """두 bbox 사이의 x/y 최소 간격."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    dx = max(0, max(ax, bx) - min(ax2, bx2))
    dy = max(0, max(ay, by) - min(ay2, by2))

    return dx, dy


def _group_near_contours(contours):
    """
    빛반사로 하나의 화살표가 여러 조각으로 갈라져도
    서로 가까운 contour끼리 같은 Blob 그룹으로 묶는다.
    모폴로지 연산은 사용하지 않는다.
    """
    items = []

    for c in contours:
        area = cv2.contourArea(c)

        # 아주 작은 점 노이즈만 제거
        if area < 45:
            continue

        items.append({
            "contour": c,
            "bbox": cv2.boundingRect(c)
        })

    groups = []
    used = [False] * len(items)

    for i in range(len(items)):
        if used[i]:
            continue

        used[i] = True
        group = [items[i]]
        changed = True

        # 연결되는 조각이 더 없을 때까지 확장
        while changed:
            changed = False

            # 현재 그룹 전체 bbox
            xs = [g["bbox"][0] for g in group]
            ys = [g["bbox"][1] for g in group]
            x2s = [g["bbox"][0] + g["bbox"][2] for g in group]
            y2s = [g["bbox"][1] + g["bbox"][3] for g in group]

            gb = (
                min(xs),
                min(ys),
                max(x2s) - min(xs),
                max(y2s) - min(ys)
            )

            for j in range(len(items)):
                if used[j]:
                    continue

                dx, dy = _bbox_gap(gb, items[j]["bbox"])

                if dx <= BLOB_GROUP_GAP and dy <= BLOB_GROUP_GAP:
                    used[j] = True
                    group.append(items[j])
                    changed = True

        groups.append(group)

    return groups


def detect_arrow_target(black_mask):
    """
    화살표 모양 자체는 사용하지 않는다.

    1) 이진화된 검은 조각들을 찾는다.
    2) 서로 가까운 조각들을 하나의 그룹으로 묶는다.
    3) 그 그룹 전체를 감싸는 사각 영역을 하나의 'solid blob'으로 만든다.
       -> 실제 화살표 윤곽/머리 모양은 버린다.
    4) solid blob에 cv2.moments()를 적용해서 무게중심만 사용한다.
    5) 화면 가장자리에서 일부가 잘려도 충분한 면적이 있으면 허용한다.
    """
    contours, _ = cv2.findContours(
        black_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    groups = _group_near_contours(contours)

    if not groups:
        return None

    H, W = black_mask.shape[:2]
    candidates = []

    for group in groups:
        # 그룹 전체 조각의 좌표를 모아서 전체 bbox 계산
        xs = []
        ys = []
        x2s = []
        y2s = []
        visible_area = 0

        for item in group:
            x, y, w, h = item["bbox"]
            xs.append(x)
            ys.append(y)
            x2s.append(x + w)
            y2s.append(y + h)
            visible_area += int(cv2.contourArea(item["contour"]))

        if not xs:
            continue

        x1 = max(0, min(xs))
        y1 = max(0, min(ys))
        x2 = min(W, max(x2s))
        y2 = min(H, max(y2s))

        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            continue

        # YOLO가 글씨를 놓친 경우의 보조 안전장치:
        # 매우 가로로 긴 검은 덩어리는 STOP/STATION 글자열일 가능성이 높으므로
        # 화살표 Blob 후보에서 제외한다.
        aspect = w / max(h, 1)
        if aspect > 2.8:
            continue

        touches_edge = (
            x1 <= BLOB_EDGE_MARGIN or
            y1 <= BLOB_EDGE_MARGIN or
            x2 >= W - BLOB_EDGE_MARGIN or
            y2 >= H - BLOB_EDGE_MARGIN
        )

        min_area = (
            PARTIAL_BLOB_MIN_AREA
            if touches_edge
            else MIN_ARROW_AREA
        )

        if visible_area < min_area:
            continue

        # ----------------------------------------------------
        # 핵심:
        # 화살표 실제 모양을 버리고
        # 그룹 전체 영역을 하나의 꽉 찬 덩어리로 만들어버림.
        # ----------------------------------------------------
        solid_blob = np.zeros_like(black_mask)

        cv2.rectangle(
            solid_blob,
            (x1, y1),
            (x2 - 1, y2 - 1),
            255,
            -1
        )

        # solid blob의 모멘트
        M = cv2.moments(solid_blob, binaryImage=True)

        if abs(M["m00"]) < 1e-6:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        candidates.append({
            "type": "ARROW",
            "bbox": (x1, y1, w, h),
            "center": (cx, cy),
            "mask": solid_blob,
            "skeleton": None,
            "path": [],
            "follow_point": (cx, cy),
            "blob_area": int(M["m00"]),
            "partial": touches_edge
        })

    if not candidates:
        return None

    # 화면 아래쪽에 있는 덩어리를 우선
    return max(
        candidates,
        key=lambda z: (z["bbox"][1] + z["bbox"][3], z["blob_area"])
    )


def stabilize_text_target(prev, current):
    """
    STOP/STATION bbox를 '고정'하지 않고 현재 프레임을 계속 따라가게 한다.

    - 매 프레임 YOLO 현재 bbox를 입력으로 사용
    - 작은 떨림만 deadband로 제거
    - EMA는 약하게만 적용
    - 실제 로봇 이동에 따른 bbox 이동은 즉시 반영
    """
    if current is None:
        return None

    if prev is None:
        return current.copy()

    # 다른 클래스면 새 목표로 즉시 전환
    if prev.get("type") != current.get("type"):
        return current.copy()

    pcx, pcy = prev["center"]
    ccx, ccy = current["center"]

    dx = ccx - pcx
    dy = ccy - pcy
    jump = (dx * dx + dy * dy) ** 0.5

    # 아주 비정상적으로 큰 순간 점프만 차단
    if jump > TEXT_MAX_JUMP:
        return prev

    # 실제 이동은 현재값을 많이 반영
    if abs(dx) <= TEXT_CENTER_DEADBAND:
        scx = pcx
    else:
        scx = int((1.0 - TEXT_BBOX_ALPHA) * pcx + TEXT_BBOX_ALPHA * ccx)

    if abs(dy) <= TEXT_CENTER_DEADBAND:
        scy = pcy
    else:
        scy = int((1.0 - TEXT_BBOX_ALPHA) * pcy + TEXT_BBOX_ALPHA * ccy)

    px, py, pw, ph = prev["bbox"]
    cx, cy, cw, ch = current["bbox"]

    sx = int((1.0 - TEXT_BBOX_ALPHA) * px + TEXT_BBOX_ALPHA * cx)
    sy = int((1.0 - TEXT_BBOX_ALPHA) * py + TEXT_BBOX_ALPHA * cy)
    sw = int((1.0 - TEXT_BBOX_ALPHA) * pw + TEXT_BBOX_ALPHA * cw)
    sh = int((1.0 - TEXT_BBOX_ALPHA) * ph + TEXT_BBOX_ALPHA * ch)

    out = current.copy()
    out["center"] = (scx, scy)
    out["follow_point"] = (scx, scy)
    out["bbox"] = (sx, sy, sw, sh)

    # 마스킹용 frame_bbox도 현재 위치를 따라가게 갱신
    if "frame_bbox" in current:
        out["frame_bbox"] = current["frame_bbox"]

    return out

def detect_text_yolo(frame, ox, oy):
    """
    STOP / STATION은 YOLO로 검출한다.

    반환:
      primary_target : YOLO_CONF 이상인 STOP/STATION 중 주행에 사용할 목표 1개
      mask_targets   : YOLO_MASK_CONF 이상인 모든 STOP/STATION bbox
                       -> 이진화 black_mask에서 글씨를 제거하는 용도

    이렇게 분리하는 이유:
    글씨가 주행 목표로 쓰일 만큼 확신이 높지 않더라도,
    검은 글자가 화살표 Blob으로 오인되는 것은 막기 위해서다.
    """
    results = yolo_model(
        frame,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_MASK_CONF,
        verbose=False
    )

    steering_candidates = []
    mask_targets = []

    if not results:
        return None, []

    r = results[0]

    if r.boxes is None:
        return None, []

    fh, fw = frame.shape[:2]

    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = str(yolo_model.names[cls_id]).upper().strip()

        if name not in TEXT_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        clipped = (
            x1 <= YOLO_EDGE_MARGIN or
            y1 <= YOLO_EDGE_MARGIN or
            x2 >= fw - YOLO_EDGE_MARGIN or
            y2 >= fh - YOLO_EDGE_MARGIN
        )

        rx1 = x1 - ox
        ry1 = y1 - oy
        rw = x2 - x1
        rh = y2 - y1

        cx = ((x1 + x2) // 2) - ox
        cy = ((y1 + y2) // 2) - oy

        item = {
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
        }

        # 낮은 신뢰도라도 글씨 영역 마스킹에는 사용
        mask_targets.append(item)

        # 실제 주행 목표는 기존 YOLO_CONF 이상만 사용
        if conf >= YOLO_CONF:
            steering_candidates.append(item)

    if not steering_candidates:
        return None, mask_targets

    # 화면에서 더 아래쪽 = 더 가까운 글씨 우선
    primary_target = max(
        steering_candidates,
        key=lambda z: z["bbox"][1] + z["bbox"][3]
    )

    return primary_target, mask_targets

def remove_yolo_text_from_black_mask(black_mask, text_targets, ox, oy):
    """
    YOLO가 STOP/STATION으로 본 모든 영역을 black_mask에서 제거한다.

    주행 목표가 될 만큼 confidence가 높지 않아도
    YOLO_MASK_CONF 이상이면 글씨를 Blob 후보에서 제외한다.
    따라서 STATION/STOP 글자가 화살표 solid Blob으로 뭉쳐지는 것을 방지한다.
    """
    clean = black_mask.copy()

    if not text_targets:
        return clean

    for target in text_targets:
        x1, y1, x2, y2 = target["frame_bbox"]

        rx1 = max(0, x1 - ox)
        ry1 = max(0, y1 - oy)
        rx2 = min(clean.shape[1], x2 - ox)
        ry2 = min(clean.shape[0], y2 - oy)

        if rx2 <= rx1 or ry2 <= ry1:
            continue

        # 글자 끝부분까지 확실히 제거
        pad = 12
        rx1 = max(0, rx1 - pad)
        ry1 = max(0, ry1 - pad)
        rx2 = min(clean.shape[1], rx2 + pad)
        ry2 = min(clean.shape[0], ry2 + pad)

        clean[ry1:ry2, rx1:rx2] = 0

    return clean

# ============================================================
# Visualization
# ============================================================

# ============================================================

def draw_target(vis, target, ox, oy):
    if target is None:
        return

    cx, cy = target["center"]

    if target["type"] == "ARROW":
        x, y, w, h = target["bbox"]

        # 화살표 모양이 아니라 하나의 solid blob으로 표시
        overlay = vis.copy()

        cv2.rectangle(
            overlay,
            (x + ox, y + oy),
            (x + w + ox, y + h + oy),
            (255, 0, 0),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.28,
            vis,
            0.72,
            0,
            vis
        )

        cv2.rectangle(
            vis,
            (x + ox, y + oy),
            (x + w + ox, y + h + oy),
            (255, 0, 0),
            3
        )

        # solid blob 무게중심
        cv2.circle(
            vis,
            (cx + ox, cy + oy),
            9,
            (0, 0, 255),
            -1
        )

        label = f"BLOB {int(target.get('blob_area', 0))}"

        if target.get("partial", False):
            label += " PARTIAL"

        cv2.putText(
            vis,
            label,
            (x + ox, max(25, y + oy - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            .6,
            (255, 0, 0),
            2
        )

    else:
        x, y, w, h = target["bbox"]
        color = (255, 0, 255)

        cv2.rectangle(
            vis,
            (x + ox, y + oy),
            (x + w + ox, y + h + oy),
            color,
            2
        )

        cv2.circle(
            vis,
            (cx + ox, cy + oy),
            7,
            (0, 0, 255),
            -1
        )

        label = target["type"]

        if "conf" in target:
            label = f'{label} {target["conf"]:.2f}'

        if target.get("clipped", False):
            label += " PARTIAL"

        cv2.putText(
            vis,
            label,
            (x + ox, max(20, y + oy - 8)),
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
    global last_stable_yolo_text, yolo_partial_count, last_text_mask_targets
    global blob_count, blob_count_armed, blob_ir_passed, blob_missing_frames, ir_blob_count
    global search_right_allowed, third_arrow_trigger_count, second_ir_arrow_bottom

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
        detected_text_masks = []

        if yolo_frame_count % YOLO_EVERY == 0:
            try:
                detected_text, detected_text_masks = detect_text_yolo(frame, ox, oy)
                last_text_mask_targets = detected_text_masks

                if detected_text is not None:
                    yolo_miss_count = 0

                    # 현재 프레임의 bbox를 계속 따라가도록 갱신
                    last_stable_yolo_text = stabilize_text_target(
                        last_stable_yolo_text,
                        detected_text
                    )
                    last_yolo_text = last_stable_yolo_text

                    if detected_text.get("clipped", False):
                        yolo_partial_count += 1
                    else:
                        yolo_partial_count = 0

                else:
                    # 이번 프레임에서 글씨를 못 잡으면
                    # 이전 위치를 붙잡아 두지 않고 즉시 해제
                    yolo_miss_count += 1
                    last_yolo_text = None
                    last_stable_yolo_text = None
                    yolo_partial_count = 0

            except Exception as e:
                print("YOLO ERROR:", e)

        text_target = last_yolo_text

        # YOLO 글씨 영역을 이진화 마스크에서 제거
        arrow_black_mask = remove_yolo_text_from_black_mask(
            black_mask,
            last_text_mask_targets,
            ox,
            oy
        )

        # 이진화 결과에서 화살표 Blob 검출
        arrow_target = detect_arrow_target(arrow_black_mask)

        # ----------------------------------------------------
        # 화살표 Blob 카운팅
        #
        # 핵심:
        # 1) 새 Blob이 처음 잡히면 딱 1번만 카운트
        # 2) 같은 화살표가 흔들리거나 잠깐 끊겨 보여도 재카운트 금지
        # 3) 그 Blob을 따라가서 IR 센서를 실제로 밟은 뒤에만
        #    다음 카운트를 준비할 수 있음
        # 4) IR이 다시 clear되어 ir_armed=True가 되고,
        #    기존 Blob도 충분한 프레임 동안 완전히 사라져야
        #    다음 Blob 카운트를 허용
        # ----------------------------------------------------
        if arrow_target is not None:
            blob_missing_frames = 0

            if blob_count_armed:
                blob_count += 1
                blob_count_armed = False
                blob_ir_passed = False
                print(f"[BLOB COUNT] {blob_count}")

        else:
            # 다음 카운트는 반드시:
            #   이전 Blob IR 통과 완료
            #   + IR 센서 clear 후 재활성화
            #   + 이전 Blob이 화면에서 완전히 사라짐
            # 세 조건을 모두 만족해야 허용한다.
            if (
                blob_ir_passed
                and ir_armed
                and not blob_count_armed
            ):
                blob_missing_frames += 1

                if blob_missing_frames >= BLOB_GONE_FRAMES:
                    blob_count_armed = True
                    blob_ir_passed = False
                    blob_missing_frames = 0
                    print("[BLOB] previous blob fully passed -> next count armed")
            else:
                # 조건이 아직 안 됐으면 누적하지 않음
                blob_missing_frames = 0

        # ----------------------------------------------------
        # 목표 선택
        #
        # 평상시:
        #   STOP / STATION / 화살표 Blob 중 더 가까운 목표 선택
        #
        # 단, Blob을 이미 카운트했고 아직 그 Blob의 IR을 밟지 않았다면:
        #   그 순간부터는 현재 Blob만 추종
        #   STOP/STATION은 보여도 무시
        #
        # 따라서 2번째 Blob을 인식한 뒤 IR을 밟기 전에
        # 앞쪽 STOP이 보여도 STOP으로 목표가 바뀌지 않는다.
        # ----------------------------------------------------
        waiting_for_blob_ir = (
            blob_count >= 1
            and not blob_count_armed
            and not blob_ir_passed
        )

        # 오른쪽 교차로 진입 중에는 STOP만 사용한다.
        # STATION이나 남아 있는 3번째 화살표 때문에 회전이 끊기는 것을 막는다.
        stop_target = None
        if text_target is not None and text_target.get("type") == "STOP":
            stop_target = text_target

        if state in ("FIRST_BLOB_STRAIGHT", "SECOND_BLOB_STRAIGHT", "THIRD_ARROW_APPROACH"):
            # 1번 통과 후 2번 탐색 / 2번 통과 후 3번 탐색 / 3번 접근 중에는
            # STOP/STATION을 무시하고 화살표만 본다.
            target = arrow_target

        elif state in ("SEARCH_STOP_RIGHT", "ALIGN_STOP"):
            # 교차로 우회전 구간에서는 오직 STOP만 목표로 인정한다.
            target = stop_target

        elif waiting_for_blob_ir:
            # 1번/2번 화살표는 IR을 실제로 밟기 전까지 화살표만 추종한다.
            # 3번 화살표는 THIRD_ARROW_APPROACH 상태에서 IR 대신 화면 기준선을 사용한다.
            target = arrow_target

        else:
            # 그 외 구간은 기존처럼 화면에서 더 가까운 목표 선택
            visible_targets = []

            if text_target is not None:
                visible_targets.append(text_target)

            if arrow_target is not None:
                visible_targets.append(arrow_target)

            if visible_targets:
                target = max(
                    visible_targets,
                    key=lambda z: z["bbox"][1] + z["bbox"][3]
                )
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
                        # IR을 밟기 전에는 절대 우회전 탐색으로 가지 않는다.
                        # 목표를 잠깐 놓치면 그대로 전진하며 다시 찾는다.
                        state = "FOLLOW"
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
            # 화살표는 가장 큰 Blob 무게중심 / STOP·STATION은 YOLO bbox 중심 추종
            # ====================================================
            elif state == "FOLLOW":
                if ir_armed and ir_hit:
                    # IR을 실제로 밟기 전에는 회전 허용 안 함
                    search_right_allowed = False

                    # ------------------------------------------------
                    # 중요:
                    # Blob이 화면에 잡혀 카운트된 것만으로는 절대 회전하지 않는다.
                    # 반드시 그 Blob을 따라간 뒤 IR 센서를 실제로 밟아야
                    # 다음 단계로 넘어간다.
                    # ------------------------------------------------
                    ir_blob_count = blob_count
                    blob_ir_passed = True

                    ir_armed = False
                    ir_clear_count = 0

                    # ------------------------------------------------
                    # 첫 번째 Blob:
                    # IR을 밟아도 정지/회전하지 않고 그대로 직진.
                    # 기존 Blob이 사라진 뒤 다음 Blob 카운트를 허용한다.
                    # ------------------------------------------------
                    if ir_blob_count == 1:
                        state = "FIRST_BLOB_STRAIGHT"

                    # ------------------------------------------------
                    # 두 번째 Blob:
                    # IR을 밟은 뒤에는 아직 우회전하지 않는다.
                    # 그대로 직진하며 3번째 화살표를 찾는다.
                    # ------------------------------------------------
                    elif ir_blob_count == 2:
                        third_arrow_trigger_count = 0

                        # 2번째 화살표를 실제로 밟은 순간의 bbox 하단 위치 저장.
                        # 이후 화면 아래의 2번 화살표가 빠지고, 더 먼 3번 화살표가
                        # 선택되면서 bbox 하단이 위로 점프하는 것을 이용한다.
                        if arrow_target is not None:
                            ax, ay, aw, ah = arrow_target["bbox"]
                            second_ir_arrow_bottom = ay + ah
                        else:
                            second_ir_arrow_bottom = None

                        state = "SECOND_BLOB_STRAIGHT"

                    # 3번째 화살표는 원칙적으로 IR을 밟기 전에 화면 기준선에서
                    # 우회전해야 하므로 여기까지 들어오면 너무 늦은 상황이다.
                    # 안전하게 기존 후속 동작으로 넘긴다.
                    elif ir_blob_count >= 3:
                        ir_stop_time = time.time()
                        state = "IR_CONTINUE"

                    else:
                        drive(FOLLOW_SPEED, FOLLOW_SPEED)

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

            # ====================================================
            # FIRST BLOB STRAIGHT
            # 첫 번째 Blob에서 IR을 밟은 뒤에는 회전하지 않고 직진.
            # 기존 Blob이 사라지면 카운트가 다시 활성화되고,
            # 다음 Blob이 보일 때 BLOB_COUNT가 2가 된다.
            # ====================================================
            elif state == "FIRST_BLOB_STRAIGHT":
                # 첫 번째 Blob의 IR을 통과한 뒤에는
                # 반드시 "다음 Blob"을 먼저 찾아야 한다.
                # 멀리 STOP/STATION이 보여도 이 단계에서는 전부 무시한다.
                drive(STRAIGHT_SPEED, STRAIGHT_SPEED)

                # 두 번째 Blob이 실제로 새로 카운트되고 보일 때만
                # 그 Blob으로 ALIGN/FOLLOW 한다.
                if blob_count >= 2 and arrow_target is not None:
                    stop_robot()
                    last_target = arrow_target
                    lost_count = 0
                    state = "ALIGN"

            # ====================================================
            # SECOND BLOB STRAIGHT
            # 2번째 화살표를 IR로 통과한 뒤에는 우회전하지 않는다.
            # 직진하며 3번째 화살표가 새 Blob으로 카운트될 때까지 기다린다.
            # ====================================================
            elif state == "SECOND_BLOB_STRAIGHT":
                drive(STRAIGHT_SPEED, STRAIGHT_SPEED)

                # 기존 방식처럼 화살표 사이에 빈 프레임이 충분하면 blob_count가 3이 된다.
                # 하지만 2번과 3번이 카메라에 연속해서 보일 수도 있으므로,
                # 2번 IR 순간의 bbox 하단보다 현재 화살표 하단이 크게 위로 점프하면
                # '가까운 2번이 빠지고 먼 3번으로 대상이 바뀌었다'고 판단한다.
                third_arrow_ready = False

                if arrow_target is not None:
                    ax, ay, aw, ah = arrow_target["bbox"]
                    current_bottom = ay + ah
                    switch_jump_px = int(roi.shape[0] * THIRD_ARROW_SWITCH_JUMP_RATIO)

                    if blob_count >= 3:
                        third_arrow_ready = True

                    elif (
                        ir_armed
                        and second_ir_arrow_bottom is not None
                        and current_bottom <= second_ir_arrow_bottom - switch_jump_px
                    ):
                        # 빈 프레임 없이 바로 3번이 나타난 경우 수동으로 3번째 카운트 확정
                        blob_count = 3
                        blob_count_armed = False
                        blob_ir_passed = False
                        third_arrow_ready = True
                        print(
                            f"[BLOB COUNT] 3 (switch detected: "
                            f"{second_ir_arrow_bottom} -> {current_bottom})"
                        )

                if third_arrow_ready:
                    third_arrow_trigger_count = 0
                    last_target = arrow_target
                    state = "THIRD_ARROW_APPROACH"
                    print("[3RD ARROW] detected -> visual turn trigger active")

            # ====================================================
            # THIRD ARROW APPROACH
            # 3번째 화살표는 IR로 밟지 않는다.
            # bbox의 아래쪽 끝이 가상의 TURN LINE에 도달하면 즉시 우회전 단계로 간다.
            # ====================================================
            elif state == "THIRD_ARROW_APPROACH":
                if arrow_target is not None:
                    last_target = arrow_target

                    ax, ay, aw, ah = arrow_target["bbox"]
                    arrow_bottom = ay + ah
                    turn_line = int(roi.shape[0] * THIRD_ARROW_TURN_RATIO)

                    # 기준선에 닿기 전까지는 3번째 화살표 중심을 약하게 추종하며 접근
                    target_x = arrow_target["center"][0] + ox
                    error = target_x - (w / 2)

                    if arrow_bottom >= turn_line:
                        third_arrow_trigger_count += 1
                    else:
                        third_arrow_trigger_count = 0

                    if third_arrow_trigger_count >= THIRD_ARROW_TRIGGER_FRAMES:
                        stop_robot()
                        search_right_allowed = True

                        # 3번째 화살표는 IR 대신 시각 트리거로 완료 처리.
                        # 이후 화살표가 사라지면 다음 Blob을 다시 셀 수 있게 한다.
                        blob_ir_passed = True
                        third_arrow_trigger_count = 0

                        state = "SEARCH_STOP_RIGHT"
                        print(
                            f"[3RD ARROW] bottom={arrow_bottom}, "
                            f"turn_line={turn_line} -> TURN RIGHT"
                        )
                    else:
                        p_drive(
                            error,
                            THIRD_ARROW_SPEED,
                            THIRD_ARROW_KP,
                            THIRD_ARROW_MAX_CORR
                        )

                else:
                    # 잠깐 검출을 놓쳐도 급하게 돌지 않고 천천히 직진하며 다시 찾음
                    third_arrow_trigger_count = 0
                    drive(THIRD_ARROW_SPEED, THIRD_ARROW_SPEED)

            # ====================================================
            # SEARCH STOP RIGHT
            # 3번째 화살표 기준선에서 오른쪽 제자리 회전.
            # 이때는 남아 있는 화살표와 STATION을 모두 무시하고 STOP만 찾는다.
            # ====================================================
            elif state == "SEARCH_STOP_RIGHT":
                if not search_right_allowed:
                    stop_robot()
                    search_right_allowed = True

                if stop_target is not None:
                    stop_robot()
                    last_target = stop_target
                    lost_count = 0
                    state = "ALIGN_STOP"
                    print("[STOP] detected during right turn -> align")
                else:
                    drive(SEARCH_SPEED, -SEARCH_SPEED)

            # ====================================================
            # ALIGN STOP
            # STOP bbox 중심이 화면 중앙에 올 때까지 제자리 회전으로 미세 정렬.
            # 정렬 완료 뒤 STOP을 향해 직진 추종한다.
            # ====================================================
            elif state == "ALIGN_STOP":
                if stop_target is None:
                    # STOP을 순간적으로 놓치면 다시 오른쪽 탐색으로 복귀
                    lost_count += 1
                    if lost_count >= LOST_TARGET_FRAMES:
                        lost_count = 0
                        state = "SEARCH_STOP_RIGHT"
                else:
                    lost_count = 0
                    last_target = stop_target

                    target_x = stop_target["center"][0] + ox
                    error = target_x - (w / 2)

                    if abs(error) <= CENTER_TOL:
                        stop_robot()
                        search_right_allowed = False
                        state = "FOLLOW"
                        print("[STOP] centered -> follow straight")
                    else:
                        if error > 0:
                            drive(ALIGN_SPEED, -ALIGN_SPEED)
                        else:
                            drive(-ALIGN_SPEED, ALIGN_SPEED)

            # ====================================================
            # 기존 IR 후속 상태
            # 이후 경로에서 필요할 수 있으므로 기존 동작은 남겨둔다.
            # ====================================================
            elif state == "IR_CONTINUE":
                if time.time() - ir_stop_time < IR_AFTER_HIT_DRIVE_SEC:
                    drive(FOLLOW_SPEED, FOLLOW_SPEED)
                else:
                    stop_robot()
                    ir_stop_time = time.time()
                    state = "IR_STOP"

            elif state == "IR_STOP":
                stop_robot()

                if time.time() - ir_stop_time >= IR_STOP_SEC:
                    search_right_allowed = True
                    state = "SEARCH_RIGHT"

            elif state == "SEARCH_RIGHT":
                if not search_right_allowed:
                    drive(STRAIGHT_SPEED, STRAIGHT_SPEED)
                    state = "FOLLOW"

                elif target is not None:
                    stop_robot()
                    last_target = target
                    lost_count = 0
                    search_right_allowed = False
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

        # 3번째 화살표 우회전 기준선 (ROI 내부 좌표 -> 전체 프레임 좌표)
        third_turn_y = oy + int(roi.shape[0] * THIRD_ARROW_TURN_RATIO)
        cv2.line(
            vis,
            (ox, third_turn_y),
            (w-ox, third_turn_y),
            (0, 0, 255),
            2
        )
        cv2.putText(
            vis,
            f"3RD TURN LINE {THIRD_ARROW_TURN_RATIO:.2f}",
            (ox + 8, max(20, third_turn_y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            .48,
            (0, 0, 255),
            2
        )

        # 3번째 화살표 접근 중에는 bbox 하단 위치도 표시
        if state == "THIRD_ARROW_APPROACH" and arrow_target is not None:
            ax, ay, aw, ah = arrow_target["bbox"]
            arrow_bottom_frame = oy + ay + ah
            cv2.circle(
                vis,
                (ax + aw // 2 + ox, arrow_bottom_frame),
                7,
                (0, 255, 255),
                -1
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
            f"BLOB_COUNT: {blob_count}",
            (12,82),
            cv2.FONT_HERSHEY_SIMPLEX,
            .50,
            (0,255,255),
            2
        )

        cv2.putText(
            vis,
            f"IR L:{ir_l} C:{ir_c} R:{ir_r}",
            (12,108),
            cv2.FONT_HERSHEY_SIMPLEX,
            .50,
            (0,255,255),
            2
        )

        cv2.putText(
            vis,
            f"IR_ARMED: {ir_armed}",
            (12,134),
            cv2.FONT_HERSHEY_SIMPLEX,
            .48,
            (0,255,255),
            2
        )

        # 오른쪽 화면에는 실제로 인식 중인 화살표 Blob 마스크 표시
        if arrow_target is not None and arrow_target.get("mask") is not None:
            preview_mask = arrow_target["mask"]
        else:
            preview_mask = arrow_black_mask

        binary_bgr = cv2.cvtColor(preview_mask, cv2.COLOR_GRAY2BGR)
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
<h2>Pinky Solid BLOB + STOP/STATION(YOLO)</h2>
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
    global blob_count, blob_count_armed, blob_ir_passed, blob_missing_frames, ir_blob_count
    global search_right_allowed, third_arrow_trigger_count, second_ir_arrow_bottom

    if key == "p":
        auto_mode = not auto_mode

        if auto_mode:
            # AUTO를 다시 켜도 Blob 카운트는 유지한다.
            # 일시정지/재시작 때문에 BLOB_COUNT가 0으로 돌아가지 않음.
            state = "START"
        else:
            stop_robot()

    elif key == "r":
        # RESET을 눌렀을 때만 Blob 카운트를 0으로 초기화
        auto_mode = False
        state = "START"
        blob_count = 0
        blob_count_armed = True
        blob_ir_passed = False
        blob_missing_frames = 0
        ir_blob_count = 0
        search_right_allowed = False
        third_arrow_trigger_count = 0
        second_ir_arrow_bottom = None
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
