import cv2
import numpy as np
import time
import threading
from collections import deque
from flask import Flask, Response, render_template_string
from pinkylib import Camera, Motor
from ultralytics import YOLO
import os
import zmq
import json

# ============================================================
# Pinky Pro - 화살표 Blob 중심(OpenCV) + STOP/STATION(YOLO) 통합 주행
#
# 동작
# 1) 시작: 도로 이진화만 하면서 직진
# 2) 화살표는 OpenCV, STOP/STATION은 YOLO로 검출 -> 화면 중앙 정렬
# 3) 갈라진 검은 조각을 그룹화한 뒤 solid Blob으로 뭉개고 무게중심 추종
#    STOP/STATION: YOLO bbox 중심 추종
# 4) YOLO bbox 기반 STOP/STATION 정렬 및 정지
# 5) 다음 목표가 안 보이면 오른쪽 제자리 회전하며 탐색
# 6) 목표 발견 -> 중앙 정렬 -> 다시 직진/추종
# ============================================================

app = Flask(__name__)

# ----------------------------
# Hardware
# ----------------------------
motor = Motor()
camera = Camera()

motor.enable_motor()
camera.start()

# ----------------------------
# Parameters
# ----------------------------
PORT = 5000

# ----------------------------
# YOLO text recognition
# ----------------------------
MODEL_PATH = "best_ncnn_model"
YOLO_CONF = 0.45          # 실제 STOP/STATION 주행 목표로 사용할 신뢰도
YOLO_MASK_CONF = 0.20     # 글씨를 Blob에서 제외하기 위한 낮은 마스킹 신뢰도
YOLO_IMGSZ = 320
YOLO_EVERY = 1          # 글씨 bbox를 매 프레임 갱신
TEXT_CLASSES = {"STOP", "STATION"}

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"{MODEL_PATH} 파일이 없습니다. road_test.py와 best.pt를 같은 폴더에 넣어주세요."
    )

# YOLO는 COUNT 2가 된 이후에 한 번만 로드한다.
yolo_model = None


# Motor speed
STRAIGHT_SPEED = 22
FOLLOW_SPEED = 19
SEARCH_SPEED = 10
ALIGN_SPEED = 10

# Manual pulse
MANUAL_SPEED = 24
MANUAL_PULSE = 0.30

# IR removed - camera arrow trigger mode
IR_THRESHOLD = 2600

# IR 카운트가 한 번 증가하면 2초 동안 추가 카운트 금지
IR_COUNT_COOLDOWN_SEC = 2.0

# 카운트 5가 된 순간부터 4초 동안 추가 IR 카운팅 금지
COUNT5_RECOUNT_LOCK_SEC = 4.0

# 카운트 6이 된 순간부터 5초 동안 추가 IR 카운팅 금지
COUNT6_RECOUNT_LOCK_SEC = 5.0

# AUTO 시작 직후 1초 동안은 목표/IR을 무시하고 직진만 한다.
START_STRAIGHT_ONLY_SEC = 1.0

# IR 카운팅은 실제로 다음 노드를 향해 주행하는 상태에서만 허용한다.
# 회전/탐색/정렬/후진 중에는 같은 노드를 다시 밟아도 카운트하지 않는다.
# 특히 COUNT 1 직후 FIRST_BLOB_STRAIGHT에서는 같은 첫 화살표 재카운팅을 막기 위해
# 카운팅을 금지하고, 다음 화살표를 확보해 FOLLOW로 복귀한 뒤 다시 허용한다.
COUNT_ALLOWED_STATES = {
    "START",
    "FOLLOW",
    # COUNT 1 직후 FIRST_BLOB_STRAIGHT에서는 재카운팅 금지.
    # 첫 번째 화살표를 완전히 벗어나고 다음 화살표를 확보해
    # FOLLOW로 복귀한 뒤부터 COUNT 2 카운팅을 다시 허용한다.
    "SECOND_COUNT_DISTANCE_DRIVE",
    "IR_CONTINUE",
}

# Camera ROI: 아래 50%
# 기존에는 좌우 각각 10%를 제외했지만,
# 왼쪽 ROI를 10% 넓혀서 왼쪽은 0%, 오른쪽은 기존처럼 10% 제외한다.
ROI_TOP_RATIO = 0.50
ROI_LEFT_RATIO = 0.00
ROI_RIGHT_RATIO = 0.10

# HSV thresholds
WHITE_LO = np.array([0, 0, 175], dtype=np.uint8)
WHITE_HI = np.array([180, 75, 255], dtype=np.uint8)

BLACK_LO = np.array([0, 0, 0], dtype=np.uint8)
BLACK_HI = np.array([180, 130, 115], dtype=np.uint8)

# Detection
MIN_BLACK_AREA = 350
MIN_ARROW_AREA = 900
PARTIAL_BLOB_MIN_AREA = 300   # 화면에 약 1/3 정도만 보여도 허용
BLOB_GROUP_GAP = 35           # 반사로 갈라진 조각을 같은 화살표로 묶는 최대 간격(px)
BLOB_EDGE_MARGIN = 5          # 화면 경계에 닿으면 잘린 Blob으로 판단
BLOB_GONE_FRAMES = 10         # 같은 화살표 재카운트 방지: 충분히 사라진 뒤 다음 카운트 허용
CENTER_TOL = 30

# Control
KP_ALIGN = 0.11
KP_FOLLOW = 0.10
MAX_CORR = 10


# 카운트 3 전용 동작
COUNT3_REVERSE_SEC = 1.0       # 카운트 3이 되는 순간 1초 후진
COUNT3_STOP_SEC = 3.0          # 후진 후 3초 정지

# 카운트 6 전용 동작
# 6번째 IR 감지 후: 1초 정지 -> 0.2 우회전 -> 1초 정지 -> 2초 후진 -> 3초 정지
COUNT6_PRE_STOP_SEC = 1.0
COUNT6_TURN_SEC = 0.2
COUNT6_POST_TURN_STOP_SEC = 1.0
COUNT6_REVERSE_SEC = 2.0
COUNT6_FINAL_STOP_SEC = 3.0

# ------------------------------------------------------------
# 카운트 2 이후 거리 기반 직진 설정
# ------------------------------------------------------------
# 화살표 카운트는 카메라 solid blob 위치 기반으로 증가한다.
#
# 카운트 2가 된 뒤에는 바로 우회전하지 않고, 앞쪽에 보이는 다음 화살표를
# 거리 기준점으로 사용해 조금 더 직진한다. 실제 거리센서가 아니므로 카메라 영상에서
# 화살표 bbox의 아래쪽 끝(y+h)을 ROI 높이로 나눈 비율을 거리 대용값으로 쓴다.
#
# 값이 작을수록 화살표가 멀리 있을 때 일찍 우회전하고,
# 값이 클수록 화살표에 더 가까이 간 뒤 늦게 우회전한다.
# 예: 0.55 -> 일찍, 0.65 -> 중간, 0.75 -> 늦게
SECOND_COUNT_FORWARD_TRIGGER_RATIO = 0.26

# 카운트 2를 밟은 직후 화면 아래에 남아 있는 "방금 밟은 화살표"와
# 앞쪽의 다음 화살표를 구분하기 위한 점프 기준. ROI 높이의 이 비율 이상
# bbox 하단이 위쪽으로 점프하면 앞쪽의 새 화살표로 간주한다.
SECOND_NEXT_ARROW_SWITCH_JUMP_RATIO = 0.10

# 거리 기준용 다음 화살표를 추종하며 직진할 때 속도/보정값
SECOND_COUNT_FORWARD_SPEED = 17
SECOND_COUNT_FORWARD_KP = 0.08
SECOND_COUNT_FORWARD_MAX_CORR = 7

# 거리 기준선이 흔들려 한 프레임만 넘는 오검출을 막기 위한 연속 프레임 수
SECOND_COUNT_TRIGGER_FRAMES = 2

# Stream
JPEG_QUALITY = 55

# ----------------------------
# State
# ----------------------------
state = "START"
auto_mode = False
auto_start_time = 0.0
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

# 우회전 탐색 중 글씨가 화면에 반쯤 들어온 상태에서
# STOP/STATION을 성급하게 확정하지 않도록 하는 조건
TEXT_FULL_MARGIN = 35          # 화면 좌우 가장자리에서 최소 이만큼 안쪽
TEXT_FULL_STABLE_FRAMES = 3    # 전체 글씨가 연속 3프레임 보여야 확정

# STOP / STATION bbox 흔들림 보정
TEXT_BBOX_ALPHA = 0.60       # 현재 검출값을 더 많이 반영해 주행 중 bbox가 따라오게 함
TEXT_CENTER_DEADBAND = 3     # 아주 작은 흔들림만 무시
TEXT_MAX_JUMP = 180          # 실제 이동은 허용하고 비정상적인 큰 점프만 차단

# ----------------------------
# 화살표 Blob 카운트
# ----------------------------
blob_count = 0

# 카운트는 카메라 화살표 bbox trigger 기반으로 증가한다.
# 같은 검은 표식을 여러 번 세지 않는 역할은 ir_armed가 담당한다.
blob_count_armed = True
blob_ir_passed = False
blob_missing_frames = 0

ir_blob_count = 0


# IR 카운트 직후 다음 IR 카운트 전까지 화살표 정렬 금지
arrow_alignment_locked = False

# SEARCH_RIGHT에서 "먼저 발견한 목표"를 잠근다.
# None / "ARROW" / "STOP" / "STATION"
search_locked_target_type = None

# 우회전 탐색 중 글씨 전체 노출 확인용
search_text_full_count = 0
search_text_candidate_type = None

# 카운트 6 특수 동작 예약 플래그
count6_special_pending = False

# 카운트 4 이후 다음 화살표 전체 노출 확인용
count4_arrow_full_frames = 0
COUNT4_ARROW_FULL_STABLE_FRAMES = 3

# 카운트 7 전용: 다음 화살표 전체 형태 확인
count7_arrow_full_frames = 0
COUNT7_ARROW_FULL_STABLE_FRAMES = 5
COUNT7_ARROW_FULL_MARGIN = 20

# 카운트 2 이후 거리 기반 직진용 상태값
second_forward_trigger_count = 0
second_ir_arrow_bottom = None
second_reference_acquired = False



# ============================================================
# Camera based arrow trigger (NO IR)
# 3번째 화살표:
# solid blob bbox의 아래쪽 끝이 기준선에 도달하면 회전
# ============================================================
ARROW_TRIGGER_Y_RATIO = 0.78
arrow_trigger_count = 0
arrow_trigger_lock = False

def arrow_reach_trigger(arrow_target, roi_height):
    """
    화살표 solid blob의 끝선이 기준선에 닿았는지 판단
    """
    if arrow_target is None:
        return False

    x, y, w, h = arrow_target["bbox"]
    bottom = y + h

    return bottom >= int(roi_height * ARROW_TRIGGER_Y_RATIO)



# ============================================================
# FINAL MODE : Camera Arrow Trigger + STOP/STATION Sequence
# ============================================================
# 화살표는 밟음(IR)이 아니라 solid blob bbox bottom으로 판단
# STOP/STATION은 YOLO bbox 기반 정렬 후 3초 정지
# ============================================================

FINAL_STOP_HOLD_SEC = 3.0
FINAL_STATION_HOLD_SEC = 3.0
FINAL_STATION_TURN_DEG_TIME = 0.9

arrow_camera_count = 0
arrow_bottom_trigger_lock = False

def camera_arrow_trigger(target, frame_height):
    if target is None:
        return False
    x, y, w, h = target["bbox"]
    return (y + h) >= int(frame_height * 0.78)

# ============================================================
# FINAL_SEQUENCE_CONFIG_V3
# STOP -> 3s hold -> arrow disappearance drive -> 80deg turn -> STATION
# Arrow trigger uses blob bbox bottom, not IR.
# ============================================================
FINAL_STOP_HOLD_SEC = 3.0
FINAL_STATION_HOLD_SEC = 3.0
STATION_TURN_TIME = 0.9

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
    x1 = int(w * ROI_LEFT_RATIO)
    x2 = int(w * (1.0 - ROI_RIGHT_RATIO))

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
    global yolo_model

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

def bbox_iou_xywh(a, b):
    """(x,y,w,h) bbox 두 개의 IoU."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    if inter <= 0:
        return 0.0

    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0

    return inter / union


def text_overlaps_raw_arrow(text_target, raw_arrow_target, iou_threshold=0.20):
    """
    YOLO가 화살표를 STOP/STATION으로 잘못 본 경우를 막는다.
    raw black mask에서도 같은 위치가 화살표 Blob으로 잡히고
    bbox가 충분히 겹치면 글씨 후보를 무시한다.
    """
    if text_target is None or raw_arrow_target is None:
        return False

    tb = text_target.get("bbox")
    ab = raw_arrow_target.get("bbox")

    if tb is None or ab is None:
        return False

    return bbox_iou_xywh(tb, ab) >= iou_threshold


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
    global state, auto_mode, auto_start_time, latest_jpeg
    global manual_until, manual_cmd
    global ir_armed, ir_clear_count, last_ir_count_time
    global count5_recount_lock_until, count6_recount_lock_until
    global ir_event_lock_until
    global lost_count, last_target
    global ir_armed, ir_clear_count, last_ir_count_time
    global yolo_model
    global yolo_frame_count, last_yolo_text, yolo_miss_count
    global last_stable_yolo_text, yolo_partial_count, last_text_mask_targets
    global blob_count, blob_count_armed, blob_ir_passed, blob_missing_frames, ir_blob_count
    global search_right_allowed, arrow_ir_expected
    global repeat_ir_cycle, search_locked_target_type, arrow_alignment_locked
    global search_text_full_count, search_text_candidate_type
    global count6_special_pending
    global count6_action_time
    global count4_arrow_full_frames
    global count7_arrow_full_frames
    global second_forward_trigger_count, second_ir_arrow_bottom, second_reference_acquired
    global ir_event_lock_until

    ir_stop_time = 0.0
    count3_action_time = 0.0
    count6_action_time = 0.0
    count6_phase_time = 0.0

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
        detected_text_masks = []

        # YOLO는 COUNT 2가 된 이후부터만 실행한다.
        if blob_count >= 2:
            if yolo_model is None:
                try:
                    print("Loading YOLO model after COUNT 2...")
                    yolo_model = YOLO(MODEL_PATH)
                    print("YOLO model loaded.")
                    print("YOLO classes:", yolo_model.names)
                except Exception as e:
                    print("YOLO LOAD ERROR:", e)
                    yolo_model = None

            if yolo_model is not None:
                yolo_frame_count += 1

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
        else:
            # COUNT 0~1에서는 YOLO 관련 결과를 사용하지 않는다.
            last_yolo_text = None
            last_stable_yolo_text = None
            last_text_mask_targets = []
            yolo_miss_count = 0
            yolo_partial_count = 0

        text_target = last_yolo_text

        # ----------------------------------------------------
        # YOLO가 화살표를 STATION/STOP으로 오인하는 경우를 잡기 위해
        # 글씨 마스킹 전 원본 black mask에서도 화살표 후보를 하나 계산한다.
        # 이 raw 후보는 "오인 판정"에만 쓰고 실제 주행용 화살표는
        # 아래의 글씨 제거 후 arrow_target을 그대로 사용한다.
        # ----------------------------------------------------
        raw_arrow_target = detect_arrow_target(black_mask)

        # YOLO 글씨 영역을 이진화 마스크에서 제거
        arrow_black_mask = remove_yolo_text_from_black_mask(
            black_mask,
            last_text_mask_targets,
            ox,
            oy
        )

        # 이진화 결과에서 화살표 Blob 검출
        arrow_target = detect_arrow_target(arrow_black_mask)

        # 카운트 2 이후 우회전 탐색 중에는
        # 같은 위치가 raw arrow blob으로도 잡히면
        # YOLO의 STOP/STATION 판정을 화살표 오인으로 보고 무시한다.
        if (
            state in ("SEARCH_STOP_RIGHT", "ALIGN_STOP")
            and text_target is not None
            and text_overlaps_raw_arrow(text_target, raw_arrow_target)
        ):
            print(
                f"[TEXT FILTER] ignore false {text_target.get('type')} "
                f"overlapping raw arrow"
            )
            text_target = None

        # ----------------------------------------------------
        # 화살표 카운팅
        #
        # 중요: 여기서는 화살표를 "검출"해도 카운트하지 않는다.
        # blob_count는 아래 FOLLOW 상태에서 IR 센서가 실제 화살표를 밟았을 때만 증가한다.
        # ----------------------------------------------------

        # ----------------------------------------------------
        # 목표 선택
        # ----------------------------------------------------
        stop_target = None
        if text_target is not None and text_target.get("type") in ("STOP", "STATION"):
            stop_target = text_target

        if state in ("FIRST_BLOB_STRAIGHT", "SECOND_COUNT_DISTANCE_DRIVE"):
            # 카운트 1 이후 다음 화살표 탐색 / 카운트 2 이후 거리 기준 직진 중에는
            # STOP/STATION을 무시하고 화살표만 본다.
            target = arrow_target

        elif state in ("SEARCH_STOP_RIGHT", "ALIGN_STOP"):
            # 카운트 2 이후 전용 회전/정렬 단계에서는
            # STOP / STATION 둘 다 bbox 목표로 인정한다.
            target = stop_target

        else:
            # SEARCH_RIGHT에서 먼저 잡은 목표가 있으면 그 종류를 유지한다.
            # 즉, 글씨를 먼저 잡았으면 글씨를, 화살표를 먼저 잡았으면 화살표를
            # IR을 밟을 때까지 다른 종류로 바꾸지 않는다.
            if search_locked_target_type == "ARROW":
                target = arrow_target

            elif search_locked_target_type in TEXT_CLASSES:
                if (
                    text_target is not None
                    and text_target.get("type") == search_locked_target_type
                ):
                    target = text_target
                else:
                    target = None

            else:
                # 평상시는 화면에서 더 가까운 목표 사용
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

        ir_l, ir_c, ir_r = 0, 0, 0

        # ----------------------------------------------------
        # IR hit 판단
        # 기존: L/C/R 중 하나만 감지되어도 카운트
        # 변경: 3개 센서 중 2개 이상 감지될 때만 인정
        # 회전 중 한쪽 센서가 같은 화살표를 다시 보는 문제 방지
        # ----------------------------------------------------
        ir_detect_count = sum([
            ir_l >= IR_THRESHOLD,
            ir_c >= IR_THRESHOLD,
            ir_r >= IR_THRESHOLD
        ])

        ir_hit = ir_detect_count >= 2

        # 카운트 직후 일정 시간 동안 같은 표식 재감지 방지
        if time.time() < ir_event_lock_until:
            ir_hit = False

        # STOP/STATION 정렬 구간에서는 카메라(YOLO)를 우선
        # IR 튐으로 잘못된 COUNT가 발생하지 않도록 차단
        if state in ("SEARCH_STOP_RIGHT", "ALIGN_STOP"):
            ir_hit = False

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

        # ----------------------------------------------------
        # GLOBAL IR COUNT
        #
        # AUTO 시작 후 첫 1초를 제외하고,
        # 실제 다음 노드로 주행하는 COUNT_ALLOWED_STATES에서만
        # IR 카운팅을 허용한다.
        #
        # SEARCH / ALIGN / TURN / REVERSE 상태에서는
        # 같은 노드를 다시 밟아도 카운트하지 않는다.
        #
        # 같은 표식 중복 카운트는 ir_armed=False로 잠그고,
        # 흰 바닥을 IR_CLEAR_FRAMES 연속 확인한 뒤에만 재활성화한다.
        # ----------------------------------------------------
        global_ir_count_event = False
        now = time.time()

        if (
            auto_mode
            and (now - auto_start_time) >= START_STRAIGHT_ONLY_SEC
            and state in COUNT_ALLOWED_STATES
            and ir_armed
            and ir_hit
            and now >= count5_recount_lock_until
            and now >= count6_recount_lock_until
            and (now - last_ir_count_time) >= IR_COUNT_COOLDOWN_SEC
        ):
            blob_count += 1
            ir_blob_count = blob_count
            global_ir_count_event = True

            # 카운트가 올라간 순간만 시간 저장.
            # 이후 2초 동안 센서가 계속 감지돼도 카운트는 증가하지 않는다.
            last_ir_count_time = now

            # 같은 화살표를 회전 중 다시 밟는 상황 방지
            ir_event_lock_until = now + IR_EVENT_LOCK_SEC

            if blob_count == 5:
                count5_recount_lock_until = now + COUNT5_RECOUNT_LOCK_SEC
                print("[COUNT 5] IR recount locked for 4.0s")

            if blob_count == 6:
                count6_recount_lock_until = now + COUNT6_RECOUNT_LOCK_SEC
                print("[COUNT 6] IR recount locked for 5.0s")

            print(
                f"[GLOBAL IR COUNT] {blob_count} "
                f"state={state} target={search_locked_target_type}"
            )

            # 같은 검은 표식 중복 카운트 방지
            ir_armed = False
            ir_clear_count = 0
            arrow_ir_expected = False

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
            # GLOBAL IR EVENT ROUTING
            #
            # 카운트 자체는 위에서 모든 state 공통으로 이미 처리됨.
            # 여기서는 새 카운트가 발생했을 때 필요한 특수 동작만 분기한다.
            # ====================================================
            if global_ir_count_event:
                # 카운트 1
                if ir_blob_count == 1:
                    search_right_allowed = False
                    state = "FIRST_BLOB_STRAIGHT"
                    print("[COUNT 1] global IR -> FIRST_BLOB_STRAIGHT")

                # 카운트 2
                elif ir_blob_count == 2:
                    second_forward_trigger_count = 0
                    second_reference_acquired = False

                    if arrow_target is not None:
                        ax, ay, aw, ah = arrow_target["bbox"]
                        second_ir_arrow_bottom = ay + ah
                    else:
                        second_ir_arrow_bottom = None

                    search_right_allowed = False
                    state = "SECOND_COUNT_DISTANCE_DRIVE"
                    print("[COUNT 2] global IR -> distance-based forward drive")

                # 카운트 3
                elif ir_blob_count == 3:
                    search_locked_target_type = None
                    search_right_allowed = False
                    arrow_ir_expected = False
                    count3_action_time = time.time()
                    state = "COUNT3_REVERSE"
                    print("[COUNT 3] global IR -> immediate reverse 1.0s")

                # 카운트 4
                elif ir_blob_count == 4:
                    count4_arrow_full_frames = 0
                    search_locked_target_type = None
                    search_right_allowed = True
                    state = "COUNT4_SEARCH_ARROW_RIGHT"
                    print("[COUNT 4] global IR -> search full arrow to the right")

                # 카운트 6 특수동작 예약
                elif ir_blob_count == 6:
                    count6_special_pending = True
                    arrow_alignment_locked = True
                    search_locked_target_type = None
                    search_right_allowed = False

                    ir_stop_time = time.time()
                    state = "IR_CONTINUE"
                    print("[COUNT 6] global IR -> station maneuver reserved")

                # 카운트 7:
                # COUNT 6 이후 다시 직진하다가 IR을 밟아 7이 되는 순간
                # 바로 오른쪽으로 회전하며 다음 화살표를 탐색한다.
                elif ir_blob_count == 7:
                    stop_robot()
                    count7_arrow_full_frames = 0
                    search_text_full_count = 0
                    search_text_candidate_type = None
                    search_locked_target_type = None
                    search_right_allowed = True
                    arrow_ir_expected = False
                    state = "COUNT7_SEARCH_ARROW_RIGHT"
                    print("[COUNT 7] global IR -> search full arrow to the right")

                # 그 외 카운트는 기존 반복 로직처럼
                # 1초 직진 후 다음 목표 탐색으로 이어간다.
                else:
                    search_locked_target_type = None
                    search_right_allowed = False
                    ir_stop_time = time.time()
                    state = "IR_CONTINUE"
                    print(f"[COUNT {ir_blob_count}] global IR -> 1.0s straight")

            # ====================================================
            # AUTO 시작 직후 1초
            # IR/화살표/글씨 판단을 전부 무시하고 직진만 한다.
            # 따라서 이 1초 동안은 IR 카운트가 절대 증가하지 않는다.
            # ====================================================
            elif time.time() - auto_start_time < START_STRAIGHT_ONLY_SEC:
                state = "START"
                arrow_ir_expected = False
                search_locked_target_type = None
                drive(STRAIGHT_SPEED, STRAIGHT_SPEED)

            # ====================================================
            # START
            # 검은 목표가 보일 때까지 직진
            # ====================================================
            elif state == "START":
                if target is None:
                    drive(STRAIGHT_SPEED, STRAIGHT_SPEED)
                else:
                    last_target = target

                    if target.get("type") == "ARROW":
                        # 화살표는 제자리 정렬하지 않고 바로 추종
                        arrow_ir_expected = True
                        state = "FOLLOW"
                    else:
                        # STOP / STATION만 bbox 중심으로 미세정렬
                        arrow_ir_expected = False
                        state = "ALIGN"

            # ====================================================
            # ALIGN
            # STOP / STATION 글씨만 bbox 중심을 화면 가운데로 정렬
            # 화살표는 정렬하지 않고 바로 FOLLOW
            # ====================================================
            elif state == "ALIGN":
                # ALIGN은 글씨(STOP/STATION) 전용.
                # 화살표는 제자리 정렬하지 않고 바로 FOLLOW로 넘긴다.
                if target is not None and target.get("type") == "ARROW":
                    # STOP/STATION 정렬 구간에서는 화살표 절대 정렬 기준으로 사용하지 않음
                    # 화살표는 다음 IR 카운팅용으로만 유지
                    arrow_ir_expected = True
                    state = "FOLLOW"

                elif target is None:
                    lost_count += 1

                    if lost_count >= LOST_TARGET_FRAMES:
                        # 글씨를 잠깐 놓치면 FOLLOW로 복귀
                        state = "FOLLOW"
                        lost_count = 0
                else:
                    lost_count = 0
                    last_target = target
                    arrow_ir_expected = False

                    target_x = target["center"][0] + ox
                    error = target_x - (w/2)

                    if abs(error) <= CENTER_TOL:
                        stop_robot()
                        state = "FOLLOW"
                    else:
                        # STOP / STATION만 제자리 미세정렬
                        turn = int(np.clip(KP_ALIGN * error, -ALIGN_SPEED, ALIGN_SPEED))

                        if turn > 0:
                            drive(ALIGN_SPEED, -ALIGN_SPEED)
                        else:
                            drive(-ALIGN_SPEED, ALIGN_SPEED)

            # ====================================================
            # FOLLOW
            # 화살표는 Blob 중심 / STOP·STATION은 YOLO bbox 중심 추종
            # 화살표 카운트는 IR을 실제로 밟은 순간에만 증가한다.
            # ====================================================
            elif state == "FOLLOW":
                # ------------------------------------------------
                # 반복 후속 구간:
                # SEARCH_RIGHT에서 먼저 발견한 글씨/화살표를 정렬해서 따라간 뒤
                # IR을 밟으면 목표 종류와 상관없이 1초 직진 -> 정지 -> 다시 우회전 탐색
                # ------------------------------------------------
                # IR 카운팅은 이제 state와 무관하게 위의 GLOBAL IR COUNT에서 처리한다.

                # 화살표가 현재 프레임에서 보이면 IR 대기 latch를 켠다.
                # 이후 화살표가 카메라 아래로 빠져 target=None이 되어도
                # IR이 검은 표식을 밟는 순간 해당 화살표를 통과한 것으로 인정한다.
                if target is not None and target.get("type") == "ARROW" and not arrow_alignment_locked:
                    arrow_ir_expected = True

                if target is not None and target.get("type") in ("STOP", "STATION"):
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
            # 카운트 1 이후에는 회전하지 않고 다음 화살표까지 직진한다.
            # 카운트는 여기서 증가하지 않으며, 다음 화살표를 IR로 밟을 때 2가 된다.
            # ====================================================
            elif state == "FIRST_BLOB_STRAIGHT":
                # 다음 카운팅 화살표가 카메라 중앙선에 들어올 때까지 방향만 보정
                # 화살표는 정렬용이 아니라 다음 IR 위치 확보용으로만 사용한다.
                # 중심이 맞은 뒤 직진하여 IR을 밟도록 한다.
                if arrow_target is not None:
                    arrow_x = arrow_target["center"][0] + ox
                    error = arrow_x - (w / 2)

                    if abs(error) > CENTER_TOL:
                        if error > 0:
                            drive(ALIGN_SPEED, -ALIGN_SPEED)
                        else:
                            drive(-ALIGN_SPEED, ALIGN_SPEED)
                    else:
                        drive(STRAIGHT_SPEED, STRAIGHT_SPEED)
                else:
                    drive(STRAIGHT_SPEED, STRAIGHT_SPEED)

                # IR이 다시 흰 바닥에서 재활성화된 뒤 보이는 화살표를
                # 다음 화살표로 보고 ALIGN/FOLLOW로 복귀한다.
                if ir_armed and arrow_target is not None:
                    ax, ay, aw, ah = arrow_target["bbox"]
                    arrow_bottom = ay + ah

                    # 방금 밟은 화살표가 화면 맨 아래에 남아 있는 동안은 무시.
                    # ROI 아래 88%보다 위쪽에 있는 화살표부터 '앞쪽 화살표'로 인정한다.
                    if arrow_bottom <= int(roi.shape[0] * 0.88):
                        stop_robot()
                        last_target = arrow_target
                        lost_count = 0
                        arrow_ir_expected = True
                        # 화살표는 제자리 정렬하지 않고 바로 추종
                        state = "FOLLOW"
                        print("[COUNT 1] next arrow acquired -> follow")

            # ====================================================
            # SECOND COUNT DISTANCE DRIVE
            # 카운트 2 이후에는 고정 1초가 아니라 앞 화살표와의 화면상 거리로
            # 직진 종료 시점을 정한다. 이 화살표는 '거리 기준점'일 뿐 카운트하지 않는다.
            # ====================================================
            elif state == "SECOND_COUNT_DISTANCE_DRIVE":
                # 우선 기본은 직진. 앞 화살표가 안정적으로 잡히면 약하게 중심 추종한다.
                if arrow_target is not None:
                    ax, ay, aw, ah = arrow_target["bbox"]
                    current_bottom = ay + ah

                    # 방금 밟은 2번 화살표가 화면에서 빠지고
                    # 더 먼 앞 화살표로 target이 바뀌었는지 먼저 확인한다.
                    if not second_reference_acquired:
                        switch_jump_px = int(roi.shape[0] * SECOND_NEXT_ARROW_SWITCH_JUMP_RATIO)

                        if (
                            ir_armed
                            and (
                                second_ir_arrow_bottom is None
                                or current_bottom <= second_ir_arrow_bottom - switch_jump_px
                            )
                        ):
                            second_reference_acquired = True
                            print(f"[COUNT 2] reference arrow acquired, bottom={current_bottom}")

                    if second_reference_acquired:
                        trigger_y = int(roi.shape[0] * SECOND_COUNT_FORWARD_TRIGGER_RATIO)

                        # 앞 화살표가 화면 아래쪽으로 내려올수록 로봇과 가까워진다.
                        if current_bottom >= trigger_y:
                            second_forward_trigger_count += 1
                        else:
                            second_forward_trigger_count = 0

                        if second_forward_trigger_count >= SECOND_COUNT_TRIGGER_FRAMES:
                            stop_robot()
                            arrow_ir_expected = False
                            search_right_allowed = True
                            second_forward_trigger_count = 0
                            state = "SEARCH_STOP_RIGHT"
                            print(
                                f"[COUNT 2] reference bottom={current_bottom}, "
                                f"trigger={trigger_y} -> SEARCH STOP RIGHT"
                            )
                        else:
                            target_x = arrow_target["center"][0] + ox
                            error = target_x - (w / 2)
                            p_drive(
                                error,
                                SECOND_COUNT_FORWARD_SPEED,
                                SECOND_COUNT_FORWARD_KP,
                                SECOND_COUNT_FORWARD_MAX_CORR
                            )
                    else:
                        drive(SECOND_COUNT_FORWARD_SPEED, SECOND_COUNT_FORWARD_SPEED)

                else:
                    # 다음 화살표가 아직 안 보이면 그대로 천천히 직진하며 탐색
                    second_forward_trigger_count = 0
                    drive(SECOND_COUNT_FORWARD_SPEED, SECOND_COUNT_FORWARD_SPEED)

            # ====================================================
            # SEARCH TEXT RIGHT
            # 3번째 화살표 기준선에서 오른쪽 제자리 회전.
            # STOP / STATION 둘 중 먼저 인식되는 글씨의 bbox를 목표로 사용한다.
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
                    print(
                        f"[{stop_target.get('type')}] "
                        f"detected during right turn -> align"
                    )
                else:
                    drive(SEARCH_SPEED, -SEARCH_SPEED)

            # ====================================================
            # ALIGN TEXT
            # STOP / STATION bbox 중심이 화면 중앙에 올 때까지 제자리 회전으로 미세 정렬.
            # 정렬 완료 뒤 해당 글씨를 향해 직진 추종한다.
            # ====================================================
            elif state == "ALIGN_STOP":
                # STOP/STATION 정렬 전용 상태. ARROW는 절대 정렬 대상으로 사용하지 않음.
                # YOLO stop_target만 사용한다.
                if stop_target is None:
                    # STOP/STATION을 순간적으로 놓치면 다시 오른쪽 탐색으로 복귀
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
                        arrow_ir_expected = False
                        state = "FOLLOW"
                        print(
                            f"[{stop_target.get('type')}] "
                            f"centered -> follow straight"
                        )
                    else:
                        if error > 0:
                            drive(ALIGN_SPEED, -ALIGN_SPEED)
                        else:
                            drive(-ALIGN_SPEED, ALIGN_SPEED)

            # ====================================================
            # COUNT 3 REVERSE
            # 카운트 3이 된 직후 1초 후진
            # ====================================================
            elif state == "COUNT3_REVERSE":
                if time.time() - count3_action_time < COUNT3_REVERSE_SEC:
                    drive(-FOLLOW_SPEED, -FOLLOW_SPEED)
                else:
                    stop_robot()
                    count3_action_time = time.time()
                    state = "COUNT3_STOP"

            # ====================================================
            # COUNT 3 STOP
            # 1초 후진 완료 후 3초 정지
            # ====================================================
            elif state == "COUNT3_STOP":
                stop_robot()

                if time.time() - count3_action_time >= COUNT3_STOP_SEC:
                    # 카운트 3은 기존 흐름 그대로 복귀:
                    # 3초 정지 후 반복 주행 로직으로 진입한다.
                    repeat_ir_cycle = True
                    search_locked_target_type = None

                    # 같은 IR 표식 재감지 방지
                    ir_armed = False
                    ir_clear_count = 0
                    arrow_ir_expected = False

                    ir_stop_time = time.time()
                    state = "IR_CONTINUE"
                    print("[COUNT 3] enter repeat target cycle")

            # ====================================================
            # COUNT 4 -> NEXT ARROW SEARCH
            # 3초 정지 후 오른쪽으로 돌면서
            # 화살표 전체 형태가 보일 때까지 기다린다.
            # ====================================================
            elif state == "COUNT4_SEARCH_ARROW_RIGHT":
                full_arrow_visible = (
                    arrow_target is not None
                    and not arrow_target.get("partial", False)
                )

                if full_arrow_visible:
                    count4_arrow_full_frames += 1
                else:
                    count4_arrow_full_frames = 0

                if count4_arrow_full_frames >= COUNT4_ARROW_FULL_STABLE_FRAMES:
                    stop_robot()
                    last_target = arrow_target
                    lost_count = 0
                    count4_arrow_full_frames = 0
                    state = "COUNT4_ALIGN_ARROW"
                    print("[COUNT 4] full arrow visible -> align center")
                else:
                    drive(SEARCH_SPEED, -SEARCH_SPEED)

            # ====================================================
            # COUNT 4 -> NEXT ARROW ALIGN
            # 화살표 전체가 확인된 뒤에만 중앙 정렬한다.
            # 중앙 정렬 완료 후 직진한다.
            # ====================================================
            elif state == "COUNT4_ALIGN_ARROW":
                if arrow_target is None:
                    count4_arrow_full_frames = 0
                    state = "COUNT4_SEARCH_ARROW_RIGHT"

                elif arrow_target.get("partial", False):
                    count4_arrow_full_frames = 0
                    state = "COUNT4_SEARCH_ARROW_RIGHT"

                else:
                    target_x = arrow_target["center"][0] + ox
                    error = target_x - (w / 2)

                    if abs(error) <= CENTER_TOL:
                        stop_robot()
                        last_target = arrow_target

                        # 카운트4 이후 다음 화살표를 정확히 추종 대상으로 고정.
                        # 화살표 중앙 정렬 완료 후 FOLLOW로 들어가고,
                        # IR이 아직 잠겨 있다면 흰 바닥을 벗어난 뒤 자동 재활성화된다.
                        arrow_ir_expected = True
                        search_locked_target_type = "ARROW"
                        search_right_allowed = False
                        state = "FOLLOW"
                        print(
                            "[COUNT 4] arrow centered -> FOLLOW / "
                            f"IR_ARMED={ir_armed}"
                        )
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

                    if count6_special_pending and blob_count >= 6:
                        # COUNT 6 전용: 우회전 1초 후 후진 1초
                        count6_phase_time = time.time()
                        state = "COUNT6_PRE_STOP"
                        print("[COUNT 6] pre stop 1s -> turn")
                    else:
                        ir_stop_time = time.time()
                        state = "IR_STOP"

            # ====================================================
            # COUNT 6 SPECIAL
            # 6번째 IR에서만 실행:
            # 정지 1초 -> 우회전 0.6초 -> 정지 1초 -> 후진 2초 -> 정지 3초
            # ====================================================
            elif state == "COUNT6_PRE_STOP":
                stop_robot()
                if time.time() - count6_phase_time >= COUNT6_PRE_STOP_SEC:
                    count6_phase_time = time.time()
                    state = "COUNT6_TURN"
                    print("[COUNT 6] pre stop done -> turn 0.6s")

            elif state == "COUNT6_TURN":
                if time.time() - count6_phase_time < COUNT6_TURN_SEC:
                    drive(FOLLOW_SPEED, -FOLLOW_SPEED)
                else:
                    stop_robot()
                    count6_phase_time = time.time()
                    state = "COUNT6_POST_TURN_STOP"
                    print("[COUNT 6] turn done -> stop 1s")

            elif state == "COUNT6_POST_TURN_STOP":
                stop_robot()
                if time.time() - count6_phase_time >= COUNT6_POST_TURN_STOP_SEC:
                    count6_phase_time = time.time()
                    state = "COUNT6_REVERSE"
                    print("[COUNT 6] stop done -> reverse 2s")

            elif state == "COUNT6_REVERSE":
                if time.time() - count6_phase_time < COUNT6_REVERSE_SEC:
                    drive(-FOLLOW_SPEED, -FOLLOW_SPEED)
                else:
                    stop_robot()
                    count6_phase_time = time.time()
                    state = "COUNT6_FINAL_STOP"
                    print("[COUNT 6] reverse done -> stop 3s")

            elif state == "COUNT6_FINAL_STOP":
                stop_robot()
                if time.time() - count6_phase_time >= COUNT6_FINAL_STOP_SEC:
                    count6_special_pending = False
                    arrow_alignment_locked = True
                    ir_armed = False
                    ir_clear_count = 0

                    # COUNT 6이 끝났다고 바로 다음 노드를 탐색하지 않는다.
                    # 다시 직진해서 다음 IR 표식을 밟아 COUNT 7이 될 때까지 간다.
                    search_right_allowed = False
                    search_locked_target_type = None
                    state = "FOLLOW"
                    print("[COUNT 6] final stop done -> drive straight until COUNT 7")

            # ====================================================
            # COUNT 7 -> NEXT ARROW SEARCH
            # COUNT 7이 되는 순간부터 오른쪽으로 회전하면서
            # 다음 화살표 전체 형태가 보일 때까지 기다린다.
            # STOP / STATION은 COUNT 7 탐색 목표로 사용하지 않는다.
            # ====================================================
            elif state == "COUNT7_SEARCH_ARROW_RIGHT":
                full_arrow_visible = False

                if arrow_target is not None:
                    ax, ay, aw, ah = arrow_target["bbox"]

                    full_arrow_visible = (
                        not arrow_target.get("partial", False)
                        and ax >= COUNT7_ARROW_FULL_MARGIN
                        and ay >= COUNT7_ARROW_FULL_MARGIN
                        and (ax + aw) <= (roi.shape[1] - COUNT7_ARROW_FULL_MARGIN)
                        and (ay + ah) <= (roi.shape[0] - COUNT7_ARROW_FULL_MARGIN)
                    )

                if full_arrow_visible:
                    count7_arrow_full_frames += 1
                else:
                    count7_arrow_full_frames = 0

                if count7_arrow_full_frames >= COUNT7_ARROW_FULL_STABLE_FRAMES:
                    stop_robot()
                    last_target = arrow_target
                    lost_count = 0
                    count7_arrow_full_frames = 0
                    search_locked_target_type = "ARROW"
                    search_right_allowed = False
                    state = "COUNT7_ALIGN_ARROW"
                    print("[COUNT 7] full arrow confirmed -> align center")
                else:
                    drive(SEARCH_SPEED, -SEARCH_SPEED)

            # ====================================================
            # COUNT 7 -> NEXT ARROW ALIGN
            # 화살표 전체 형태가 유지되는 동안 중앙 정렬하고,
            # 중앙 정렬 완료 후에만 직진한다.
            # ====================================================
            elif state == "COUNT7_ALIGN_ARROW":
                if arrow_target is None:
                    count7_arrow_full_frames = 0
                    search_locked_target_type = None
                    search_right_allowed = True
                    state = "COUNT7_SEARCH_ARROW_RIGHT"

                else:
                    ax, ay, aw, ah = arrow_target["bbox"]

                    full_arrow_visible = (
                        not arrow_target.get("partial", False)
                        and ax >= COUNT7_ARROW_FULL_MARGIN
                        and ay >= COUNT7_ARROW_FULL_MARGIN
                        and (ax + aw) <= (roi.shape[1] - COUNT7_ARROW_FULL_MARGIN)
                        and (ay + ah) <= (roi.shape[0] - COUNT7_ARROW_FULL_MARGIN)
                    )

                    # 정렬 중 화살표가 다시 잘려 보이면 탐색부터 다시 한다.
                    if not full_arrow_visible:
                        stop_robot()
                        count7_arrow_full_frames = 0
                        search_locked_target_type = None
                        search_right_allowed = True
                        state = "COUNT7_SEARCH_ARROW_RIGHT"
                        print("[COUNT 7] arrow partial again -> search again")

                    else:
                        target_x = arrow_target["center"][0] + ox
                        error = target_x - (w / 2)

                        if abs(error) <= CENTER_TOL:
                            stop_robot()
                            last_target = arrow_target
                            arrow_ir_expected = True
                            search_locked_target_type = "ARROW"
                            search_right_allowed = False
                            state = "FOLLOW"
                            print("[COUNT 7] full arrow centered -> drive straight")
                        else:
                            if error > 0:
                                drive(ALIGN_SPEED, -ALIGN_SPEED)
                            else:
                                drive(-ALIGN_SPEED, ALIGN_SPEED)

            elif state == "IR_STOP":
                stop_robot()

                if time.time() - ir_stop_time >= IR_STOP_SEC:
                    search_right_allowed = True
                    search_locked_target_type = None
                    search_text_full_count = 0
                    search_text_candidate_type = None
                    state = "SEARCH_RIGHT"

            elif state == "SEARCH_RIGHT":
                if not search_right_allowed:
                    drive(STRAIGHT_SPEED, STRAIGHT_SPEED)
                    state = "FOLLOW"

                else:
                    # ------------------------------------------------
                    # 화살표:
                    # 기존처럼 보이면 바로 목표 후보로 사용 가능.
                    #
                    # STOP / STATION:
                    # 회전 중 글씨가 반쯤만 보일 때는 절대 바로 정렬하지 않는다.
                    # bbox가 화면 좌우 가장자리에서 충분히 안쪽으로 들어오고,
                    # clipped=False 상태가 연속 TEXT_FULL_STABLE_FRAMES 동안
                    # 유지된 뒤에만 "글씨 전체가 보였다"고 판단한다.
                    # ------------------------------------------------
                    full_text_target = None

                    if text_target is not None:
                        fx1, fy1, fx2, fy2 = text_target.get(
                            "frame_bbox",
                            (0, 0, 0, 0)
                        )

                        text_fully_inside = (
                            not text_target.get("clipped", False)
                            and fx1 >= TEXT_FULL_MARGIN
                            and fx2 <= (w - TEXT_FULL_MARGIN)
                        )

                        current_text_type = text_target.get("type")

                        if text_fully_inside:
                            if search_text_candidate_type == current_text_type:
                                search_text_full_count += 1
                            else:
                                search_text_candidate_type = current_text_type
                                search_text_full_count = 1

                            if search_text_full_count >= TEXT_FULL_STABLE_FRAMES:
                                full_text_target = text_target
                        else:
                            search_text_full_count = 0
                            search_text_candidate_type = None
                    else:
                        search_text_full_count = 0
                        search_text_candidate_type = None

                    # 글씨는 "전체 노출 확인 완료"된 경우에만 후보가 된다.
                    # 화살표는 기존처럼 바로 후보 가능.
                    first_target = None

                    if full_text_target is not None and arrow_target is None:
                        first_target = full_text_target

                    elif arrow_target is not None and full_text_target is None:
                        first_target = arrow_target

                    elif full_text_target is not None and arrow_target is not None:
                        # 둘 다 유효한 상태라면 더 가까운 목표를 선택
                        first_target = max(
                            [full_text_target, arrow_target],
                            key=lambda z: z["bbox"][1] + z["bbox"][3]
                        )

                    if first_target is not None:
                        stop_robot()
                        last_target = first_target
                        lost_count = 0
                        search_right_allowed = False

                        search_locked_target_type = first_target.get("type")

                        # 다음 탐색을 위해 글씨 확인 카운터 초기화
                        search_text_full_count = 0
                        search_text_candidate_type = None

                        if search_locked_target_type == "ARROW":
                            arrow_ir_expected = True
                            # 화살표는 미세정렬 없이 바로 추종
                            state = "FOLLOW"
                        else:
                            arrow_ir_expected = False
                            # STOP / STATION만 bbox 중심 미세정렬
                            state = "ALIGN"

                        print(
                            f"[SEARCH] full target locked: "
                            f"{search_locked_target_type}"
                        )
                    else:
                        # 아직 글씨 전체가 안 들어왔으면 계속 오른쪽으로 돌면서 더 본다.
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

        # 카운트 2 이후 거리 기반 직진 기준선
        # 앞 화살표 bbox 하단이 이 선까지 내려오면 우회전을 시작한다.
        if state == "SECOND_COUNT_DISTANCE_DRIVE":
            second_trigger_y = oy + int(roi.shape[0] * SECOND_COUNT_FORWARD_TRIGGER_RATIO)
            cv2.line(
                vis,
                (ox, second_trigger_y),
                (w-ox, second_trigger_y),
                (0, 0, 255),
                2
            )
            cv2.putText(
                vis,
                f"COUNT2 DIST LINE {SECOND_COUNT_FORWARD_TRIGGER_RATIO:.2f}",
                (ox + 8, max(20, second_trigger_y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                .48,
                (0, 0, 255),
                2
            )

            if arrow_target is not None:
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

        cv2.putText(
            vis,
            f"IR_COUNT_ALLOWED: {state in COUNT_ALLOWED_STATES}",
            (12,160),
            cv2.FONT_HERSHEY_SIMPLEX,
            .46,
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
<h2>Pinky IR-COUNT BLOB + STOP/STATION(YOLO)</h2>
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
    global auto_mode, state, auto_start_time
    global manual_until, manual_cmd
    global ir_armed, ir_clear_count, last_ir_count_time
    global count5_recount_lock_until, count6_recount_lock_until
    global blob_count, blob_count_armed, blob_ir_passed, blob_missing_frames, ir_blob_count
    global search_right_allowed, arrow_ir_expected
    global repeat_ir_cycle, search_locked_target_type, arrow_alignment_locked
    global search_text_full_count, search_text_candidate_type
    global count6_special_pending
    global count6_action_time
    global count4_arrow_full_frames
    global count7_arrow_full_frames
    global second_forward_trigger_count, second_ir_arrow_bottom, second_reference_acquired
    global ir_event_lock_until

    if key == "p":
        auto_mode = not auto_mode

        if auto_mode:
            # AUTO를 켠 순간부터 첫 1초는 IR/목표를 무시하고 직진만 한다.
            auto_start_time = time.time()

            # AUTO를 다시 켜도 Blob 카운트는 유지한다.
            # 일시정지/재시작 때문에 BLOB_COUNT가 0으로 돌아가지 않음.
            state = "START"
        else:
            stop_robot()

    elif key == "r":
        # RESET을 눌렀을 때만 Blob 카운트를 0으로 초기화
        auto_mode = False
        state = "START"
        auto_start_time = 0.0
        blob_count = 0
        blob_count_armed = True
        blob_ir_passed = False
        blob_missing_frames = 0
        ir_blob_count = 0
        search_right_allowed = False
        arrow_ir_expected = False
        repeat_ir_cycle = False
        search_locked_target_type = None
        search_text_full_count = 0
        search_text_candidate_type = None
        count6_special_pending = False
        count4_arrow_full_frames = 0
        count7_arrow_full_frames = 0
        second_forward_trigger_count = 0
        second_ir_arrow_bottom = None
        second_reference_acquired = False
        arrow_alignment_locked = False
        ir_armed = True
        ir_clear_count = 0
        last_ir_count_time = -999.0

        # IR 카운트 직후 일시 잠금 시간
        ir_event_lock_until = 0.0
        count5_recount_lock_until = 0.0
        count6_recount_lock_until = 0.0
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
# ZMQ 통신 클라이언트 대기 스레드
# ============================================================
def zmq_wait_for_start():
    global auto_mode, state, auto_start_time

    context = zmq.Context()
    socket = context.socket(zmq.REQ)

    # 노트북 서버 IP로 변경 필요
    socket.connect("tcp://192.168.4.20:6000")

    print("[핑키봇] 노트북 서버 접속 및 시작 명령 대기")

    socket.send_string("핑키봇 준비 완료!")

    response_raw = socket.recv_string()
    response = json.loads(response_raw)

    if response.get("status") == "START_AUTONAV":
        print("[핑키봇] START_AUTONAV 수신 -> 자동주행 시작")

        auto_mode = True
        auto_start_time = time.time()
        state = "START"


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    threading.Thread(
        target=zmq_wait_for_start,
        daemon=True
    ).start()

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
