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

LAPTOP_ZMQ_IP = "172.20.10.14"  # 노트북 아이피
LAPTOP_ZMQ_PORT = 6000


# ============================================================
# 주행 설정
# ============================================================
BASE_SPEED = 20
KP = 0.12
MAX_SPEED = 32

CURVE_SLOWDOWN = 0.05   # 중심선 오차가 커질수록 기본 속도 감소
MIN_CURVE_SPEED = 12    # 커브 최소 속도
INNER_MIN_SPEED = -10   # 안쪽 바퀴 역회전 허용 속도

SEARCH_SPEED = 16
SEARCH_INNER_SPEED = -6  # 중심선 이탈 시 안쪽 바퀴 역회전 탐색

TEXT_FOLLOW_SPEED = 16
TEXT_FOLLOW_KP = 0.10
TEXT_FOLLOW_MAX_CORR = 10

TURN_SPEED = 18

MANUAL_SPEED = 24
MANUAL_PULSE = 0.30
JPEG_QUALITY = 55


# ============================================================
# ROI / HSV / 중심선 검출
# ============================================================
ROI_START_RATIO = 0.50

LOWER_WHITE = np.array([0, 0, 175], dtype=np.uint8)
UPPER_WHITE = np.array([180, 75, 255], dtype=np.uint8)

MIN_AREA = 500
STEP = 15
MIN_WIDTH = 25
MAX_JUMP = 60
MAX_POINTS = 8
INTERNAL_GAP_RATIO = 0.30

# ============================================================
# ★ [조정용] STATION 이후 우측 오프셋 중심선 (화면에 주황색 선)
#   0.0 = 도로 정중앙, 1.0 = 도로 오른쪽 가장자리, 음수 = 왼쪽
# ============================================================
CENTERLINE_RIGHT_OFFSET_RATIO = 0.15
RIGHT_OFFSET_STAGES = {"STOP2"}

# ============================================================
# STATION 이후 화살표를 흰 도로로 채워서 중심선이 흔들리지 않게 함
# ============================================================
ARROW_FILL_STAGES = {"STOP2"}
ARROW_GAP_RATIO = 0.80
ARROW_CLOSE_KERNEL = 15

# ============================================================
# ★ [조정용] 마지막 출구(STOP2 이후 우회전 -> GOAL 구간) 주행 안정화
# 교차로/조명 때문에 흰색 이진화가 끊겨서 버벅이는 문제 대응
# ============================================================
EXIT_STAGES = {"GOAL"}          # 적용 구간
EXIT_V_MIN = 120                # 자동 이진화 밝기 하한 (바닥 검정까지 흰색으로 잡히면 올릴 것)
EXIT_S_MAX = 90                 # 이 채도 이하만 흰색으로 인정 (색 있는 물체 제외)
EXIT_ERR_SMOOTH = 0.6           # 오차 평활화 0~1 (클수록 부드럽지만 반응 느림)
EXIT_LOST_STRAIGHT_SEC = 0.6    # 중심선을 놓쳐도 이 시간 동안은 제자리 탐색 대신 천천히 직진
EXIT_LOST_SPEED = 14
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# 시작 후 25초 동안은 YOLO 없이 중심선 추종
CENTERLINE_RUN_SEC = 25.0


# ============================================================
# YOLO
# ============================================================
MODEL_PATH = "best_ncnn_model"
YOLO_CONF = 0.45
YOLO_IMGSZ = 320
TEXT_CLASSES = {"STOP", "STATION", "GOAL"}

# ★ [조정용] 화살표 인식 3중 차단
#  1) 모델에 STOP/STATION/GOAL 클래스만 요청
#  2) 이름에 아래 단어가 있으면 버림
#  3) 박스 가로/세로 비율이 아래 값보다 작으면(글씨가 아닌 모양) 버림 -> 화면에 회색 REJECT
ARROW_BLOCK_WORDS = ("ARROW", "LEFT", "RIGHT", "STRAIGHT", "TURN", "화살")
TEXT_MIN_ASPECT = {"STOP": 1.5, "STATION": 2.0, "GOAL": 1.3}

# ★ [조정용] 구간별 최소 신뢰도
#   STOP2: 화살표가 있어서 더 엄격
#   GOAL : 모델 신뢰도가 낮게 나와서 낮춤 (엉뚱한 곳에 GOAL이 잡히면 0.30~0.35로 올릴 것)
STAGE_MIN_CONF = {"STATION": 0.35, "STOP2": 0.60, "GOAL": 0.25}

# route_stage 별로 찾아야 하는 글씨
STAGE_TARGET = {"STOP1": "STOP", "STATION": "STATION", "STOP2": "STOP", "GOAL": "GOAL"}
# STOP_3SEC 이 끝난 뒤 다음 route_stage (STOP2 는 FINAL_FORWARD 로 따로 처리)
NEXT_STAGE = {"STOP1": "STATION", "STATION": "STOP2"}

# YOLO를 돌리는 주행 상태
YOLO_STATES = {
    "CENTERLINE", "SEARCH_TEXT", "TEXT_SEEK_FULL",
    "APPROACH_TEXT", "DRIVE_TEXT", "GOAL_APPROACH",
}

# 중심선 주행 중 글씨가 이 프레임 수만큼 연속으로 보여야 이벤트 실행 (화살표 순간 오인식 무시)
EVENT2_CONFIRM_FRAMES = 3
EVENT2_CONFIRM_FRAMES_BY_TYPE = {"GOAL": 2, "STATION": 2}   # GOAL/STATION은 2프레임만 확인

# ============================================================
# ★ [조정용] STATION 인식 속도
# 이전: 확인 3프레임 -> 멈춤 -> 전체 보임 3프레임 더 확인 = 라즈베리파이에서 약 3초
# 지금: 확인하는 동안 이미 글자 전체가 보였으면 추가 확인 없이 바로 접근
# ============================================================
FAST_FULL_FRAMES = 2          # 확인 중 "전체 보임"이 이 프레임 이상이면 바로 APPROACH_TEXT
# 화살표 없는 구간에서는 한 프레임 놓쳐도 카운트를 0으로 리셋하지 않고 1만 깎음
# (STOP2 구간은 화살표 때문에 기존처럼 엄격하게 리셋)
EVENT2_STRICT_STAGES = {"STOP2"}

# ★ [조정용] 글자 전체 확인 (SEARCH_TEXT / TEXT_SEEK_FULL 공통)
FULL_MARGIN = 20           # 박스 좌우가 화면 끝에서 이 픽셀 이상 떨어져야 "전체 보임"
                           # (다시 STATIO에서 출발하면 30~35로 올릴 것)
FULL_STABLE_FRAMES = 2     # (SEARCH_TEXT용) 전체 보임이 연속 몇 프레임 유지돼야 출발할지

# ============================================================
# ★★★ [조정용] 고개 돌리기 (TEXT_SEEK_FULL: 글자가 잘려 보일 때 회전 정렬) ★★★
# 한 번에: SEEK_TURN_SPEED 속도로 SEEK_PULSE_ON 초 돌고 -> 멈춤 -> YOLO로 다시 확인
#   더 빨리 찾게 하려면 : SEEK_PULSE_ON 을 0.20 -> 0.25 로 (한 번에 더 많이 돔)
#   글자를 지나쳐 버리면 : SEEK_PULSE_ON 을 0.15 로 줄이기
#   회전이 약해서 안 돌면: SEEK_TURN_SPEED 를 18 -> 20 으로
# ============================================================
SEEK_TURN_SPEED = 18       # 고개 돌리는 속도 (이전 14)
SEEK_PULSE_ON = 0.20       # 한 번에 도는 시간(초) (이전 0.12)
SEEK_SETTLE = 0.05         # 돌고 나서 카메라 흔들림이 멎을 때까지 잠깐 대기(초)
SEEK_STABLE_FRAMES = 1     # 멈춘 뒤 글자 전체가 몇 프레임 보이면 출발 (이전 2)
                           # 다시 STATIO 에서 출발하면 2로 올릴 것
SEEK_LOST_TIMEOUT = 3.0    # 이 시간 이상 못 보면 중심선 추종으로 복귀

# 글씨 접근 / 정지
TEXT_ALIGN_TRIGGER_RATIO = 0.60
TEXT_BOTTOM_TRIGGER_RATIO = 0.98
FORWARD_AFTER_TEXT_SEC = 3.0   # STOP / STATION 공통: 박스 하단 도달 후 직진 시간
STOP_AFTER_TEXT_SEC = 3.0

# STOP2 이후 GOAL 구간
FINAL_STOP_FORWARD_SEC = 3.0
# ★ [조정용] STOP2 이후 마지막 우회전 각도
# 기존 0.90초 = 실제로 약 80도 -> 초당 약 89도로 계산
#   더 돌아야 하면 FINAL_TURN_DEG 를 올리고, 너무 돌면 내릴 것
#   각도가 맞는데 매번 조금씩 다르면 TURN_DEG_PER_SEC 를 현장에서 다시 측정
FINAL_TURN_DEG = 120
TURN_DEG_PER_SEC = 80 / 0.90
FINAL_TURN_SPEED = 18
FINAL_RIGHT_TURN_SEC = FINAL_TURN_DEG / TURN_DEG_PER_SEC   # 120도 -> 약 1.35초

# ★ [조정용] GOAL: 박스를 따라 직진 -> 박스가 사라지면 2초 더 직진 -> 완전 정지
GOAL_NEAR_RATIO = 0.70         # 박스 하단이 화면 70% 아래까지 오면 "가까이 왔다"
GOAL_MISS_TIMEOUT = 0.40       # 가까이 온 뒤 이 시간 동안 안 보이면 "사라졌다"
GOAL_GIVEUP_SEC = 3.0          # 멀리서 놓친 채 이 시간이 지나면 그래도 마무리 직진
GOAL_FINAL_FORWARD_SEC = 2.0   # 사라진 뒤 추가 직진 시간


# ============================================================
# YOLO 로드
# ============================================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} 파일이 없습니다. 이 코드와 같은 폴더에 넣어주세요.")

print("Loading YOLO model...")
yolo_model = YOLO(MODEL_PATH)

try:
    yolo_model.predict(
        np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8),
        imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False
    )
    print("YOLO warm-up done.")
except Exception as e:
    print("YOLO warm-up failed:", e)

_raw_names = yolo_model.names
_raw_names = _raw_names.items() if isinstance(_raw_names, dict) else enumerate(_raw_names)
CLASS_NAMES = {int(i): str(n).upper().strip() for i, n in _raw_names}
print("YOLO classes:", CLASS_NAMES)


def is_text_class(name):
    # [차단 2] 화살표 계열 이름은 무조건 제외
    return name in TEXT_CLASSES and not any(word in name for word in ARROW_BLOCK_WORDS)


# [차단 1] 허용 클래스 id 목록
ALLOWED_CLASS_IDS = [i for i, n in CLASS_NAMES.items() if is_text_class(n)] or None
if ALLOWED_CLASS_IDS is None:
    print("[WARN] STOP/STATION/GOAL class names not found in model -> classes filter off")
print("YOLO allowed class ids (arrow blocked):", ALLOWED_CLASS_IDS)


# ============================================================
# 상태
# ============================================================
class State:
    def __init__(self):
        self.manual_until = 0.0
        self.manual_cmd = None
        self.reset()

    def reset(self):
        self.auto_mode = False
        self.drive_state = "CENTERLINE"
        self.route_stage = "STOP1"      # STOP1 -> STATION -> STOP2 -> GOAL -> DONE
        self.state_start_time = 0.0
        self.last_error = 0

        self.auto_start_time = None
        self.initial_25s_done = False

        self.current_text_type = None
        self.text_candidate_type = None
        self.text_full_count = 0
        self.event2_hits = 0
        self.event2_full_hits = 0      # 확인 중 글자 전체가 보인 프레임 수

        self.seek_dir = 1               # 1 = 오른쪽 회전, -1 = 왼쪽 회전
        self.seek_lost_since = None

        self.goal_seen_near = False
        self.goal_last_seen = None

        self.exit_err = None            # 출구 구간 평활화된 오차
        self.exit_lost_since = None


S = State()
stop_event = threading.Event()
latest_jpeg = None
jpeg_lock = threading.Lock()
yolo_rejected = []   # 화면 표시용: 이번 프레임에서 모양 때문에 버려진 박스


def enter(state, msg=None):
    S.drive_state = state
    S.state_start_time = time.time()
    if msg:
        print(msg)


def state_elapsed():
    return time.time() - S.state_start_time


def start_auto(source):
    """AUTO 시작 공통 (노트북 ZMQ 신호 / p 키)"""
    if S.auto_mode:
        return
    S.auto_mode = True
    S.drive_state = "CENTERLINE"
    if not S.initial_25s_done:
        S.auto_start_time = time.time()
        print(f"[{source}] AUTO ON -> FIRST 25s CENTERLINE")
    else:
        print(f"[{source}] AUTO ON -> NORMAL CENTERLINE")


# ============================================================
# ZMQ: 노트북에서 START_AUTONAV 받으면 AUTO 시작
# (신호를 기다리는 동안에도 p 키로 AUTO를 켤 수 있음)
# ============================================================
def zmq_wait_for_start():
    try:
        socket = zmq.Context().socket(zmq.REQ)
        socket.connect(f"tcp://{LAPTOP_ZMQ_IP}:{LAPTOP_ZMQ_PORT}")

        print("[핑키봇] 노트북 ZMQ 서버에 접속합니다...")
        socket.send_string("핑키봇 준비 완료!")

        if json.loads(socket.recv_string()).get("status") == "START_AUTONAV":
            print("[핑키봇] START_AUTONAV 수신")
            start_auto("ZMQ")
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


def follow_centerline(err):
    """기본 주행: 중심선 추종(P제어 + 커브 감속), 선을 놓치면 마지막 오차 방향으로 탐색"""
    if err is not None:
        base = max(MIN_CURVE_SPEED, BASE_SPEED - CURVE_SLOWDOWN * abs(err))
        corr = KP * err
        drive(np.clip(base + corr, INNER_MIN_SPEED, MAX_SPEED),
              np.clip(base - corr, INNER_MIN_SPEED, MAX_SPEED))
    elif S.last_error < 0:
        drive(SEARCH_INNER_SPEED, SEARCH_SPEED)
    elif S.last_error > 0:
        drive(SEARCH_SPEED, SEARCH_INNER_SPEED)
    else:
        stop_robot()


def follow_exit(err):
    """출구(GOAL 구간) 주행: 오차를 부드럽게 + 선을 잠깐 놓치면 제자리 탐색 대신 천천히 직진"""
    if err is not None:
        S.exit_lost_since = None
        if S.exit_err is None:
            S.exit_err = err
        else:
            S.exit_err = EXIT_ERR_SMOOTH * S.exit_err + (1 - EXIT_ERR_SMOOTH) * err
        follow_centerline(S.exit_err)
        return

    if S.exit_lost_since is None:
        S.exit_lost_since = time.time()
    if time.time() - S.exit_lost_since < EXIT_LOST_STRAIGHT_SEC:
        drive(EXIT_LOST_SPEED, EXIT_LOST_SPEED)
    else:
        S.exit_err = None
        follow_centerline(None)


def follow_text(target, w, lost_scale=1.0):
    """YOLO 박스 중심을 화면 중앙에 맞추며 직진. 박스를 놓치면 lost_scale 배 속도로 직진"""
    if target is None:
        v = TEXT_FOLLOW_SPEED * lost_scale
        drive(v, v)
        return
    corr = np.clip(TEXT_FOLLOW_KP * (target["center"][0] - w // 2),
                   -TEXT_FOLLOW_MAX_CORR, TEXT_FOLLOW_MAX_CORR)
    drive(TEXT_FOLLOW_SPEED + corr, TEXT_FOLLOW_SPEED - corr)


def fully_visible(target, w):
    x1, _, x2, _ = target["bbox"]
    return x1 >= FULL_MARGIN and x2 <= w - FULL_MARGIN


def seek_turn_dir(target, w):
    """박스가 잘린 쪽으로 회전 방향 결정 (오른쪽 잘림 -> 1, 왼쪽 잘림 -> -1)"""
    x1, _, x2, _ = target["bbox"]
    left_cut = x1 < FULL_MARGIN
    right_cut = x2 > w - FULL_MARGIN
    if right_cut != left_cut:
        return 1 if right_cut else -1
    return 1 if (x1 + x2) // 2 >= w // 2 else -1


def seek_pulse(direction):
    """정확히 SEEK_PULSE_ON 초만 돌고 멈춤.
    이전 방식은 YOLO가 느려서(한 프레임 0.3~0.5초) 도는 시간이 들쭉날쭉하고
    멈춰서 기다리는 시간이 낭비됐음 -> 이제는 돌고 멈춘 직후 바로 YOLO가 봄"""
    drive(direction * SEEK_TURN_SPEED, -direction * SEEK_TURN_SPEED)
    time.sleep(SEEK_PULSE_ON)
    stop_robot()
    time.sleep(SEEK_SETTLE)


# ============================================================
# 마스크 / 중심선
# ============================================================
KERNEL_OPEN = np.ones((3, 3), np.uint8)
KERNEL_CLOSE = np.ones((7, 7), np.uint8)
KERNEL_ARROW = np.ones((ARROW_CLOSE_KERNEL, ARROW_CLOSE_KERNEL), np.uint8)


def row_groups(row):
    """한 행에서 연속된 흰 픽셀 묶음 중 MIN_WIDTH 이상인 것만 반환"""
    xs = np.where(row == 255)[0]
    if len(xs) == 0:
        return []
    groups = np.split(xs, np.where(np.diff(xs) > 1)[0] + 1)
    return [g for g in groups if len(g) >= MIN_WIDTH]


def remove_small_components(mask, min_area):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def fill_road_internal_gaps(mask, gap_ratio=INTERNAL_GAP_RATIO):
    filled = mask.copy()
    max_gap = int(mask.shape[1] * gap_ratio)
    for y in range(mask.shape[0]):
        groups = row_groups(mask[y])
        for left, right in zip(groups, groups[1:]):
            l_end, r_start = int(left[-1]), int(right[0])
            if 0 < r_start - l_end - 1 <= max_gap:
                filled[y, l_end:r_start + 1] = 255
    return filled


def fill_arrow_as_road(mask):
    """화살표를 흰 도로로 채움: 틈 메우기 -> 둘러싸인 구멍 채우기 -> 넓은 행 간격 메우기"""
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL_ARROW)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return fill_road_internal_gaps(filled, gap_ratio=ARROW_GAP_RATIO)


def exit_adaptive_white(hsv):
    """출구 구간 전용 자동 이진화: 밝기 대비 보정(CLAHE) + Otsu 자동 임계값
    고정 HSV(V>=175)가 조명 때문에 흰 선을 놓칠 때 보충해 줌"""
    v = CLAHE.apply(hsv[:, :, 2])
    otsu_t, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(EXIT_V_MIN, int(otsu_t))
    return (((v >= t) & (hsv[:, :, 1] <= EXIT_S_MAX)) * 255).astype(np.uint8)


def build_road_mask(roi):
    hsv = cv2.cvtColor(cv2.GaussianBlur(roi, (5, 5), 0), cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)

    # ★ 출구 구간: 고정 HSV + 자동 이진화 합치기
    if S.route_stage in EXIT_STAGES:
        white = cv2.bitwise_or(white, exit_adaptive_white(hsv))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, KERNEL_OPEN)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, KERNEL_CLOSE)
    white = remove_small_components(white, MIN_AREA)

    if S.route_stage in ARROW_FILL_STAGES:
        return fill_arrow_as_road(white)
    return fill_road_internal_gaps(white)


def find_center_points(road_mask, roi_start):
    """ROI 아래에서 위로 올라가며 도로 중심점과 도로 반폭을 찾음"""
    roi_h, roi_w = road_mask.shape
    points, half_widths = [], []

    for y in range(roi_h - 1, 0, -STEP):
        groups = row_groups(road_mask[y])
        if not groups:
            continue

        # 기준 x: 첫 점은 화면 중앙, 이후는 직전 기울기로 예측
        if not points:
            ref_x = roi_w // 2
        elif len(points) >= 2:
            ref_x = points[-1][0] + int(np.clip(points[-1][0] - points[-2][0], -35, 35))
        else:
            ref_x = points[-1][0]

        chosen = min(groups, key=lambda g: abs((int(g[0]) + int(g[-1])) // 2 - ref_x))
        x_left, x_right = int(chosen[0]), int(chosen[-1])
        x_center = (x_left + x_right) // 2

        if points and abs(x_center - points[-1][0]) > MAX_JUMP:
            continue

        points.append((x_center, y + roi_start))
        half_widths.append((x_right - x_left) / 2.0)

        if len(points) >= MAX_POINTS:
            break

    return points, half_widths


def pick_target(points, w):
    """3번째 점을 목표점으로 사용 -> (목표점, 화면 중앙 대비 오차)"""
    if len(points) < 3:
        return None, None
    return points[2], points[2][0] - w // 2


# ============================================================
# YOLO 글씨 검출
# ============================================================
def detect_text_yolo(frame, y_offset=0, expected_class=None):
    # 구간별 신뢰도를 모델 호출에도 적용 (GOAL처럼 0.45보다 낮은 값도 통과되게)
    min_conf = STAGE_MIN_CONF.get(S.route_stage, YOLO_CONF)
    results = yolo_model(frame, imgsz=YOLO_IMGSZ, conf=min_conf,
                         classes=ALLOWED_CLASS_IDS, verbose=False)
    if not results or results[0].boxes is None:
        return None
    candidates = []

    for box in results[0].boxes:
        name = CLASS_NAMES[int(box.cls[0])]
        conf = float(box.conf[0])

        if conf < min_conf or not is_text_class(name):
            continue
        if expected_class is not None and name != expected_class:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        y1, y2 = y1 + y_offset, y2 + y_offset

        # [차단 3] 글씨처럼 가로로 길지 않으면(화살표 모양) 버림
        aspect = max(1, x2 - x1) / max(1, y2 - y1)
        if aspect < TEXT_MIN_ASPECT.get(name, 1.5):
            yolo_rejected.append((x1, y1, x2, y2, name, aspect))
            continue

        candidates.append({
            "type": name,
            "conf": conf,
            "bbox": (x1, y1, x2, y2),
            "center": ((x1 + x2) // 2, (y1 + y2) // 2),
        })

    # 화면 아래쪽(가장 가까운) 박스 선택
    return max(candidates, key=lambda z: z["bbox"][3]) if candidates else None


# ============================================================
# 상태 전환 공통
# ============================================================
def start_goal_approach(msg):
    S.goal_seen_near = False
    S.goal_last_seen = time.time()
    enter("GOAL_APPROACH", msg)


def start_text_seek(target, w):
    """STOP / STATION: 박스 전체가 화면에 들어올 때까지 회전 정렬"""
    stop_robot()
    S.text_full_count = 0
    S.seek_dir = seek_turn_dir(target, w)
    S.seek_lost_since = None
    enter("TEXT_SEEK_FULL", f"[YOLO] {S.current_text_type} detected -> TEXT_SEEK_FULL")


def back_to_centerline(msg=None):
    S.current_text_type = None
    enter("CENTERLINE", msg)


# ============================================================
# 자동 주행 상태 머신 (한 프레임)
# ============================================================
def step_auto(t, w, h, error, offset_error, event2_confirmed):
    st = S.drive_state

    # ---------------- CENTERLINE (기본 주행) ----------------
    if st == "CENTERLINE":
        if S.auto_start_time is None:
            S.auto_start_time = time.time()

        # [이벤트 1] 시작 후 25초 -> 정지 후 제자리 회전하며 글씨 탐색
        if not S.initial_25s_done and time.time() - S.auto_start_time >= CENTERLINE_RUN_SEC:
            stop_robot()
            S.initial_25s_done = True
            S.text_full_count = 0
            S.text_candidate_type = None
            enter("SEARCH_TEXT", "[TIMER] first 25s centerline done -> SEARCH_TEXT")

        # [이벤트 2] 글씨가 연속 프레임으로 확인되면 글씨 처리 시작
        elif t is not None and event2_confirmed:
            S.current_text_type = t["type"]
            full_hits = S.event2_full_hits
            S.event2_hits = 0
            S.event2_full_hits = 0
            if S.current_text_type == "GOAL":
                start_goal_approach("[YOLO] GOAL detected -> GOAL_APPROACH")
            elif fully_visible(t, w) and full_hits >= FAST_FULL_FRAMES:
                # 이미 글자 전체가 보였음 -> 멈춰서 다시 확인하지 않고 바로 접근
                enter("APPROACH_TEXT", f"[YOLO] {S.current_text_type} already full -> APPROACH_TEXT (fast)")
            else:
                start_text_seek(t, w)

        # [이벤트 3] STATION 이후: 우측 오프셋 중심선
        elif S.route_stage in RIGHT_OFFSET_STAGES:
            follow_centerline(offset_error)

        # [출구] STOP2 이후 GOAL 구간: 부드러운 중심선 추종
        elif S.route_stage in EXIT_STAGES:
            follow_exit(error)

        # [기본] 도로 중심선
        else:
            follow_centerline(error)

    # ---------------- TEXT_SEEK_FULL (STOP / STATION) ----------------
    elif st == "TEXT_SEEK_FULL":
        if t is not None:
            S.seek_lost_since = None
            if fully_visible(t, w):
                stop_robot()
                S.text_full_count += 1
                if S.text_full_count >= SEEK_STABLE_FRAMES:
                    S.text_full_count = 0
                    enter("APPROACH_TEXT", f"[YOLO] {S.current_text_type} all letters visible -> APPROACH_TEXT")
            else:
                S.text_full_count = 0
                S.seek_dir = seek_turn_dir(t, w)
                seek_pulse(S.seek_dir)
        else:
            S.text_full_count = 0
            if S.seek_lost_since is None:
                S.seek_lost_since = time.time()

            if time.time() - S.seek_lost_since >= SEEK_LOST_TIMEOUT:
                stop_robot()
                back_to_centerline(f"[YOLO] {S.current_text_type} lost too long -> back to CENTERLINE")
            else:
                seek_pulse(S.seek_dir)

    # ---------------- SEARCH_TEXT (25초 직후 제자리 회전 탐색) ----------------
    elif st == "SEARCH_TEXT":
        full_target = None

        if t is not None and fully_visible(t, w):
            if S.text_candidate_type == t["type"]:
                S.text_full_count += 1
            else:
                S.text_candidate_type = t["type"]
                S.text_full_count = 1
            if S.text_full_count >= FULL_STABLE_FRAMES:
                full_target = t
        else:
            S.text_full_count = 0
            S.text_candidate_type = None

        if full_target is None:
            drive(TURN_SPEED, -TURN_SPEED)
        else:
            stop_robot()
            S.current_text_type = full_target["type"]
            if S.current_text_type == "GOAL":
                start_goal_approach("[YOLO] GOAL full -> GOAL_APPROACH")
            else:
                enter("APPROACH_TEXT", f"[YOLO] {S.current_text_type} full -> APPROACH_TEXT")

    # ---------------- APPROACH_TEXT: 박스 중심 맞추며 접근 ----------------
    elif st == "APPROACH_TEXT":
        follow_text(t, w)
        if t is not None and t["bbox"][3] >= int(h * TEXT_ALIGN_TRIGGER_RATIO):
            enter("DRIVE_TEXT", f"[YOLO] {t['type']} reached 60% -> DRIVE_TEXT")

    # ---------------- DRIVE_TEXT: 박스 하단이 화면 바닥에 닿을 때까지 ----------------
    elif st == "DRIVE_TEXT":
        if t is not None and t["bbox"][3] >= int(h * TEXT_BOTTOM_TRIGGER_RATIO):
            stop_robot()
            if S.current_text_type == "GOAL":
                S.goal_seen_near = True
                S.goal_last_seen = time.time()
                enter("GOAL_APPROACH", "[YOLO] GOAL near -> GOAL_APPROACH")
            else:
                enter("FORWARD_2SEC", f"[YOLO] {S.current_text_type} bbox bottom reached -> forward {FORWARD_AFTER_TEXT_SEC:.1f}s")
        else:
            follow_text(t, w)

    # ---------------- FORWARD_2SEC -> STOP_3SEC ----------------
    elif st == "FORWARD_2SEC":
        if state_elapsed() < FORWARD_AFTER_TEXT_SEC:
            drive(TEXT_FOLLOW_SPEED, TEXT_FOLLOW_SPEED)
        else:
            stop_robot()
            enter("STOP_3SEC", f"[TEXT] forward {FORWARD_AFTER_TEXT_SEC:.1f}s done -> stop {STOP_AFTER_TEXT_SEC:.1f}s")

    # ---------------- STOP_3SEC: 정지 후 다음 구간 ----------------
    elif st == "STOP_3SEC":
        stop_robot()
        if state_elapsed() >= STOP_AFTER_TEXT_SEC:
            prev_stage = S.route_stage
            if prev_stage in NEXT_STAGE:
                S.route_stage = NEXT_STAGE[prev_stage]
                back_to_centerline(f"[ROUTE] {prev_stage} done -> CENTERLINE / SEARCH {S.route_stage}")
            elif prev_stage == "STOP2":
                S.current_text_type = None
                enter("FINAL_FORWARD", f"[ROUTE] STOP2 done -> FINAL_FORWARD {FINAL_STOP_FORWARD_SEC:.1f}s")
            else:
                back_to_centerline()

    # ---------------- FINAL_FORWARD -> FINAL_RIGHT_TURN -> GOAL 탐색 ----------------
    elif st == "FINAL_FORWARD":
        if state_elapsed() < FINAL_STOP_FORWARD_SEC:
            drive(TEXT_FOLLOW_SPEED, TEXT_FOLLOW_SPEED)
        else:
            stop_robot()
            enter("FINAL_RIGHT_TURN", f"[ROUTE] final forward {FINAL_STOP_FORWARD_SEC:.1f}s done -> right turn {FINAL_TURN_DEG}deg ({FINAL_RIGHT_TURN_SEC:.2f}s)")

    elif st == "FINAL_RIGHT_TURN":
        if state_elapsed() < FINAL_RIGHT_TURN_SEC:
            drive(FINAL_TURN_SPEED, -FINAL_TURN_SPEED)
        else:
            stop_robot()
            S.route_stage = "GOAL"
            S.exit_err = None          # 회전 전 오차 기억 지우고 새로 중심선 시작
            S.exit_lost_since = None
            enter("CENTERLINE", f"[ROUTE] final right turn {FINAL_TURN_DEG}deg done -> CENTERLINE / SEARCH GOAL")

    # ---------------- GOAL ----------------
    # GOAL 박스를 따라 직진. 박스가 안 보여도 멈추지 않고 그대로 직진
    elif st == "GOAL_APPROACH":
        follow_text(t, w)
        if t is not None:
            S.goal_last_seen = time.time()
            if t["bbox"][3] >= int(h * GOAL_NEAR_RATIO):
                S.goal_seen_near = True
        else:
            missing = time.time() - S.goal_last_seen
            if S.goal_seen_near and missing >= GOAL_MISS_TIMEOUT:
                enter("GOAL_FINAL_FORWARD",
                      f"[GOAL] box gone -> forward {GOAL_FINAL_FORWARD_SEC:.1f}s more")
            elif missing >= GOAL_GIVEUP_SEC:
                enter("GOAL_FINAL_FORWARD",
                      f"[GOAL] lost {GOAL_GIVEUP_SEC:.1f}s -> forward {GOAL_FINAL_FORWARD_SEC:.1f}s more")

    # 박스가 사라진 뒤 2초 더 직진 -> 완전 정지
    elif st == "GOAL_FINAL_FORWARD":
        if state_elapsed() < GOAL_FINAL_FORWARD_SEC:
            drive(TEXT_FOLLOW_SPEED, TEXT_FOLLOW_SPEED)
        else:
            stop_robot()
            S.route_stage = "DONE"
            S.auto_mode = False
            enter("DONE", "[GOAL] final forward done -> ROBOT STOP / DONE")

    elif st == "DONE":
        stop_robot()


# ============================================================
# 화면 표시
# ============================================================
YELLOW, GREEN, RED, BLUE = (0, 255, 255), (0, 255, 0), (0, 0, 255), (255, 0, 0)
ORANGE, GRAY, MAGENTA, CYAN = (0, 165, 255), (128, 128, 128), (255, 0, 255), (255, 255, 0)


def put(img, text, org, color, scale=0.52, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def draw_polyline(img, points, line_color, dot_color):
    for p, q in zip(points, points[1:]):
        cv2.line(img, p, q, line_color, 3)
    for p in points:
        cv2.circle(img, p, 4, dot_color, -1)


def render(frame, roi_start, road_mask, center_points, target, offset_points, offset_target, t):
    h, w = frame.shape[:2]
    img = frame.copy()

    cv2.line(img, (0, roi_start), (w, roi_start), YELLOW, 2)
    cv2.line(img, (w // 2, roi_start), (w // 2, h), GREEN, 2)

    draw_polyline(img, center_points, BLUE, RED)
    if target is not None:
        cv2.circle(img, target, 9, YELLOW, -1)

    # ★ STATION 이후 실제로 따라가는 우측 오프셋 중심선 (주황색)
    if S.route_stage in RIGHT_OFFSET_STAGES:
        draw_polyline(img, offset_points, ORANGE, ORANGE)
        if offset_target is not None:
            cv2.circle(img, offset_target, 9, ORANGE, 2)
        put(img, f"EVENT3 R-OFFSET: {CENTERLINE_RIGHT_OFFSET_RATIO:+.2f}", (12, 76), ORANGE)

    # ★ 모양 때문에 버려진 박스(화살표 등) - 회색
    for x1, y1, x2, y2, name, aspect in yolo_rejected:
        cv2.rectangle(img, (x1, y1), (x2, y2), GRAY, 1)
        put(img, f"REJECT {name} {aspect:.1f}", (x1, max(20, y1 - 6)), GRAY, 0.45, 1)

    if t is not None:
        x1, y1, x2, y2 = t["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), MAGENTA, 2)
        put(img, f"{t['type']} {t['conf']:.2f}", (x1, max(20, y1 - 8)), MAGENTA, 0.60)
        cv2.circle(img, t["center"], 7, RED, -1)

    if S.route_stage in EXIT_STAGES:
        put(img, "EXIT: adaptive binarize + smooth", (12, 76), ORANGE)

    shown_state = S.drive_state if S.auto_mode else f"PAUSED / {S.drive_state}"
    put(img, f"STATE: {shown_state}", (12, 28), YELLOW, 0.60)
    put(img, f"ROUTE: {S.route_stage}", (12, 52), CYAN)

    if (S.auto_mode and S.drive_state == "CENTERLINE"
            and not S.initial_25s_done and S.auto_start_time is not None):
        remain = max(0.0, CENTERLINE_RUN_SEC - (time.time() - S.auto_start_time))
        put(img, f"CENTERLINE TIMER: {remain:.1f}s", (12, 55), YELLOW)

    road_bgr = cv2.resize(cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR), (w, h),
                          interpolation=cv2.INTER_NEAREST)
    return np.hstack([img, road_bgr])


# ============================================================
# Main control loop
# ============================================================
def control_loop():
    global latest_jpeg

    while not stop_event.is_set():
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.02)
            continue

        frame = frame.copy()
        h, w = frame.shape[:2]
        roi_start = int(h * ROI_START_RATIO)
        roi = frame[roi_start:h, :]

        # ---------------- 중심선 ----------------
        road_mask = build_road_mask(roi)
        center_points, half_widths = find_center_points(road_mask, roi_start)
        target, error = pick_target(center_points, w)
        if error is not None:
            S.last_error = error

        # 우측 오프셋 중심선: 오프셋 = RATIO x 도로 반폭
        offset_points = [
            (int(cx + CENTERLINE_RIGHT_OFFSET_RATIO * hw), cy)
            for (cx, cy), hw in zip(center_points, half_widths)
        ]
        offset_target, offset_error = pick_target(offset_points, w)

        # ---------------- YOLO (25초 이후) ----------------
        t = None
        yolo_rejected.clear()
        if S.initial_25s_done and S.drive_state in YOLO_STATES:
            try:
                t = detect_text_yolo(roi, y_offset=roi_start,
                                     expected_class=STAGE_TARGET.get(S.route_stage))
            except Exception as e:
                print("YOLO ERROR:", e)

        # 이벤트 2 연속 확인 카운트 (CENTERLINE에서만)
        if S.drive_state == "CENTERLINE" and t is not None:
            S.event2_hits += 1
            S.event2_full_hits = S.event2_full_hits + 1 if fully_visible(t, w) else 0
        elif S.drive_state == "CENTERLINE" and S.route_stage not in EVENT2_STRICT_STAGES:
            # 화살표 없는 구간: 한 프레임 놓쳐도 1만 깎음
            S.event2_hits = max(0, S.event2_hits - 1)
        else:
            S.event2_hits = 0
            S.event2_full_hits = 0
        need = EVENT2_CONFIRM_FRAMES_BY_TYPE.get(t["type"], EVENT2_CONFIRM_FRAMES) if t else EVENT2_CONFIRM_FRAMES
        event2_confirmed = S.event2_hits >= need

        # ---------------- 주행 ----------------
        if time.time() < S.manual_until and S.manual_cmd is not None:
            drive(*S.manual_cmd)
        elif not S.auto_mode:
            stop_robot()
        else:
            step_auto(t, w, h, error, offset_error, event2_confirmed)

        # ---------------- 화면 ----------------
        combo = render(frame, roi_start, road_mask, center_points, target,
                       offset_points, offset_target, t)
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
body{background:#111;color:white;font-family:Arial;text-align:center}
img{width:95%;max-width:1200px;border:2px solid #555}
button{font-size:24px;margin:5px;padding:10px 20px}
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
function cmd(k){ fetch('/cmd/'+k); }
document.addEventListener('keydown', function(e){
    if(e.repeat) return;
    let k=e.key.toLowerCase();
    if(e.code==='Space'){ e.preventDefault(); cmd('space'); }
    else if(['w','a','s','d','p','r'].includes(k)){ cmd(k); }
});
</script>
</body>
</html>
"""

MANUAL_CMDS = {
    "w": (MANUAL_SPEED, MANUAL_SPEED),
    "s": (-MANUAL_SPEED, -MANUAL_SPEED),
    "a": (-MANUAL_SPEED, MANUAL_SPEED),
    "d": (MANUAL_SPEED, -MANUAL_SPEED),
}


@app.route("/")
def index():
    return render_template_string(HTML)


def mjpeg():
    while True:
        with jpeg_lock:
            data = latest_jpeg
        if data is not None:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + data + b'\r\n'
        time.sleep(0.04)


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/cmd/<key>")
def command(key):
    if key == "p":
        if S.auto_mode:
            S.auto_mode = False
            stop_robot()
            print("AUTO OFF")
        else:
            start_auto("KEY")

    elif key == "r":
        S.reset()
        stop_robot()
        print("RESET")

    elif key == "space":
        S.auto_mode = False
        stop_robot()

    elif key in MANUAL_CMDS:
        S.manual_cmd = MANUAL_CMDS[key]
        S.manual_until = time.time() + MANUAL_PULSE

    return "OK"


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    threading.Thread(target=control_loop, daemon=True).start()
    threading.Thread(target=zmq_wait_for_start, daemon=True).start()

    print("Pinky server started")
    print(f"http://ROBOT_IP:{PORT}")

    try:
        app.run(host="0.0.0.0", port=PORT, threaded=True, debug=False)
    finally:
        stop_event.set()
        stop_robot()
