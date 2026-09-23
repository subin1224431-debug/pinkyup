import cv2
import numpy as np
import requests
import time
from ultralytics import YOLO
import os


# =========================================================
# Pinky Pro 주소
# =========================================================
HOST = "http://192.168.4.1:8000"


# =========================================================
# 주행 설정
# =========================================================
BASE_SPEED = 25
KP = 0.12
MAX_SPEED = 40
SEARCH_SPEED = 18

TEXT_FOLLOW_SPEED = 19
TEXT_FOLLOW_KP = 0.10
TEXT_FOLLOW_MAX_CORR = 10

TURN_SPEED = 30
ALIGN_SPEED = 10
CENTER_TOL = 30


# =========================================================
# ROI
# 기존 중심선 추종 코드 그대로: 화면 아래쪽 32%
# =========================================================
ROI_START_RATIO = 0.68


# =========================================================
# 흰색 / 검은색 HSV
#
# 중심선용 이진화에서는 흰색과 검은색만 사용한다.
# 검은 화살표는 별도로 검출하지만,
# 중심선 계산용 road_mask에서는 흰색으로 메워 제거한다.
# =========================================================
LOWER_WHITE = np.array([0, 0, 175])
UPPER_WHITE = np.array([180, 75, 255])

LOWER_BLACK = np.array([0, 0, 0])
UPPER_BLACK = np.array([180, 130, 115])


# =========================================================
# 중심선 검출 설정
# =========================================================
MIN_AREA = 500
STEP = 15
MIN_WIDTH = 25
MAX_JUMP = 60
MAX_POINTS = 8

# 도로 내부 검은 화살표를 메울 최대 간격
INTERNAL_GAP_RATIO = 0.30


# =========================================================
# 카메라 화살표 카운팅
#
# IR로 화살표를 카운트하지 않는다.
# 카메라에서 화살표가 새로 나타날 때마다 1회만 카운트한다.
# 3번째 화살표까지만 사용한다.
# =========================================================
ARROW_MIN_AREA = 350
ARROW_STABLE_FRAMES = 3
ARROW_GONE_FRAMES = 8

arrow_count = 0
arrow_stable_count = 0
arrow_missing_count = 0
arrow_count_armed = True


# =========================================================
# 3번째 화살표 거리 기준선
#
# 이전 코드 COUNT 2의 거리 기준 방식을 그대로 가져온다.
# 화살표 bbox 아래쪽 끝이 ROI 높이의 0.26 지점까지 내려오면
# 2프레임 연속 확인 후 오른쪽 회전을 시작한다.
# =========================================================
THIRD_ARROW_TRIGGER_RATIO = 0.26
THIRD_ARROW_TRIGGER_FRAMES = 2
third_arrow_trigger_count = 0


# =========================================================
# YOLO STOP / STATION
# =========================================================
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


# =========================================================
# IR
#
# 화살표 카운팅에는 사용하지 않는다.
# YOLO 글씨를 향해 주행한 뒤 검은 표식을 실제로 밟았을 때만
# 후진 동작의 트리거로 사용한다.
#
# 이 클라이언트 코드는 Pinky 서버의 /ir 엔드포인트에서
# {"l":값, "c":값, "r":값} 형식 JSON을 받는 것을 전제로 한다.
# =========================================================
IR_THRESHOLD = 2600

# IR은 평소에는 완전히 무시한다.
# YOLO가 STOP/STATION을 실제로 인식한 뒤에만 딱 한 번 활성화되고,
# 그 다음 처음 들어오는 IR 감지만 사용한다.
yolo_ir_waiting = False
yolo_ir_consumed = False

IR_REVERSE_SEC = 2.0
IR_STOP_SEC = 3.0


# =========================================================
# 주행 단계
#
# 카운트별로 서로 다른 주행 상태를 만들지 않는다.
#
# CENTERLINE  : 기존 중심선 추종
# SEARCH_TEXT : 3번째 화살표 기준선 도달 후 오른쪽 회전하며 YOLO 탐색
# ALIGN_TEXT  : YOLO 글씨 중앙 정렬
# DRIVE_TEXT  : 글씨 방향으로 직진
# REVERSE_IR  : IR 감지 후 2초 후진
# STOP_3SEC   : 후진 후 3초 정지
#
# 이후 다시 CENTERLINE으로 복귀한다.
# =========================================================
drive_state = "CENTERLINE"
auto_mode = False

last_error = 0
state_start_time = 0.0


# =========================================================
# 모터 제어
# =========================================================
def drive(left, right):
    left = int(np.clip(left, -100, 100))
    right = int(np.clip(right, -100, 100))

    try:
        requests.post(
            HOST + "/drive",
            json={"l": left, "r": right},
            timeout=0.3
        )
    except requests.RequestException:
        pass


def stop():
    try:
        requests.post(
            HOST + "/stop",
            timeout=0.3
        )
    except requests.RequestException:
        pass


def read_ir():
    """
    Pinky 서버의 /ir 엔드포인트에서 IR 값을 읽는다.
    예상 응답:
        {"l": 1234, "c": 1234, "r": 1234}
    """
    try:
        r = requests.get(
            HOST + "/ir",
            timeout=0.20
        )
        data = r.json()

        ir_l = int(data.get("l", 0))
        ir_c = int(data.get("c", 0))
        ir_r = int(data.get("r", 0))

        return ir_l, ir_c, ir_r

    except Exception:
        return 0, 0, 0


# =========================================================
# 작은 흰색 노이즈 제거
# =========================================================
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


# =========================================================
# 중심선 계산용 화살표 제거
#
# 흰색 도로 사이의 간격이 실제 검은색일 때만
# 그 검은 부분을 흰색으로 메워서 road_mask에서는 없앤다.
# =========================================================
def fill_road_internal_gaps(white_mask, black_mask):
    """
    중심선 계산용 road_mask를 만든다.

    핵심:
    - 흰색 도로 사이에 끼어 있는 내부 영역은 색과 상관없이 흰 도로로 메운다.
      따라서 황토색 박스도 중심선용 이진화에서는 흰 도로와 동일하게 처리된다.
    - 검은 화살표도 흰 도로 사이에 있으면 같이 흰색으로 메워져 중심선에서 제거된다.
    - black_mask는 카메라 화살표 검출용으로는 따로 유지한다.
    """

    filled = white_mask.copy()

    h, w = white_mask.shape
    max_internal_gap = int(w * INTERNAL_GAP_RATIO)

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

        for i in range(len(groups) - 1):
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

            # 흰 도로 사이의 작은 내부 간격은
            # 검은 화살표든 황토색 박스든 모두 흰 도로로 메운다.
            if (
                gap > 0
                and gap <= max_internal_gap
            ):
                filled[
                    y,
                    left_end:right_start + 1
                ] = 255

    return filled


# =========================================================
# 카메라 화살표 검출
#
# 카운팅용이다.
# 중심선 계산은 이 화살표를 지운 road_mask로 따로 수행한다.
# =========================================================
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

        aspect = bw / max(bh, 1)

        # 너무 길쭉한 검은 글자열/선은 화살표 후보에서 제외
        if aspect > 2.8:
            continue

        # ROI 대부분을 차지하는 거대한 외곽 검정은 제외
        if bw > int(w * 0.80) or bh > int(h * 0.95):
            continue

        candidates.append({
            "bbox": (x, y, bw, bh),
            "area": float(area),
            "center": (x + bw // 2, y + bh // 2)
        })

    if not candidates:
        return None

    # 화면에서 가장 아래쪽 = 가장 가까운 검은 화살표 우선
    return max(
        candidates,
        key=lambda z: (
            z["bbox"][1] + z["bbox"][3],
            z["area"]
        )
    )


# =========================================================
# YOLO STOP/STATION 검출
# =========================================================
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
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = str(yolo_model.names[cls_id]).upper().strip()

        if name not in TEXT_CLASSES:
            continue

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        candidates.append({
            "type": name,
            "conf": conf,
            "bbox": (x1, y1, x2, y2),
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


# =========================================================
# 카메라 연결
# =========================================================
cap = cv2.VideoCapture(
    HOST + "/video"
)

if not cap.isOpened():
    print("카메라 연결 실패")
    raise SystemExit


print("""
================================
Pinky Pro Centerline + Camera Arrow Count + YOLO
================================

P       : AUTO ON / OFF
W       : 수동 전진
S       : 수동 후진
A       : 수동 좌회전
D       : 수동 우회전
SPACE   : 정지
R       : 초기화
ESC     : 종료
""")


# =========================================================
# 메인 루프
# =========================================================
while True:

    ret, frame = cap.read()

    if not ret:
        print("영상 수신 실패")
        break

    h, w = frame.shape[:2]

    # =====================================================
    # 1. ROI
    # =====================================================
    roi_start = int(
        h * ROI_START_RATIO
    )

    roi = frame[
        roi_start:h,
        :
    ]

    roi_h, roi_w = roi.shape[:2]


    # =====================================================
    # 2. Blur + HSV
    # =====================================================
    roi_blur = cv2.GaussianBlur(
        roi,
        (5, 5),
        0
    )

    hsv = cv2.cvtColor(
        roi_blur,
        cv2.COLOR_BGR2HSV
    )


    # =====================================================
    # 3. 흰색 / 검은색만 이진화
    # =====================================================
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


    # =====================================================
    # 4. 작은 노이즈 제거
    # =====================================================
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


    # =====================================================
    # 5. 흰 도로 작은 틈 메우기
    # =====================================================
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


    # =====================================================
    # 6. 작은 흰색 잡영 제거
    # =====================================================
    white_mask = remove_small_components(
        white_mask,
        MIN_AREA
    )


    # =====================================================
    # 7. 중심선용 road_mask
    # 검은 화살표는 흰색으로 메워서 없앤다.
    # =====================================================
    road_mask = fill_road_internal_gaps(
        white_mask,
        black_mask
    )


    # =====================================================
    # 8. 중심선 계산
    # =====================================================
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

        x_left = int(chosen[0])
        x_right = int(chosen[-1])

        x_center = (
            x_left + x_right
        ) // 2

        if prev_center is not None:
            if abs(
                x_center - prev_center
            ) > MAX_JUMP:
                continue

        original_y = (
            y + roi_start
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


    # =====================================================
    # 9. 중심선 목표점 / 오차
    # =====================================================
    target_x = None
    target_y = None
    error = None

    if len(center_points) >= 3:
        target_x, target_y = center_points[2]

    if target_x is not None:
        error = (
            target_x
            - w // 2
        )

        last_error = error


    # =====================================================
    # 10. 카메라 화살표 검출 + 카운팅
    #
    # 카운트별 주행상태는 없다.
    # 화살표 번호는 "3번째 화살표인지" 확인하는 용도로만 쓴다.
    # =====================================================
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

                if arrow_stable_count >= ARROW_STABLE_FRAMES:
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

                if arrow_missing_count >= ARROW_GONE_FRAMES:
                    arrow_count_armed = True
                    arrow_missing_count = 0

    else:
        arrow_stable_count = 0


    # =====================================================
    # 11. 3번째 화살표 거리 기준선
    #
    # 카메라가 3번째 화살표로 카운트한 뒤,
    # 그 화살표 bbox 아래쪽 끝이 기준선까지 내려오면 우회전.
    # =====================================================
    if (
        drive_state == "CENTERLINE"
        and arrow_count == 3
        and arrow_target is not None
    ):
        ax, ay, aw, ah = arrow_target["bbox"]

        arrow_bottom = ay + ah

        trigger_y = int(
            roi_h
            * THIRD_ARROW_TRIGGER_RATIO
        )

        if arrow_bottom >= trigger_y:
            third_arrow_trigger_count += 1
        else:
            third_arrow_trigger_count = 0

        if (
            third_arrow_trigger_count
            >= THIRD_ARROW_TRIGGER_FRAMES
        ):
            stop()

            drive_state = "SEARCH_TEXT"
            state_start_time = time.time()

            third_arrow_trigger_count = 0

            text_full_count = 0
            text_candidate_type = None

            print(
                "[ARROW 3] distance line reached -> SEARCH_TEXT"
            )


    # =====================================================
    # 12. YOLO
    #
    # 3번째 화살표 이후부터만 주행 판단에 사용.
    # bbox는 SEARCH / ALIGN / DRIVE에서 화면에 계속 표시한다.
    # =====================================================
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
            print("YOLO ERROR:", e)
            text_target = None


    # =====================================================
    # 13. IR
    #
    # 화살표 카운팅에는 전혀 사용하지 않는다.
    # DRIVE_TEXT 상태에서만 후진 트리거로 사용.
    # =====================================================
    ir_l, ir_c, ir_r = read_ir()

    ir_hit = (
        ir_l >= IR_THRESHOLD
        or ir_c >= IR_THRESHOLD
        or ir_r >= IR_THRESHOLD
    )


    # =====================================================
    # 14. 자동주행
    # =====================================================
    if auto_mode:

        # -------------------------------------------------
        # CENTERLINE
        # 기존 중심선 추종
        # -------------------------------------------------
        if drive_state == "CENTERLINE":

            # 첫 3번째 화살표 특수 진입이 끝난 뒤에는
            # 중심선 추종 중 다음 STOP/STATION이 YOLO로 다시 보이면
            # 새로운 IR 1회 사이클을 시작한다.
            if (
                arrow_count >= 3
                and text_target is not None
                and not yolo_ir_waiting
                and not yolo_ir_consumed
            ):
                stop()
                yolo_ir_waiting = True
                drive_state = "ALIGN_TEXT"

                print(
                    f"[YOLO] next {text_target['type']} detected during CENTERLINE -> ALIGN_TEXT / wait next IR"
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
                    stop()


        # -------------------------------------------------
        # SEARCH_TEXT
        # 3번째 화살표 기준선 도달 후 오른쪽 제자리 회전
        # STOP/STATION 전체가 보일 때까지 탐색
        # -------------------------------------------------
        elif drive_state == "SEARCH_TEXT":

            full_text_target = None

            if text_target is not None:
                x1, y1, x2, y2 = text_target["bbox"]

                text_fully_inside = (
                    x1 >= TEXT_FULL_MARGIN
                    and x2 <= (
                        w
                        - TEXT_FULL_MARGIN
                    )
                )

                current_type = text_target["type"]

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
                stop()

                # YOLO 글씨가 확실히 인식된 이 시점부터만
                # '다음에 처음 들어오는 IR 1회'를 기다린다.
                if not yolo_ir_consumed:
                    yolo_ir_waiting = True
                    print("[IR ARM] YOLO recognized -> waiting for the NEXT IR only")

                drive_state = "ALIGN_TEXT"

                print(
                    f"[YOLO] {full_text_target['type']} full -> ALIGN_TEXT"
                )

            else:
                drive(
                    TURN_SPEED,
                    -TURN_SPEED
                )


        # -------------------------------------------------
        # ALIGN_TEXT
        # YOLO bbox 중심을 화면 중앙으로 제자리 정렬
        # -------------------------------------------------
        elif drive_state == "ALIGN_TEXT":

            if text_target is None:
                drive_state = "SEARCH_TEXT"

            else:
                text_x = text_target["center"][0]
                text_error = text_x - w // 2

                if abs(text_error) <= CENTER_TOL:
                    stop()
                    drive_state = "DRIVE_TEXT"

                    print(
                        f"[YOLO] {text_target['type']} centered -> DRIVE_TEXT"
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


        # -------------------------------------------------
        # DRIVE_TEXT
        # YOLO 글씨를 보면서 직진.
        # 여기서만 IR을 후진 트리거로 사용.
        # -------------------------------------------------
        elif drive_state == "DRIVE_TEXT":

            # IR은 YOLO 인식 이후에 arm된 경우에만 딱 한 번 사용한다.
            if (
                yolo_ir_waiting
                and not yolo_ir_consumed
                and ir_hit
            ):
                yolo_ir_waiting = False
                yolo_ir_consumed = True

                drive(
                    -BASE_SPEED,
                    -BASE_SPEED
                )

                state_start_time = time.time()
                drive_state = "REVERSE_IR"

                print(
                    "[IR] FIRST IR after YOLO -> reverse 2.0s"
                )

            elif text_target is not None:
                text_x = text_target["center"][0]
                text_error = text_x - w // 2

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
                # 글씨가 카메라 아래로 빠진 뒤에는
                # IR을 밟을 때까지 직진한다.
                drive(
                    TEXT_FOLLOW_SPEED,
                    TEXT_FOLLOW_SPEED
                )


        # -------------------------------------------------
        # REVERSE_IR
        # IR 감지 후 2초 후진
        # -------------------------------------------------
        elif drive_state == "REVERSE_IR":

            if (
                time.time()
                - state_start_time
                < IR_REVERSE_SEC
            ):
                drive(
                    -BASE_SPEED,
                    -BASE_SPEED
                )

            else:
                stop()
                state_start_time = time.time()
                drive_state = "STOP_3SEC"

                print(
                    "[IR] reverse done -> stop 3.0s"
                )


        # -------------------------------------------------
        # STOP_3SEC
        # 3초 정지 후 중심선 추종으로 복귀하고 다음 YOLO를 다시 기다림
        # -------------------------------------------------
        elif drive_state == "STOP_3SEC":

            stop()

            if (
                time.time()
                - state_start_time
                >= IR_STOP_SEC
            ):
                # 이번 YOLO -> IR 1회 사이클 완료.
                # 다음 YOLO 글씨를 다시 인식했을 때
                # 새로운 IR 1회를 기다릴 수 있도록 초기화한다.
                yolo_ir_waiting = False
                yolo_ir_consumed = False

                drive_state = "CENTERLINE"

                print(
                    "[IR] 3s stop done -> CENTERLINE / ready for next YOLO"
                )


    else:
        stop()


    # =====================================================
    # 15. 결과 화면
    # =====================================================
    result = frame.copy()


    # ROI 시작선
    cv2.line(
        result,
        (0, roi_start),
        (w, roi_start),
        (0, 255, 255),
        2
    )


    # 화면 중앙선
    cv2.line(
        result,
        (w // 2, roi_start),
        (w // 2, h),
        (0, 255, 0),
        2
    )


    # 중심점
    for x, y in center_points:
        cv2.circle(
            result,
            (x, y),
            4,
            (0, 0, 255),
            -1
        )


    # 중심선
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


    # 중심선 목표점
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


    # 카메라 화살표 bbox
    if arrow_target is not None:
        ax, ay, aw, ah = arrow_target["bbox"]

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


    # 3번째 화살표가 잡힌 뒤 거리 기준선 표시
    if (
        arrow_count == 3
        and drive_state == "CENTERLINE"
    ):
        third_line_y = (
            roi_start
            + int(
                roi_h
                * THIRD_ARROW_TRIGGER_RATIO
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
            f"ARROW3 DIST LINE {THIRD_ARROW_TRIGGER_RATIO:.2f}",
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


    # YOLO bbox
    if text_target is not None:
        x1, y1, x2, y2 = text_target["bbox"]

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


    # 화면 정보
    mode_text = (
        "AUTO"
        if auto_mode
        else "MANUAL"
    )

    cv2.putText(
        result,
        f"MODE: {mode_text}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )

    cv2.putText(
        result,
        f"STATE: {drive_state}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )

    cv2.putText(
        result,
        f"CAM ARROW: {arrow_count}/3",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 0),
        2
    )

    cv2.putText(
        result,
        f"IR L:{ir_l} C:{ir_c} R:{ir_r}",
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 0, 0),
        2
    )

    cv2.putText(
        result,
        f"IR AFTER YOLO: {yolo_ir_waiting and not yolo_ir_consumed}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 0, 0),
        2
    )

    if error is not None:
        cv2.putText(
            result,
            f"Center Error: {error}",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 255),
            2
        )


    # =====================================================
    # 16. 영상 출력
    # =====================================================
    cv2.imshow(
        "Pinky Centerline + YOLO",
        result
    )

    cv2.imshow(
        "White Mask",
        white_mask
    )

    cv2.imshow(
        "Black Mask",
        black_mask
    )

    cv2.imshow(
        "Road Mask Filled",
        road_mask
    )


    # =====================================================
    # 17. 키보드
    # =====================================================
    key = (
        cv2.waitKey(1)
        & 0xFF
    )


    # P = AUTO
    if key == ord("p"):
        auto_mode = (
            not auto_mode
        )

        stop()

        print(
            "AUTO ON"
            if auto_mode
            else "AUTO OFF"
        )


    # W
    elif key == ord("w"):
        auto_mode = False
        drive(
            30,
            30
        )


    # S
    elif key == ord("s"):
        auto_mode = False
        drive(
            -30,
            -30
        )


    # A
    elif key == ord("a"):
        auto_mode = False
        drive(
            -25,
            25
        )


    # D
    elif key == ord("d"):
        auto_mode = False
        drive(
            25,
            -25
        )


    # SPACE
    elif key == 32:
        auto_mode = False
        stop()
        print("STOP")


    # R
    elif key == ord("r"):
        auto_mode = False

        drive_state = "CENTERLINE"
        last_error = 0

        arrow_count = 0
        arrow_stable_count = 0
        arrow_missing_count = 0
        arrow_count_armed = True

        third_arrow_trigger_count = 0

        text_full_count = 0
        text_candidate_type = None

        state_start_time = 0.0

        yolo_ir_waiting = False
        yolo_ir_consumed = False

        stop()

        print("RESET")


    # ESC
    elif key == 27:
        stop()
        break


# =========================================================
# 종료
# =========================================================
stop()

cap.release()

cv2.destroyAllWindows()

print("종료")
