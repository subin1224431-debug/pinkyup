"""흰 도로 마스크 -> 중심선 검출, STATION 앞 도로 좁아짐 감지."""
import time
from collections import deque, namedtuple

import cv2
import numpy as np

from . import config as C


KERNEL_OPEN = np.ones((3, 3), np.uint8)
KERNEL_CLOSE = np.ones((7, 7), np.uint8)
KERNEL_ARROW = np.ones((C.ARROW_CLOSE_KERNEL, C.ARROW_CLOSE_KERNEL), np.uint8)
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# 한 프레임의 중심선 검출 결과
Lane = namedtuple("Lane", [
    "roi_start", "mask",
    "points", "half_widths", "target", "error",           # 도로 중심선
    "offset_points", "offset_target", "offset_error",     # 우측 오프셋 중심선
])


# ============================================================
# 마스크
# ============================================================
def row_groups(row):
    """한 행에서 연속된 흰 픽셀 묶음 중 MIN_WIDTH 이상인 것만 반환"""
    xs = np.where(row == 255)[0]
    if len(xs) == 0:
        return []
    groups = np.split(xs, np.where(np.diff(xs) > 1)[0] + 1)
    return [g for g in groups if len(g) >= C.MIN_WIDTH]


def remove_small_components(mask, min_area):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def fill_road_internal_gaps(mask, gap_ratio=C.INTERNAL_GAP_RATIO):
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
    return fill_road_internal_gaps(filled, gap_ratio=C.ARROW_GAP_RATIO)


def exit_adaptive_white(hsv):
    """GOAL 구간 전용 자동 이진화: 밝기 대비 보정(CLAHE) + Otsu 자동 임계값
    고정 HSV(V>=175)가 조명 때문에 흰 선을 놓칠 때 보충해 줌"""
    v = CLAHE.apply(hsv[:, :, 2])
    otsu_t, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(C.EXIT_V_MIN, int(otsu_t))
    return (((v >= t) & (hsv[:, :, 1] <= C.EXIT_S_MAX)) * 255).astype(np.uint8)


def build_road_mask(roi, route_stage):
    hsv = cv2.cvtColor(cv2.GaussianBlur(roi, (5, 5), 0), cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, C.LOWER_WHITE, C.UPPER_WHITE)

    # GOAL 구간: 고정 HSV + 자동 이진화 합치기
    if route_stage in C.EXIT_STAGES:
        white = cv2.bitwise_or(white, exit_adaptive_white(hsv))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, KERNEL_OPEN)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, KERNEL_CLOSE)
    white = remove_small_components(white, C.MIN_AREA)

    if route_stage in C.ARROW_FILL_STAGES:
        return fill_arrow_as_road(white)
    return fill_road_internal_gaps(white)


# ============================================================
# 중심선
# ============================================================
def find_center_points(road_mask, roi_start):
    """ROI 아래에서 위로 올라가며 도로 중심점과 도로 반폭을 찾음"""
    roi_h, roi_w = road_mask.shape
    points, half_widths = [], []

    for y in range(roi_h - 1, 0, -C.STEP):
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

        if points and abs(x_center - points[-1][0]) > C.MAX_JUMP:
            continue

        points.append((x_center, y + roi_start))
        half_widths.append((x_right - x_left) / 2.0)

        if len(points) >= C.MAX_POINTS:
            break

    return points, half_widths


def pick_target(points, w):
    """3번째 점을 목표점으로 사용 -> (목표점, 화면 중앙 대비 오차)"""
    if len(points) < 3:
        return None, None
    return points[2], points[2][0] - w // 2


def analyze_lane(roi, roi_start, route_stage):
    w = roi.shape[1]
    mask = build_road_mask(roi, route_stage)
    points, half_widths = find_center_points(mask, roi_start)
    target, error = pick_target(points, w)

    # 우측 오프셋 중심선: 오프셋 = RATIO x 도로 반폭
    offset_points = [
        (int(cx + C.CENTERLINE_RIGHT_OFFSET_RATIO * hw), cy)
        for (cx, cy), hw in zip(points, half_widths)
    ]
    offset_target, offset_error = pick_target(offset_points, w)

    return Lane(roi_start, mask, points, half_widths, target, error,
                offset_points, offset_target, offset_error)


# ============================================================
# STATION 앞 도로 좁아짐 감지
# ============================================================
class RoadNarrowDetector:
    """흰 도로 폭이 평소(BASE)보다 갑자기 좁아지고 중심선이 옆으로 쏠리면 trigger"""

    def __init__(self):
        self.road_w = None
        self.hist = deque(maxlen=C.NARROW_BASE_FRAMES)
        self.hits = 0
        self.trigger = False
        self.leg_start = None
        self.cooldown_until = 0.0

    def base(self):
        return float(np.median(self.hist)) if len(self.hist) >= 3 else None

    def update(self, half_widths, error, active):
        """active: STATION 구간에서 중심선 주행 중일 때만 True"""
        self.trigger = False
        self.road_w = 2 * float(np.mean(half_widths[:3])) if len(half_widths) >= 3 else None

        if not active:
            self.leg_start = None
            self.hist.clear()
            self.hits = 0
            return

        now = time.time()
        if self.leg_start is None:
            self.leg_start = now
        if self.road_w is None:
            return

        base = self.base()
        narrow = base is not None and self.road_w < base * C.NARROW_RATIO
        side = error is not None and abs(error) >= C.NARROW_SIDE_ERR
        armed = (now - self.leg_start >= C.NARROW_ARM_SEC
                 and now >= self.cooldown_until)

        if narrow and side and armed:
            self.hits += 1
        else:
            self.hits = 0

        if not narrow:
            self.hist.append(self.road_w)   # 평소 폭은 좁아지지 않은 프레임으로만 학습

        self.trigger = self.hits >= C.NARROW_CONFIRM_FRAMES
