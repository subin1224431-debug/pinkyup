"""디버그 화면: 카메라 영상 + 중심선/글씨 박스/상태 표시 | 흰 도로 마스크."""
import time

import cv2
import numpy as np

from . import config as C


YELLOW, GREEN, RED, BLUE = (0, 255, 255), (0, 255, 0), (0, 0, 255), (255, 0, 0)
ORANGE, GRAY, MAGENTA, CYAN = (0, 165, 255), (128, 128, 128), (255, 0, 255), (255, 255, 0)


def put(img, text, org, color, scale=0.52, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def draw_polyline(img, points, line_color, dot_color):
    for p, q in zip(points, points[1:]):
        cv2.line(img, p, q, line_color, 3)
    for p in points:
        cv2.circle(img, p, 4, dot_color, -1)


def render(frame, lane, text, rejected, s):
    h, w = frame.shape[:2]
    img = frame.copy()

    cv2.line(img, (0, lane.roi_start), (w, lane.roi_start), YELLOW, 2)
    cv2.line(img, (w // 2, lane.roi_start), (w // 2, h), GREEN, 2)

    draw_polyline(img, lane.points, BLUE, RED)
    if lane.target is not None:
        cv2.circle(img, lane.target, 9, YELLOW, -1)

    # STOP2 구간에서 실제로 따라가는 우측 오프셋 중심선 (주황색)
    if s.route_stage in C.RIGHT_OFFSET_STAGES:
        draw_polyline(img, lane.offset_points, ORANGE, ORANGE)
        if lane.offset_target is not None:
            cv2.circle(img, lane.offset_target, 9, ORANGE, 2)
        put(img, f"EVENT3 R-OFFSET: {C.CENTERLINE_RIGHT_OFFSET_RATIO:+.2f}", (12, 76), ORANGE)

    # 모양 때문에 버려진 박스(화살표 등) - 회색
    for x1, y1, x2, y2, name, aspect in rejected:
        cv2.rectangle(img, (x1, y1), (x2, y2), GRAY, 1)
        put(img, f"REJECT {name} {aspect:.1f}", (x1, max(20, y1 - 6)), GRAY, 0.45, 1)

    if text is not None:
        x1, y1, x2, y2 = text["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), MAGENTA, 2)
        put(img, f"{text['type']} {text['conf']:.2f}", (x1, max(20, y1 - 8)), MAGENTA, 0.60)
        cv2.circle(img, text["center"], 7, RED, -1)

    if s.route_stage in C.EXIT_STAGES:
        put(img, "EXIT: adaptive binarize + smooth", (12, 76), ORANGE)

    shown_state = s.drive_state if s.auto_mode else f"PAUSED / {s.drive_state}"
    put(img, f"STATE: {shown_state}", (12, 28), YELLOW, 0.60)
    put(img, f"ROUTE: {s.route_stage}", (12, 52), CYAN)

    # STATION 구간: 도로 폭 / 평소 폭 / 좁아짐 카운트 (조정할 때 보는 숫자)
    if s.route_stage == "STATION":
        base = s.narrow.base()
        rw = f"{s.narrow.road_w:.0f}" if s.narrow.road_w is not None else "-"
        bs = f"{base:.0f}" if base is not None else "-"
        put(img, f"ROAD W {rw} / BASE {bs} (x{C.NARROW_RATIO:.2f})  NARROW {s.narrow.hits}",
            (12, 100), ORANGE)

    if (s.auto_mode and s.drive_state == "CENTERLINE"
            and not s.initial_25s_done and s.auto_start_time is not None):
        remain = max(0.0, C.CENTERLINE_RUN_SEC - (time.time() - s.auto_start_time))
        put(img, f"CENTERLINE TIMER: {remain:.1f}s", (12, 55), YELLOW)

    # STOP2 부드러운 주행: 지금 좌우 속도 차이 / 최대 (조정할 때 보는 숫자)
    if s.route_stage in C.GENTLE_STAGES and s.drive_state == "CENTERLINE":
        put(img, f"GENTLE corr {s.gentle_corr:+.1f} / max {C.GENTLE_MAX_CORR}", (12, 100), ORANGE)

    # 글자 정렬: 정렬 시작 후 총 회전 각도 / 스텝 수
    if s.drive_state in ("TEXT_SEEK_FULL", "TEXT_SEEK_TURN") and s.odom is not None:
        put(img, f"ALIGN {s.odom.yaw - s.seek_start_yaw:+.0f}/{C.SEEK_MAX_DEG}deg  step {s.seek_steps}",
            (12, 124), ORANGE)

    # 오도메트리: 전체 누적 + 지금 상태에서 간 거리 / 돈 각도 (직진 거리, 정렬 각도 맞출 때 보는 숫자)
    if s.odom is not None:
        put(img, f"ODOM {s.odom.dist:.0f}cm {s.odom.yaw:+.0f}deg | "
                 f"STATE {s.moved_cm():.1f}cm {s.turned_deg():+.0f}deg",
            (12, h - 12), CYAN, 0.48)

    road_bgr = cv2.resize(cv2.cvtColor(lane.mask, cv2.COLOR_GRAY2BGR), (w, h),
                          interpolation=cv2.INTER_NEAREST)
    return np.hstack([img, road_bgr])
