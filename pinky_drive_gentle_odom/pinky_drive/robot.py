"""메인 제어 루프 + AUTO / RESET / 수동 조작 명령."""
import threading
import time

import cv2

from . import config as C
from .autopilot import YOLO_STATES, Autopilot, Perception
from .lane import analyze_lane
from .motion import Motion
from .overlay import render
from .state import State


MANUAL_CMDS = {
    "w": (C.MANUAL_SPEED, C.MANUAL_SPEED),
    "s": (-C.MANUAL_SPEED, -C.MANUAL_SPEED),
    "a": (-C.MANUAL_SPEED, C.MANUAL_SPEED),
    "d": (C.MANUAL_SPEED, -C.MANUAL_SPEED),
}


class Robot:
    def __init__(self, camera, motor, detector, odom):
        self.camera = camera
        self.detector = detector
        self.odom = odom                  # 직진 / 글자 정렬 에서만 사용
        self.s = State(odom)
        self.motion = Motion(motor, self.s)
        self.autopilot = Autopilot(self.s, self.motion)

        self.stop_event = threading.Event()
        self._jpeg = None
        self._jpeg_lock = threading.Lock()

    # ============================================================
    # 제어 루프
    # ============================================================
    def run(self):
        while not self.stop_event.is_set():
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            combo = self.step(frame.copy())
            ok, jpg = cv2.imencode(".jpg", combo, [int(cv2.IMWRITE_JPEG_QUALITY), C.JPEG_QUALITY])
            if ok:
                with self._jpeg_lock:
                    self._jpeg = jpg.tobytes()

            time.sleep(0.03)

    def step(self, frame):
        """한 프레임 처리: 인식 -> 주행 -> 화면 이미지 반환"""
        s = self.s
        self.odom.update()          # 지난 프레임 이후 이동 거리 / 회전 각도 반영
        h, w = frame.shape[:2]
        roi_start = int(h * C.ROI_START_RATIO)
        roi = frame[roi_start:h, :]

        # ---------------- 중심선 ----------------
        lane = analyze_lane(roi, roi_start, s.route_stage)
        if lane.error is not None:
            s.last_error = lane.error

        s.narrow.update(lane.half_widths, lane.error, active=(
            s.auto_mode and s.initial_25s_done
            and s.route_stage == "STATION" and s.drive_state == "CENTERLINE"))

        # ---------------- YOLO (25초 이후) ----------------
        text, rejected = None, []
        if s.initial_25s_done and s.drive_state in YOLO_STATES:
            try:
                text, rejected = self.detector.detect(roi, s.route_stage, y_offset=roi_start)
            except Exception as e:
                print("YOLO ERROR:", e)

        confirmed = s.confirm.update(text, w,
                                     in_centerline=(s.drive_state == "CENTERLINE"),
                                     strict=(s.route_stage in C.EVENT2_STRICT_STAGES))

        # ---------------- 주행 ----------------
        if time.time() < s.manual_until and s.manual_cmd is not None:
            self.motion.drive(*s.manual_cmd)
        elif not s.auto_mode:
            self.motion.stop()
        else:
            self.autopilot.step(Perception(text, w, h, lane.error, lane.offset_error, confirmed))

        # ---------------- 화면 ----------------
        return render(frame, lane, text, rejected, s)

    def latest_jpeg(self):
        with self._jpeg_lock:
            return self._jpeg

    # ============================================================
    # 명령 (웹 버튼 / 키보드 / ZMQ)
    # ============================================================
    def start_auto(self, source):
        """AUTO 시작 공통 (노트북 ZMQ 신호 / p 키)"""
        s = self.s
        if s.auto_mode:
            return
        s.auto_mode = True
        s.enter("CENTERLINE")       # 상태 시작 시각 / 오도메트리 시작 위치도 같이 기록
        if not s.initial_25s_done:
            s.auto_start_time = time.time()
            print(f"[{source}] AUTO ON -> FIRST {C.CENTERLINE_RUN_SEC:g}s CENTERLINE")
        else:
            print(f"[{source}] AUTO ON -> NORMAL CENTERLINE")

    def handle_key(self, key):
        s = self.s
        if key == "p":
            if s.auto_mode:
                s.auto_mode = False
                self.motion.stop()
                print("AUTO OFF")
            else:
                self.start_auto("KEY")

        elif key == "r":
            s.reset()
            self.motion.stop()
            print("RESET")

        elif key == "space":
            s.auto_mode = False
            self.motion.stop()

        elif key in MANUAL_CMDS:
            s.manual_cmd = MANUAL_CMDS[key]
            s.manual_until = time.time() + C.MANUAL_PULSE
