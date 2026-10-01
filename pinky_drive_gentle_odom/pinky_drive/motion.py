"""모터 명령 + 기본 주행 동작 (중심선 / STOP2 부드러운 주행 / GOAL 구간 / 글씨 추종, 펄스 회전)
+ 오도메트리 동작 (거리만큼 직진 / 각도만큼 회전) - 직진과 글자 정렬에서만 사용."""
import math
import time

import numpy as np

from . import config as C


def clamp(v, lo=-100, hi=100):
    return max(lo, min(hi, int(v)))


class Motion:
    def __init__(self, motor, state):
        self.motor = motor
        self.s = state

    # ---------------- 기본 명령 ----------------
    def drive(self, left, right):
        left, right = clamp(left), clamp(right)
        self.motor.move(left, right)
        if self.s.odom is not None:
            self.s.odom.set_command(left, right)   # 센서 없을 때 추정용

    def stop(self):
        self.drive(0, 0)

    def straight(self, speed):
        self.drive(speed, speed)

    def spin(self, speed):
        """제자리 회전 (양수 = 오른쪽)"""
        self.drive(speed, -speed)

    # ---------------- 오도메트리 동작 (직진 / 글자 정렬 전용) ----------------
    def straight_for(self, cm, speed):
        """이 상태에 들어온 뒤 cm 만큼 직진하면 정지하고 True.
        - 목표 근처에서 감속해서 덜 밀려남
        - 엔코더/IMU가 있으면 직진 중 방향이 틀어진 만큼 바로잡음
        - 시간은 안전장치로만 사용 (센서가 이상해도 영원히 가지 않게)"""
        s = self.s
        remain = cm - s.moved_cm()
        if remain <= 0 or s.elapsed() >= self._timeout(cm, speed):
            if remain > 0:
                print(f"[MOVE] timeout! moved {s.moved_cm():.1f}/{cm:.0f}cm")
            self.stop()
            return True

        v = speed if remain > C.SLOW_ZONE_CM else max(C.MIN_MOVE_SPEED, speed * C.SLOW_ZONE_RATIO)
        corr = C.HEADING_KP * s.turned_deg()        # 오른쪽으로 틀어졌으면(+) 왼쪽으로 보정
        self.drive(v - corr, v + corr)
        return False

    def turn_for(self, deg, speed):
        """이 상태에 들어온 뒤 deg 만큼 제자리 회전하면 정지하고 True (deg 양수 = 오른쪽)"""
        s = self.s
        remain = abs(deg) - abs(s.turned_deg())
        if remain <= 0 or s.elapsed() >= self._timeout(abs(deg), speed, turn=True):
            if remain > 0:
                print(f"[TURN] timeout! turned {s.turned_deg():.0f}/{deg:.0f}deg")
            self.stop()
            return True

        v = speed if remain > C.SLOW_ZONE_DEG else max(C.MIN_TURN_SPEED, speed * C.SLOW_ZONE_RATIO)
        self.spin(v if deg > 0 else -v)
        return False

    @staticmethod
    def _timeout(amount, speed, turn=False):
        """예상 소요 시간 x 여유배수 + 1초. 예상 시간은 추정 모드 환산값으로 계산"""
        cm_per_sec = max(1e-3, abs(speed) * C.SPEED_TO_CM_PER_SEC)
        if turn:
            expect = amount / math.degrees(2 * cm_per_sec / C.WHEEL_BASE_CM)
        else:
            expect = amount / cm_per_sec
        return expect * C.MOVE_TIMEOUT_FACTOR + 1.0

    # ---------------- 주행 동작 ----------------
    def follow_centerline(self, err):
        """중심선 추종(P제어 + 커브 감속), 선을 놓치면 마지막 오차 방향으로 탐색"""
        if err is not None:
            base = max(C.MIN_CURVE_SPEED, C.BASE_SPEED - C.CURVE_SLOWDOWN * abs(err))
            corr = C.KP * err
            self.drive(np.clip(base + corr, C.INNER_MIN_SPEED, C.MAX_SPEED),
                       np.clip(base - corr, C.INNER_MIN_SPEED, C.MAX_SPEED))
        elif self.s.last_error < 0:
            self.drive(C.SEARCH_INNER_SPEED, C.SEARCH_SPEED)
        elif self.s.last_error > 0:
            self.drive(C.SEARCH_SPEED, C.SEARCH_INNER_SPEED)
        else:
            self.stop()

    def follow_gentle(self, err, slow=False):
        """[GENTLE_STAGES = STOP2] 고개를 확 돌리지 않는 중심선 추종 (오도메트리 안 씀)

        - 조향 세기 GENTLE_KP (기본의 절반), 좌우 차이 최대 GENTLE_MAX_CORR -> 안쪽 바퀴 역회전 없음
        - 좌우 차이는 1초에 GENTLE_CORR_RATE 만큼만 바뀜 -> 중심선이 튀어도 갑자기 안 꺾임
        - YOLO에 STOP이 보이면(slow) GENTLE_TEXT_SPEED 로 감속
        - 중심선을 놓치면 GENTLE_LOST_STRAIGHT_SEC 동안 천천히 직진,
          그래도 없으면 바깥 바퀴만 굴려서 천천히 좌우로 살펴보기
          (마지막 오차 쪽 GENTLE_SWEEP_SEC -> 반대쪽 2배 -> 다시 2배 ... 한쪽으로 계속 돌지 않음)
        """
        s, now = self.s, time.time()
        dt = 0.0 if s.gentle_t is None else min(now - s.gentle_t, 0.5)
        s.gentle_t = now

        # ---- 중심선 놓침 ----
        if err is None:
            if s.gentle_lost_since is None:
                s.gentle_lost_since = now
            s.gentle_corr = 0.0                         # 다시 찾으면 직진부터 천천히 꺾음
            lost = now - s.gentle_lost_since
            if lost < C.GENTLE_LOST_STRAIGHT_SEC:
                self.straight(C.GENTLE_LOST_SPEED)
                return
            if s.last_error == 0:                       # 어느 쪽인지 모름 -> 정지 (예전과 같음)
                self.stop()
                return
            # 예전: 마지막 오차 쪽으로 (16, -6) 찾을 때까지 계속 제자리 회전
            k = lost - C.GENTLE_LOST_STRAIGHT_SEC
            sweep = 0 if k < C.GENTLE_SWEEP_SEC else 1 + int((k - C.GENTLE_SWEEP_SEC) // (2 * C.GENTLE_SWEEP_SEC))
            first = 1 if s.last_error > 0 else -1
            if (first if sweep % 2 == 0 else -first) > 0:
                self.drive(C.GENTLE_LOST_SPEED, 0)      # 오른쪽으로 천천히
            else:
                self.drive(0, C.GENTLE_LOST_SPEED)      # 왼쪽으로 천천히
            return
        s.gentle_lost_since = None

        # ---- 중심선 추종 ----
        speed = C.GENTLE_TEXT_SPEED if slow else C.GENTLE_SPEED
        lim = min(C.GENTLE_MAX_CORR, speed)             # 기본 속도 이하 -> 안쪽 바퀴 역회전 없음
        want = float(np.clip(C.GENTLE_KP * err, -lim, lim))
        step = C.GENTLE_CORR_RATE * dt                  # 이번 프레임에 바꿀 수 있는 최대량
        s.gentle_corr += float(np.clip(want - s.gentle_corr, -step, step))
        self.drive(speed + s.gentle_corr, speed - s.gentle_corr)

    def follow_exit(self, err):
        """GOAL 구간 주행: 오차를 부드럽게 + 선을 잠깐 놓치면 제자리 탐색 대신 천천히 직진"""
        s = self.s
        if err is not None:
            s.exit_lost_since = None
            if s.exit_err is None:
                s.exit_err = err
            else:
                s.exit_err = C.EXIT_ERR_SMOOTH * s.exit_err + (1 - C.EXIT_ERR_SMOOTH) * err
            self.follow_centerline(s.exit_err)
            return

        if s.exit_lost_since is None:
            s.exit_lost_since = time.time()
        if time.time() - s.exit_lost_since < C.EXIT_LOST_STRAIGHT_SEC:
            self.straight(C.EXIT_LOST_SPEED)
        else:
            s.exit_err = None
            self.follow_centerline(None)

    def follow_text(self, text, w):
        """YOLO 박스 중심을 화면 중앙에 맞추며 직진. 박스를 놓치면 그냥 직진"""
        if text is None:
            self.straight(C.TEXT_FOLLOW_SPEED)
            return
        corr = np.clip(C.TEXT_FOLLOW_KP * (text["center"][0] - w // 2),
                       -C.TEXT_FOLLOW_MAX_CORR, C.TEXT_FOLLOW_MAX_CORR)
        self.drive(C.TEXT_FOLLOW_SPEED + corr, C.TEXT_FOLLOW_SPEED - corr)

    def seek_pulse(self, direction):
        """조금 돌고 잠깐 멈추는 펄스 회전 (STATION 찾기 회전에서만 사용, 초 단위 그대로)"""
        if self.s.elapsed() % (C.SEEK_PULSE_ON + C.SEEK_PULSE_OFF) < C.SEEK_PULSE_ON:
            self.spin(direction * C.SEEK_TURN_SPEED)
        else:
            self.stop()
