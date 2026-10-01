"""주행 거리(cm) / 회전 각도(deg) 측정.

시간 대신 "얼마나 갔는지 / 얼마나 돌았는지"로 동작을 끝내기 위한 모듈.
센서에 따라 세 가지 방식으로 동작한다 (main.py 에서 함수를 넘겨주면 자동 선택):

  1) 엔코더 있음  (read_ticks 넘김)   -> 바퀴 틱으로 거리 + 각도 계산   ★ 추천
  2) IMU 있음     (read_yaw_deg 넘김) -> 각도는 IMU로 (미끄러져도 정확)  ★★ 회전에 최고
  3) 둘 다 없음                        -> 모터 명령값 x 시간으로 "추정"
                                          (임시용. 배터리/바닥에 따라 틀어짐 = 예전 시간 방식과 비슷)

부호: 거리는 앞으로 +, 각도는 오른쪽(시계방향) 회전이 +
"""
import math
import time

from . import config as C


class Odometry:
    def __init__(self, read_ticks=None, read_yaw_deg=None):
        """
        read_ticks   : () -> (왼쪽 누적 틱, 오른쪽 누적 틱)   엔코더 읽는 함수
        read_yaw_deg : () -> 현재 yaw 각도(도)               IMU 읽는 함수
        """
        self.read_ticks = read_ticks
        self.read_yaw_deg = read_yaw_deg

        self.dist = 0.0      # 누적 전진 거리 (cm)
        self.yaw = 0.0       # 누적 회전 각도 (deg, 오른쪽 +)

        self._cmd = (0, 0)   # 마지막 모터 명령 (추정 모드용)
        self._prev_t = None
        self._prev_ticks = None
        self._prev_yaw_raw = None

        mode = []
        mode.append("encoder" if read_ticks else "estimate(speed x time)")
        mode.append("imu-yaw" if read_yaw_deg else "wheel-yaw")
        print("[ODOM] mode:", ", ".join(mode))

    # 모터에 명령을 보낼 때마다 Motion 이 불러줌 (추정 모드용)
    def set_command(self, left, right):
        self._cmd = (left, right)

    def update(self):
        """제어 루프 매 프레임 맨 앞에서 호출"""
        now = time.time()
        dt = 0.0 if self._prev_t is None else now - self._prev_t
        self._prev_t = now

        # ---------------- 바퀴 이동량 (cm) ----------------
        if self.read_ticks is not None:
            l, r = self.read_ticks()
            if self._prev_ticks is None:
                dl = dr = 0.0
            else:
                dl = (l - self._prev_ticks[0]) * C.CM_PER_TICK
                dr = (r - self._prev_ticks[1]) * C.CM_PER_TICK
            self._prev_ticks = (l, r)
        else:
            # 추정: 속도 명령값 x 시간
            dl = self._cmd[0] * C.SPEED_TO_CM_PER_SEC * dt
            dr = self._cmd[1] * C.SPEED_TO_CM_PER_SEC * dt

        self.dist += (dl + dr) / 2.0          # 제자리 회전은 거리 0

        # ---------------- 회전 각도 (deg) ----------------
        if self.read_yaw_deg is not None:
            raw = self.read_yaw_deg() * C.IMU_YAW_SIGN
            if self._prev_yaw_raw is not None:
                d = (raw - self._prev_yaw_raw + 180.0) % 360.0 - 180.0   # -180/180 넘어가는 것 처리
                self.yaw += d
            self._prev_yaw_raw = raw
        else:
            self.yaw += math.degrees((dl - dr) / C.WHEEL_BASE_CM)

    def snapshot(self):
        return (self.dist, self.yaw)
