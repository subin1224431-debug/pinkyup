"""로봇 전체 상태 (주행 상태 머신 + 코스 진행 + 감지기 기억값)."""
import time

from .detector import TextConfirm
from .lane import RoadNarrowDetector


class State:
    def __init__(self, odom=None):
        self.manual_until = 0.0
        self.manual_cmd = None
        self.odom = odom                  # odometry.Odometry (직진 / 정렬 에서만 사용, reset 해도 유지)
        self.reset()

    def reset(self):
        self.auto_mode = False
        self.drive_state = "CENTERLINE"   # autopilot.py 의 상태 이름
        self.route_stage = "STOP1"        # STOP1 -> STATION -> STOP2 -> GOAL -> DONE
        self.state_start_time = 0.0
        self.last_error = 0

        self.auto_start_time = None
        self.initial_25s_done = False

        self.current_text_type = None
        self.text_candidate_type = None
        self.text_full_count = 0

        self.seek_dir = 1                 # 1 = 오른쪽 회전, -1 = 왼쪽 회전
        self.mark = (0.0, 0.0)            # 지금 상태에 들어올 때의 오도메트리 (거리 cm, 각도 deg)

        # 정렬 (TEXT_SEEK_FULL <-> TEXT_SEEK_TURN)
        self.seek_start_yaw = 0.0         # 정렬 시작할 때 방향
        self.seek_steps = 0               # 지금까지 돈 스텝 수
        self.seek_frames = 0              # 멈춰서 본 프레임 수
        self.seek_misses = 0              # 글씨가 연속으로 안 보인 프레임 수

        # STOP2 부드러운 주행 (motion.follow_gentle, 오도메트리 안 씀)
        self.gentle_corr = 0.0            # 지금 좌우 속도 차이
        self.gentle_t = None              # 마지막 계산 시각
        self.gentle_lost_since = None     # 중심선을 놓친 시각

        self.goal_seen_near = False
        self.goal_last_seen = None

        self.exit_err = None              # GOAL 구간 평활화된 오차
        self.exit_lost_since = None

        self.confirm = TextConfirm()          # 중심선 주행 중 글씨 연속 확인
        self.narrow = RoadNarrowDetector()    # STATION 앞 도로 좁아짐 감지

    def enter(self, drive_state, msg=None):
        self.drive_state = drive_state
        self.state_start_time = time.time()
        if self.odom is not None:
            self.mark = self.odom.snapshot()   # 이 상태 시작 위치 기억 (직진 거리 / 정렬 각도 계산용)
        self.gentle_corr = 0.0                 # 부드러운 주행은 상태가 바뀔 때마다 직진에서 새로 시작
        self.gentle_t = None
        self.gentle_lost_since = None
        if msg:
            print(msg)

    def elapsed(self):
        return time.time() - self.state_start_time

    # elapsed() 와 같은 방식: "이 상태 들어온 뒤 얼마나 갔나 / 돌았나"
    def moved_cm(self):
        return 0.0 if self.odom is None else self.odom.dist - self.mark[0]

    def turned_deg(self):
        """오른쪽 회전 +, 왼쪽 회전 -"""
        return 0.0 if self.odom is None else self.odom.yaw - self.mark[1]
