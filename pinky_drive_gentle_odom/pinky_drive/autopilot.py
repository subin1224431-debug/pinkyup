"""자동주행 상태 머신. 상태 하나 = 메서드 하나.

CENTERLINE ─(25초)──────────────> SEARCH_TEXT ──(글자 전체 보임)──┐
    │                                                           │
    ├─(글씨 확인, 잘림)──> TEXT_SEEK_FULL ──(전체 보임)──────────> APPROACH_TEXT
    │                     (멈춰서 보기) <-> TEXT_SEEK_TURN (SEEK_STEP_DEG 만큼 회전)
    ├─(글씨 확인, 전체)─────────────────────────────────────────> APPROACH_TEXT
    ├─(STATION 구간 도로 좁아짐)─> STATION_NARROW_FORWARD
    │                              -> STATION_TURN_SEARCH ─(찾음)─> TEXT_SEEK_FULL
    └─(GOAL 확인)─> GOAL_APPROACH

APPROACH_TEXT -> DRIVE_TEXT -> FORWARD_AFTER_TEXT -> STOP_AFTER_TEXT
    -> STOP1/STATION 이면 다음 구간으로 CENTERLINE
    -> STOP2 이면 FINAL_FORWARD -> FINAL_RIGHT_TURN -> CENTERLINE(GOAL 구간)

GOAL_APPROACH -> GOAL_FINAL_FORWARD -> DONE

오도메트리(거리 cm / 각도 deg)는 "직진"과 "글자 정렬"에서만 사용:
  직진 = STATION_NARROW_FORWARD, FORWARD_AFTER_TEXT, FINAL_FORWARD, GOAL_FINAL_FORWARD  (m.straight_for)
  정렬 = TEXT_SEEK_FULL <-> TEXT_SEEK_TURN                                         (m.turn_for)
나머지(처음 25초, STATION 찾기 회전, 3초 정지, 마지막 우회전)는 예전처럼 초 단위.
"""
import time
from collections import namedtuple

from . import config as C
from .detector import fully_visible, seek_turn_dir


# 한 프레임에서 상태 머신이 보는 입력
Perception = namedtuple("Perception", [
    "text",           # 이번 구간 글씨 박스 (없으면 None)
    "w", "h",         # 화면 크기
    "error",          # 도로 중심선 오차
    "offset_error",   # 우측 오프셋 중심선 오차
    "confirmed",      # 중심선 주행 중 글씨 연속 확인 완료
])

# YOLO를 돌리는 주행 상태
#   TEXT_SEEK_TURN 은 일부러 뺌: 도는 중엔 YOLO 결과를 안 쓰고, 루프가 빨라야 각도에서 정확히 멈춤
YOLO_STATES = {
    "CENTERLINE", "SEARCH_TEXT", "TEXT_SEEK_FULL", "STATION_TURN_SEARCH",
    "APPROACH_TEXT", "DRIVE_TEXT", "GOAL_APPROACH",
}


class Autopilot:
    def __init__(self, state, motion):
        self.s = state
        self.m = motion
        self.handlers = {
            "CENTERLINE": self._centerline,
            "SEARCH_TEXT": self._search_text,
            "STATION_NARROW_FORWARD": self._station_narrow_forward,
            "STATION_TURN_SEARCH": self._station_turn_search,
            "TEXT_SEEK_FULL": self._text_seek_full,
            "TEXT_SEEK_TURN": self._text_seek_turn,
            "APPROACH_TEXT": self._approach_text,
            "DRIVE_TEXT": self._drive_text,
            "FORWARD_AFTER_TEXT": self._forward_after_text,
            "STOP_AFTER_TEXT": self._stop_after_text,
            "FINAL_FORWARD": self._final_forward,
            "FINAL_RIGHT_TURN": self._final_right_turn,
            "GOAL_APPROACH": self._goal_approach,
            "GOAL_FINAL_FORWARD": self._goal_final_forward,
            "DONE": self._done,
        }

    def step(self, p):
        self.handlers[self.s.drive_state](p)

    # ============================================================
    # 공통 전환
    # ============================================================
    def _back_to_centerline(self, msg=None):
        self.s.current_text_type = None
        self.s.enter("CENTERLINE", msg)

    def _start_text_seek(self, text, w):
        """STOP / STATION: 박스 전체가 화면에 들어올 때까지 회전 정렬"""
        s = self.s
        self.m.stop()
        s.seek_dir = seek_turn_dir(text, w)
        s.seek_start_yaw = s.odom.yaw if s.odom is not None else 0.0
        s.seek_steps = 0
        self._enter_seek_look(f"[YOLO] {s.current_text_type} detected -> TEXT_SEEK_FULL "
                              f"({'right' if s.seek_dir > 0 else 'left'} cut)")

    def _enter_seek_look(self, msg=None):
        """정렬: 멈춰서 보기 (TEXT_SEEK_FULL) 시작"""
        s = self.s
        s.text_full_count = 0
        s.seek_frames = 0
        s.seek_misses = 0
        s.enter("TEXT_SEEK_FULL", msg)

    def _start_goal_approach(self, msg):
        self.s.goal_seen_near = False
        self.s.goal_last_seen = time.time()
        self.s.enter("GOAL_APPROACH", msg)

    # ============================================================
    # CENTERLINE: 기본 주행 + 이벤트 감시
    # ============================================================
    def _centerline(self, p):
        s, m = self.s, self.m
        if s.auto_start_time is None:
            s.auto_start_time = time.time()

        # [이벤트 1] 시작 후 25초 -> 정지 후 제자리 회전하며 글씨 탐색
        if not s.initial_25s_done and time.time() - s.auto_start_time >= C.CENTERLINE_RUN_SEC:
            m.stop()
            s.initial_25s_done = True
            s.text_full_count = 0
            s.text_candidate_type = None
            s.enter("SEARCH_TEXT", f"[TIMER] first {C.CENTERLINE_RUN_SEC:g}s centerline done -> SEARCH_TEXT")

        # [이벤트 2] 글씨가 연속 프레임으로 확인되면 글씨 처리 시작
        elif p.text is not None and p.confirmed:
            self._on_text_confirmed(p)

        # [STATION 앞] 흰 도로가 갑자기 좁아지고 중심선이 옆으로 쏠림
        elif s.narrow.trigger:
            s.narrow.hits = 0
            s.current_text_type = "STATION"
            s.enter("STATION_NARROW_FORWARD",
                    f"[STATION] road narrowed (W {s.narrow.road_w:.0f} / BASE {s.narrow.base():.0f})"
                    f" -> forward {C.NARROW_FORWARD_CM:.0f}cm more")

        # 주행
        else:
            # [이벤트 3] STOP2 구간은 우측 오프셋 중심선을 따라감
            err = p.offset_error if s.route_stage in C.RIGHT_OFFSET_STAGES else p.error

            if s.route_stage in C.GENTLE_STAGES:
                # ★ STOP2 인식하러 갈 때: 고개를 확 돌리지 않게 부드럽게 (오도메트리 안 씀)
                #   STOP이 보이기 시작하면 감속해서 확인 프레임을 더 확보
                m.follow_gentle(err, slow=(p.text is not None))
            elif s.route_stage in C.EXIT_STAGES:
                m.follow_exit(err)          # GOAL 구간: 평활화
            else:
                m.follow_centerline(err)    # 기본

    def _on_text_confirmed(self, p):
        s = self.s
        s.current_text_type = p.text["type"]
        full_hits = s.confirm.full_hits
        s.confirm.reset()

        if s.current_text_type == "GOAL":
            self._start_goal_approach("[YOLO] GOAL detected -> GOAL_APPROACH")
        elif fully_visible(p.text, p.w) and full_hits >= C.FAST_FULL_FRAMES:
            # 이미 글자 전체가 보였음 -> 멈춰서 다시 확인하지 않고 바로 접근
            s.enter("APPROACH_TEXT", f"[YOLO] {s.current_text_type} already full -> APPROACH_TEXT (fast)")
        else:
            self._start_text_seek(p.text, p.w)

    # ============================================================
    # SEARCH_TEXT: 25초 직후 제자리 회전하며 글자 전체가 보일 때까지 탐색
    # ============================================================
    def _search_text(self, p):
        s, m, t = self.s, self.m, p.text
        full_target = None

        if t is not None and fully_visible(t, p.w):
            if s.text_candidate_type == t["type"]:
                s.text_full_count += 1
            else:
                s.text_candidate_type = t["type"]
                s.text_full_count = 1
            if s.text_full_count >= C.FULL_STABLE_FRAMES:
                full_target = t
        else:
            s.text_full_count = 0
            s.text_candidate_type = None

        if full_target is None:
            m.spin(C.TURN_SPEED)
            return

        m.stop()
        s.current_text_type = full_target["type"]
        if s.current_text_type == "GOAL":
            self._start_goal_approach("[YOLO] GOAL full -> GOAL_APPROACH")
        else:
            s.enter("APPROACH_TEXT", f"[YOLO] {s.current_text_type} full -> APPROACH_TEXT")

    # ============================================================
    # STATION 앞 도로 좁아짐: 조금 더 직진 -> 멈춤 -> 오른쪽으로 돌며 찾기
    # ============================================================
    def _station_narrow_forward(self, p):
        # ★ 오도메트리 직진: NARROW_FORWARD_CM 만큼 (예전: 0.85초)
        if self.m.straight_for(C.NARROW_FORWARD_CM, C.NARROW_FORWARD_SPEED):
            self.s.enter("STATION_TURN_SEARCH", "[STATION] forward done -> stop, turn right to find STATION")

    def _station_turn_search(self, p):
        s = self.s
        if p.text is not None:
            # 찾음 -> 기존 정렬 과정 그대로 (잘렸으면 잘린 쪽으로 정렬, 전체면 접근)
            s.current_text_type = "STATION"
            self._start_text_seek(p.text, p.w)
        elif s.elapsed() >= C.STATION_TURN_SEARCH_MAX_SEC:
            self.m.stop()
            s.narrow.cooldown_until = time.time() + C.NARROW_COOLDOWN_SEC
            self._back_to_centerline(
                f"[STATION] not found in {C.STATION_TURN_SEARCH_MAX_SEC:.1f}s -> back to CENTERLINE")
        else:
            self.m.seek_pulse(1)   # 오른쪽

    # ============================================================
    # ★ 정렬 (오도메트리): TEXT_SEEK_FULL(멈춰서 보기) <-> TEXT_SEEK_TURN(SEEK_STEP_DEG 만큼만 회전)
    #   예전: "0.12초 돌고 0.15초 멈춤" 시간 펄스
    #         -> YOLO 때문에 루프 한 바퀴가 0.12초보다 길면 명령이 다음 프레임까지 유지돼서 한 번에 20도 넘게 돎
    #   지금: 각도로 끊어서 "8도 돌고 -> 멈춰서 보고" 반복
    # ============================================================
    def _text_seek_full(self, p):
        """멈춰서 YOLO 결과를 보고 다음 행동 결정"""
        s, t = self.s, p.text
        self.m.stop()
        s.seek_frames += 1
        if s.seek_frames <= C.SEEK_SETTLE_FRAMES:      # 멈춘 직후 프레임은 흔들림 -> 판단 안 함
            return

        total = s.odom.yaw - s.seek_start_yaw          # 정렬 시작 후 총 회전 각도

        # 안 보임: 몇 프레임 더 기다려 보고, 그래도 없으면 중심선 추종으로 복귀 (돌면서 찾지 않음)
        if t is None:
            s.text_full_count = 0
            s.seek_misses += 1
            if s.seek_misses >= C.SEEK_MISS_FRAMES:
                self._back_to_centerline(f"[ALIGN] {s.current_text_type} lost ({total:+.0f}deg) -> back to CENTERLINE")
            return
        s.seek_misses = 0

        # 전체 보임이 연속 FULL_STABLE_FRAMES 프레임 -> 접근
        if fully_visible(t, p.w):
            s.text_full_count += 1
            if s.text_full_count >= C.FULL_STABLE_FRAMES:
                s.text_full_count = 0
                s.enter("APPROACH_TEXT",
                        f"[ALIGN] {s.current_text_type} all letters visible ({total:+.0f}deg) -> APPROACH_TEXT")
            return
        s.text_full_count = 0

        # 잘려 있음 -> 잘린 쪽으로 한 스텝 더
        new_dir = seek_turn_dir(t, p.w)
        flipped = s.seek_steps > 0 and new_dir != s.seek_dir                 # 왔다 갔다 = 글씨가 화면보다 큼
        too_far = abs(total + new_dir * C.SEEK_STEP_DEG) > C.SEEK_MAX_DEG     # 더 돌면 너무 많이 돎
        if flipped or too_far:
            why = "left-right flip" if flipped else f"max {C.SEEK_MAX_DEG}deg"
            s.enter("APPROACH_TEXT", f"[ALIGN] {why} ({total:+.0f}deg) -> APPROACH_TEXT as it is")
            return

        s.seek_dir = new_dir
        s.seek_steps += 1
        s.enter("TEXT_SEEK_TURN")

    def _text_seek_turn(self, p):
        """SEEK_STEP_DEG 만큼만 돌고 멈춤 -> 다시 멈춰서 보기"""
        if self.m.turn_for(self.s.seek_dir * C.SEEK_STEP_DEG, C.SEEK_STEP_SPEED):
            self._enter_seek_look()

    # ============================================================
    # STOP / STATION 글씨 처리: 접근 -> 바닥까지 -> 직진 -> 정지 -> 다음 구간
    # ============================================================
    def _approach_text(self, p):
        """박스 중심 맞추며 접근, 박스 하단이 화면 60%까지 오면 DRIVE_TEXT"""
        self.m.follow_text(p.text, p.w)
        if p.text is not None and p.text["bbox"][3] >= int(p.h * C.TEXT_ALIGN_TRIGGER_RATIO):
            self.s.enter("DRIVE_TEXT",
                         f"[YOLO] {p.text['type']} reached {C.TEXT_ALIGN_TRIGGER_RATIO:.0%} -> DRIVE_TEXT")

    def _drive_text(self, p):
        """박스 하단이 화면 바닥에 닿을 때까지 따라감"""
        s = self.s
        if p.text is not None and p.text["bbox"][3] >= int(p.h * C.TEXT_BOTTOM_TRIGGER_RATIO):
            self.m.stop()
            s.enter("FORWARD_AFTER_TEXT",
                    f"[YOLO] {s.current_text_type} bbox bottom reached -> forward {C.FORWARD_AFTER_TEXT_CM:.0f}cm")
        else:
            self.m.follow_text(p.text, p.w)

    def _forward_after_text(self, p):
        # ★ 오도메트리 직진: FORWARD_AFTER_TEXT_CM 만큼 (예전: 3초)
        if self.m.straight_for(C.FORWARD_AFTER_TEXT_CM, C.TEXT_FOLLOW_SPEED):
            self.s.enter("STOP_AFTER_TEXT",
                         f"[TEXT] forward {self.s.moved_cm():.0f}cm done -> stop {C.STOP_AFTER_TEXT_SEC:.1f}s")

    def _stop_after_text(self, p):
        s = self.s
        self.m.stop()
        if s.elapsed() < C.STOP_AFTER_TEXT_SEC:
            return

        prev_stage = s.route_stage
        if prev_stage in C.NEXT_STAGE:
            s.route_stage = C.NEXT_STAGE[prev_stage]
            self._back_to_centerline(f"[ROUTE] {prev_stage} done -> CENTERLINE / SEARCH {s.route_stage}")
        elif prev_stage == "STOP2":
            s.current_text_type = None
            s.enter("FINAL_FORWARD", f"[ROUTE] STOP2 done -> FINAL_FORWARD {C.FINAL_STOP_FORWARD_CM:.0f}cm")
        else:
            self._back_to_centerline()

    # ============================================================
    # STOP2 이후: 직진 -> 우회전 -> GOAL 구간 중심선
    # ============================================================
    def _final_forward(self, p):
        # ★ 오도메트리 직진: FINAL_STOP_FORWARD_CM 만큼 (예전: 3초)
        if self.m.straight_for(C.FINAL_STOP_FORWARD_CM, C.TEXT_FOLLOW_SPEED):
            self.s.enter("FINAL_RIGHT_TURN",
                         f"[ROUTE] final forward {self.s.moved_cm():.0f}cm done"
                         f" -> right turn {C.FINAL_TURN_DEG}deg ({C.FINAL_RIGHT_TURN_SEC:.2f}s)")

    def _final_right_turn(self, p):
        s = self.s
        if s.elapsed() < C.FINAL_RIGHT_TURN_SEC:
            self.m.spin(C.FINAL_TURN_SPEED)
            return

        self.m.stop()
        s.route_stage = "GOAL"
        s.exit_err = None          # 회전 전 오차 기억 지우고 새로 중심선 시작
        s.exit_lost_since = None
        s.enter("CENTERLINE", f"[ROUTE] final right turn {C.FINAL_TURN_DEG}deg done -> CENTERLINE / SEARCH GOAL")

    # ============================================================
    # GOAL: 박스 따라 직진 (안 보여도 멈추지 않음) -> 사라지면 더 직진 -> 완전 정지
    # ============================================================
    def _goal_approach(self, p):
        s = self.s
        self.m.follow_text(p.text, p.w)

        if p.text is not None:
            s.goal_last_seen = time.time()
            if p.text["bbox"][3] >= int(p.h * C.GOAL_NEAR_RATIO):
                s.goal_seen_near = True
            return

        missing = time.time() - s.goal_last_seen
        if s.goal_seen_near and missing >= C.GOAL_MISS_TIMEOUT:
            s.enter("GOAL_FINAL_FORWARD", f"[GOAL] box gone -> forward {C.GOAL_FINAL_FORWARD_CM:.0f}cm more")
        elif missing >= C.GOAL_GIVEUP_SEC:
            s.enter("GOAL_FINAL_FORWARD",
                    f"[GOAL] lost {C.GOAL_GIVEUP_SEC:.1f}s -> forward {C.GOAL_FINAL_FORWARD_CM:.0f}cm more")

    def _goal_final_forward(self, p):
        s = self.s
        # ★ 오도메트리 직진: GOAL_FINAL_FORWARD_CM 만큼 (예전: 7초)
        if not self.m.straight_for(C.GOAL_FINAL_FORWARD_CM, C.TEXT_FOLLOW_SPEED):
            return

        s.route_stage = "DONE"
        s.auto_mode = False
        s.enter("DONE", f"[GOAL] final forward {s.moved_cm():.0f}cm done -> ROBOT STOP / DONE")

    def _done(self, p):
        self.m.stop()
