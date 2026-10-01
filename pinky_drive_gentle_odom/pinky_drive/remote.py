"""ZMQ: 노트북에서 START_AUTONAV 받으면 AUTO 시작.
(신호를 기다리는 동안에도 p 키로 AUTO를 켤 수 있음)"""
import json

import zmq

from . import config as C


def wait_for_start(robot):
    try:
        socket = zmq.Context().socket(zmq.REQ)
        socket.connect(f"tcp://{C.LAPTOP_ZMQ_IP}:{C.LAPTOP_ZMQ_PORT}")

        print("[핑키봇] 노트북 ZMQ 서버에 접속합니다...")
        socket.send_string("핑키봇 준비 완료!")

        if json.loads(socket.recv_string()).get("status") == "START_AUTONAV":
            print("[핑키봇] START_AUTONAV 수신")
            robot.start_auto("ZMQ")
    except Exception as e:
        print("[ZMQ] 대기 스레드 오류:", e)
