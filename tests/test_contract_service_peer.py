import importlib.util
from pathlib import Path
import socket
import sys
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from contract_service_peer import (ServicePeer, EnginePeer, Loan, encode, decode, OFFER, ACCEPT,
    RESULT, RECEIPT, HELLO, SUBMIT, ADMITTED, ROUTE, DECISION, FRAME, encode_frame, decode_frame,
    encode_hello, decode_hello, encode_submit, encode_admitted, encode_route, encode_decision,
    connect_loopback)

class ServicePeerTests(unittest.TestCase):
    def test_actual_motion_service_and_retained_receipt(self):
        source = Path(__file__).resolve().parents[2] / 'turing/examples/meowdoku/entity_motion.py'
        spec = importlib.util.spec_from_file_location('qualified_motion', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        a, b = socket.socketpair()
        b.setblocking(False)
        calls, receipts, outputs = [], [], {}
        loan = Loan(11, 2, 9, 17, 1, 4, 100)
        inputs = (0.3, 0.0, 1.0, -1.0, 0.0, 1.0, 0.8, 0.1, 0.4, 0.016)
        def dispatch(value):
            self.assertEqual(value, loan)
            calls.append(value)
            outputs[23] = module.damped_axis(*inputs)
            return 23
        with ThreadPoolExecutor(max_workers=1) as executor:
            peer = ServicePeer(a, executor, dispatch, lambda l, ok: receipts.append((l, ok)), announce=False)
            # Fragmented native frame, exact wire ABI.
            frame = encode(OFFER, loan)
            b.sendall(frame[:13]); peer.poll()
            self.assertEqual(calls, [])
            b.sendall(frame[13:])
            received = bytearray()
            for _ in range(200):
                peer.poll()
                try: received.extend(b.recv(144))
                except BlockingIOError: pass
                if len(received) == 144: break
                time.sleep(.001)
            self.assertIsNone(peer.error)
            self.assertEqual([decode(received[i:i+72])[0] for i in (0,72)], [ACCEPT, RESULT])
            self.assertEqual(decode(received[72:])[2], 23)
            self.assertEqual(outputs[23], module.damped_axis(*inputs))
            self.assertEqual(peer.loan, loan)  # Result retained until receipt.
            b.sendall(encode(RECEIPT, loan, 1)); peer.poll()
            self.assertEqual(receipts, [(loan, True)])
            b.sendall(frame); peer.poll(); peer.poll()
            self.assertEqual(len(calls), 1)  # Never replay a finished generation.
            peer.close()
        b.close()

    def test_uncertain_service_failure_stops_without_replay(self):
        a,b=socket.socketpair()
        calls=[]
        def fail(loan):
            calls.append(loan)
            raise RuntimeError('mutation outcome uncertain')
        with ThreadPoolExecutor(max_workers=1) as executor:
            peer=ServicePeer(a,executor,fail,lambda *_:None,announce=False)
            b.sendall(encode(OFFER,Loan(1,1,1,1,1,1,100)))
            for _ in range(100):
                peer.poll()
                if peer.error: break
                time.sleep(.001)
            self.assertIsInstance(peer.error,RuntimeError)
            for _ in range(5): peer.poll()
            self.assertEqual(len(calls),1)
            self.assertIsNotNone(peer.loan)
            peer.close()
        b.close()

class Producer:
    def __init__(self, decisions):
        self.decisions, self.routed, self.admits = list(decisions), [], []
    def request(self): return None
    def route(self, result):
        self.routed.append(result)
        return self.decisions.pop(0) if len(self.decisions) > 1 else self.decisions[0]
    def admitted(self, identifier): self.admits.append(identifier)


def read_frames(sock):
    data = bytearray()
    try:
        while True:
            chunk = sock.recv(4096)
            if not chunk: break
            data.extend(chunk)
    except BlockingIOError: pass
    return [decode_frame(bytes(data[i:i+72])) for i in range(0, len(data), 72)]


class EngineProtocolTests(unittest.TestCase):
    CONTRACT = (11, 2, 9, 17)
    RESULT = (11, 2, 9, 17, 3, 4, 100, 23)

    def test_new_kind_round_trips(self):
        cases = [(HELLO, encode_hello(2, 77), (2, 77, 2, 0, 0, 0, 0, 0)),
                 (SUBMIT, encode_submit(self.CONTRACT), (11, 2, 9, 17, 0, 0, 0, 0)),
                 (ADMITTED, encode_admitted(11), (11, 0, 0, 0, 0, 0, 0, 0)),
                 (ROUTE, encode_route(self.RESULT), self.RESULT),
                 (DECISION, encode_decision(11, 3, 1), (11, 0, 0, 0, 3, 0, 0, 1))]
        for kind, frame, values in cases:
            self.assertEqual(len(frame), 72)
            self.assertEqual(decode_frame(frame), (kind, values))
        self.assertEqual(decode_hello(encode_hello(1, 5)), (1, 5))
        with self.assertRaises(ValueError): decode(encode_hello(1, 5))  # not a host-protocol frame

    def test_malformed_hello_refused(self):
        with self.assertRaises(ValueError): decode_hello(encode_frame(HELLO, 2, 7, 1))
        with self.assertRaises(ValueError): decode_hello(encode_frame(HELLO, 9, 7, 2))
        with self.assertRaises(ValueError): decode_hello(encode_frame(HELLO, 2, 7, 2, 1))
        with self.assertRaises(ValueError): decode_hello(encode_submit(self.CONTRACT))

    def make(self, decisions):
        a, b = socket.socketpair(); b.setblocking(False)
        producer = Producer(decisions)
        return EnginePeer(a, producer), producer, b

    def test_submit_once_then_commit(self):
        peer, producer, b = self.make([1])
        self.assertTrue(peer.submit(self.CONTRACT)); self.assertTrue(peer.submit(self.CONTRACT))
        peer.poll(); peer.poll()
        self.assertEqual(read_frames(b), [(SUBMIT, (11, 2, 9, 17, 0, 0, 0, 0))])
        b.sendall(encode_admitted(11)); peer.poll()
        self.assertEqual(producer.admits, [11]); self.assertEqual(peer.pending, set())
        b.sendall(encode_route(self.RESULT)); peer.poll()
        self.assertEqual(producer.routed, [self.RESULT])
        self.assertEqual(read_frames(b), [(DECISION, (11, 0, 0, 0, 3, 0, 0, 1))])
        self.assertIsNone(peer.error); peer.close(); b.close()

    def test_hold_then_commit_reasks_producer(self):
        peer, producer, b = self.make([0, 1])
        b.sendall(encode_route(self.RESULT)); peer.poll()
        self.assertEqual(read_frames(b), [(DECISION, (11, 0, 0, 0, 3, 0, 0, 0))])
        b.sendall(encode_route(self.RESULT)); peer.poll()
        self.assertEqual(len(producer.routed), 2)
        self.assertEqual(read_frames(b), [(DECISION, (11, 0, 0, 0, 3, 0, 0, 1))])
        peer.close(); b.close()

    def test_reject_maps_to_two(self):
        peer, producer, b = self.make([-1])
        b.sendall(encode_route(self.RESULT)); peer.poll()
        self.assertEqual(read_frames(b), [(DECISION, (11, 0, 0, 0, 3, 0, 0, 2))])
        peer.close(); b.close()

    def test_partial_frame_retained_and_wrong_direction_stops(self):
        peer, producer, b = self.make([1])
        frame = encode_route(self.RESULT)
        b.sendall(frame[:30]); peer.poll()
        self.assertEqual(producer.routed, [])
        b.sendall(frame[30:]); peer.poll()
        self.assertEqual(len(producer.routed), 1)
        b.sendall(encode_hello(1, 1)); peer.poll()
        self.assertIsNotNone(peer.error)
        peer.close(); b.close()

    def test_submit_queue_is_bounded(self):
        a, b = socket.socketpair()
        peer = EnginePeer(a, Producer([1]), max_pending=2)
        self.assertTrue(peer.submit((1, 1, 1, 1))); self.assertTrue(peer.submit((2, 1, 1, 1)))
        self.assertFalse(peer.submit((3, 1, 1, 1)))
        peer.close(); b.close()

    def test_host_peer_hello_first_only_when_announced(self):
        for announce in (True, False):
            a, b = socket.socketpair(); b.setblocking(False)
            with ThreadPoolExecutor(max_workers=1) as executor:
                peer = ServicePeer(a, executor, lambda l: 1, lambda *_: None,
                                   announce=announce, identity=4)
                peer.poll()
                frames = read_frames(b)
                self.assertEqual(frames, [(HELLO, (1, 4, 2, 0, 0, 0, 0, 0))] if announce else [])
                peer.close()
            b.close()

    def test_connect_loopback_sends_hello(self):
        server = socket.socket(); server.bind(('127.0.0.1', 0)); server.listen(1)
        client = connect_loopback(server.getsockname()[1], 2, 9, 2.0)
        conn, _ = server.accept(); conn.settimeout(2.0)
        self.assertEqual(decode_hello(conn.recv(72)), (2, 9))
        client.close(); conn.close(); server.close()

if __name__ == '__main__': unittest.main()
