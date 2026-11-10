import importlib.util
from pathlib import Path
import socket
import sys
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from contract_service_peer import ServicePeer, Loan, encode, decode, OFFER, ACCEPT, RESULT, RECEIPT, FRAME

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
            peer = ServicePeer(a, executor, dispatch, lambda l, ok: receipts.append((l, ok)))
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
            peer=ServicePeer(a,executor,fail,lambda *_:None)
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

if __name__ == '__main__': unittest.main()
