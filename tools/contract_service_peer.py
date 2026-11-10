"""Development-service peer for the native contract_socket.h protocol.

The caller supplies an already authenticated connected socket and a dispatcher.
A contract contains opaque references, NOT permission to access a user/table.
The dispatcher must resolve those references within its authorized session and
return a retained result reference. Dispatch MUST be pure/staged computation:
an offer/accept exchange is not an ownership grant. Mutations happen only at
the authorized producer commit after native validation. Do not dispatch action
batches here directly. No Python queue replaces the Nodus FIFO.
Poll never waits for service execution. Receipts mean FIFO admission, not commit.
Failed/uncertain calls are never automatically replayed. Restart recovery and
transport authentication belong to the host; this adapter creates neither.

EnginePeer is the producer side: it submits contracts, and answers native ROUTE
frames with the producer's own commit disposition. A DECISION of hold only says
"alive, undecided"; nothing is resent except that a ROUTE the native side repeats
is re-asked of the producer. A SUBMIT is sent once per id until ADMITTED.
"""
from __future__ import annotations
from concurrent.futures import Executor
from dataclasses import dataclass
import socket
import struct

FRAME = struct.Struct('<II8Q')
MAGIC = 0x31434E4E
OFFER, ACCEPT, RESULT, RECEIPT, HELLO, SUBMIT, ADMITTED, ROUTE, DECISION = range(1, 10)
ROLE_HOST, ROLE_ENGINE = 1, 2
PROTOCOL_VERSION = 2
DECISION_HOLD, DECISION_COMMIT, DECISION_REJECT = 0, 1, 2
MAX_OUTGOING_FRAMES = 64
MAX_FRAMES_PER_POLL = 64

@dataclass(frozen=True)
class Loan:
    contract: int
    engine: int
    revision: int
    payload: int
    generation: int
    host: int
    deadline: int

    def values(self):
        return (self.contract, self.engine, self.revision, self.payload,
                self.generation, self.host, self.deadline)


def encode(kind, loan, result=0):
    return FRAME.pack(MAGIC, kind, *loan.values(), result)


def decode(frame):
    magic, kind, *values = FRAME.unpack(frame)
    if magic != MAGIC or kind not in (OFFER, ACCEPT, RESULT, RECEIPT):
        raise ValueError('invalid native contract frame')
    return kind, Loan(*values[:7]), values[7]


def encode_frame(kind, *values):
    """Generic frame: kind plus up to eight uint64 values (zero padded)."""
    if len(values) > 8:
        raise ValueError('a frame carries at most eight values')
    return FRAME.pack(MAGIC, kind, *values, *([0] * (8 - len(values))))


def decode_frame(frame, kinds=range(1, 10)):
    magic, kind, *values = FRAME.unpack(frame)
    if magic != MAGIC or kind not in kinds:
        raise ValueError('invalid native contract frame')
    return kind, tuple(values)


def encode_hello(role, identity):
    if role not in (ROLE_HOST, ROLE_ENGINE):
        raise ValueError('unknown hello role')
    return encode_frame(HELLO, role, identity, PROTOCOL_VERSION)


def decode_hello(frame):
    kind, v = decode_frame(frame)
    if (kind != HELLO or v[0] not in (ROLE_HOST, ROLE_ENGINE)
            or v[2] != PROTOCOL_VERSION or any(v[3:])):
        raise ValueError('malformed or unsupported hello')
    return v[0], v[1]


def encode_submit(contract):
    if len(contract) != 4:
        raise ValueError('contract is (id, engine, revision, payload)')
    return encode_frame(SUBMIT, *contract)


def encode_admitted(identifier):
    return encode_frame(ADMITTED, identifier)


def encode_route(result):
    if len(result) != 8:
        raise ValueError('result is eight uint64 values')
    return encode_frame(ROUTE, *result)


def encode_decision(identifier, generation, decision):
    return encode_frame(DECISION, identifier, 0, 0, 0, generation, 0, 0, decision)


def connect_loopback(port, role, identity, timeout=5.0):
    """Connect to 127.0.0.1 only; Hello is sent before the socket is returned."""
    connection = socket.create_connection(('127.0.0.1', port), timeout=timeout)
    try:
        connection.sendall(encode_hello(role, identity))
    except Exception:
        connection.close()
        raise
    return connection


class ServicePeer:
    def __init__(self, connection: socket.socket, executor: Executor, dispatch, on_receipt,
                 announce=True, identity=0):
        self.connection = connection
        connection.setblocking(False)
        self.executor, self.dispatch, self.on_receipt = executor, dispatch, on_receipt
        self.incoming = bytearray()
        self.outgoing = bytearray()
        if announce:
            self.outgoing.extend(encode_hello(ROLE_HOST, identity))
        self.loan = self.future = None
        self.result_sent = False
        self.high_water_generation = 0
        self.closed = False
        self.error = None

    def poll(self):
        if self.closed or self.error is not None:
            return
        try:
            if self.future is not None and self.future.done() and not self.result_sent:
                # An exception may follow a partially applied operation. Retain
                # identity and stop: the owner must reconcile, not resubmit.
                result = self.future.result()
                if type(result) is not int or not 0 <= result < 2**64:
                    raise ValueError('dispatcher must return retained uint64 result reference')
                self.outgoing.extend(encode(RESULT, self.loan, result))
                self.result_sent = True
            if self.outgoing:
                try:
                    sent = self.connection.send(self.outgoing)
                    if not sent:
                        self.closed = True
                        return
                    del self.outgoing[:sent]
                except BlockingIOError:
                    pass
            try:
                data = self.connection.recv(FRAME.size - len(self.incoming))
            except BlockingIOError:
                return
            if not data:
                self.closed = True
                return
            self.incoming.extend(data)
            if len(self.incoming) != FRAME.size:
                return
            kind, loan, result = decode(self.incoming)
            self.incoming.clear()
            if kind == OFFER:
                if self.loan is not None or loan.generation <= self.high_water_generation:
                    # Same offer is harmless; different offers are declined by
                    # silence until our retained result has its FIFO receipt.
                    return
                self.loan = loan
                self.high_water_generation = loan.generation
                self.future = self.executor.submit(self.dispatch, loan)
                self.outgoing.extend(encode(ACCEPT, loan))
            elif kind == RECEIPT:
                if self.loan is None or loan.generation != self.loan.generation or not self.result_sent:
                    raise ValueError('unmatched or premature FIFO receipt')
                if result not in (0, 1):
                    raise ValueError('invalid FIFO receipt')
                self.on_receipt(self.loan, bool(result))
                self.loan = self.future = None
                self.result_sent = False
            else:
                raise ValueError('service peer received wrong direction frame')
        except Exception as exc:
            self.error = exc

    def close(self):
        self.connection.close()
        self.closed = True


class EnginePeer:
    """Producer-side peer. Never blocks; whole frames only; bounded queues.

    producer.request() -> contract tuple or None (peek; retained until admitted)
    producer.route(result) -> 1 commit / 0 hold / -1 reject (result: 8-tuple)
    producer.admitted(identifier)
    """
    def __init__(self, connection: socket.socket, producer, max_pending=MAX_OUTGOING_FRAMES):
        self.connection, self.producer = connection, producer
        connection.setblocking(False)
        self.max_pending = max_pending
        self.incoming = bytearray()
        self.outgoing = bytearray()
        self.pending = set()      # submitted, not yet ADMITTED
        self.errors = []          # producer exceptions; answered with hold
        self.closed = False
        self.error = None

    def submit(self, contract):
        """Send SUBMIT once per contract id until ADMITTED. False = not queued."""
        if self.closed or self.error is not None:
            return False
        identifier = contract[0]
        if identifier in self.pending:
            return True
        if (len(self.pending) >= self.max_pending
                or len(self.outgoing) >= MAX_OUTGOING_FRAMES * FRAME.size):
            return False
        self.outgoing.extend(encode_submit(contract))
        self.pending.add(identifier)
        return True

    def _flush(self):
        if self.outgoing:
            try:
                sent = self.connection.send(self.outgoing)
            except BlockingIOError:
                return
            if not sent:
                self.closed = True
                return
            del self.outgoing[:sent]

    def poll(self):
        if self.closed or self.error is not None:
            return
        try:
            self._flush()
            for _ in range(MAX_FRAMES_PER_POLL):
                try:
                    data = self.connection.recv(FRAME.size - len(self.incoming))
                except BlockingIOError:
                    break
                if not data:
                    self.closed = True
                    break
                self.incoming.extend(data)
                if len(self.incoming) != FRAME.size:
                    continue
                kind, v = decode_frame(bytes(self.incoming))
                self.incoming.clear()
                if kind == ADMITTED:
                    self.pending.discard(v[0])
                    self._guard(self.producer.admitted, v[0])
                elif kind == ROUTE:
                    self._route(v)
                else:
                    raise ValueError('engine peer received wrong direction frame')
            self._flush()
        except Exception as exc:
            self.error = exc

    def _guard(self, call, *args):
        try:
            return call(*args)
        except Exception as exc:
            self.errors.append(exc)

    def _route(self, result):
        # A repeated ROUTE (native re-asks while held) re-asks the producer.
        disposition = self._guard(self.producer.route, result)
        decision = {1: DECISION_COMMIT, 0: DECISION_HOLD, -1: DECISION_REJECT}.get(disposition)
        if decision is None:
            self.errors.append(ValueError('invalid commit disposition'))
            decision = DECISION_HOLD
        if len(self.outgoing) >= MAX_OUTGOING_FRAMES * FRAME.size:
            raise OverflowError('engine peer outgoing queue full')
        self.outgoing.extend(encode_decision(result[0], result[4], decision))

    def close(self):
        self.connection.close()
        self.closed = True


