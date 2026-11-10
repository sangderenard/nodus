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
"""
from __future__ import annotations
from concurrent.futures import Executor
from dataclasses import dataclass
import socket
import struct

FRAME = struct.Struct('<II8Q')
MAGIC = 0x31434E4E
OFFER, ACCEPT, RESULT, RECEIPT = range(1, 5)

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


class ServicePeer:
    def __init__(self, connection: socket.socket, executor: Executor, dispatch, on_receipt):
        self.connection = connection
        connection.setblocking(False)
        self.executor, self.dispatch, self.on_receipt = executor, dispatch, on_receipt
        self.incoming = bytearray()
        self.outgoing = bytearray()
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


