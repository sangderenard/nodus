"""ctypes host for actual native Nodus two-table/FIFO contract negotiation.

Callbacks are mailbox-only, must not throw or touch GL. Route returns 1 commit,
0 defer/uncertain hold, -1 authoritative reject/requeue. All full scene identity,
authorization and estimator commit remain the engine owner's responsibility.
Create/step/retire/close must be serialized; keep callbacks alive until close.
"""
import ctypes as C

class Contract(C.Structure):
    _fields_=[(n,C.c_uint64) for n in ('id','engine','revision','payload')]
class Loan(C.Structure):
    _fields_=[('work',Contract)]+[(n,C.c_uint64) for n in ('generation','host','deadline')]
class Result(C.Structure):
    _fields_=[('loan',Loan),('payload',C.c_uint64)]
Request=C.CFUNCTYPE(C.c_int,C.c_void_p,C.POINTER(Contract))
Route=C.CFUNCTYPE(C.c_int,C.c_void_p,C.POINTER(Result))
Admitted=C.CFUNCTYPE(None,C.c_void_p,C.c_uint64)

class NativeMailHost:
    def __init__(self,dll,connection,engine,request,route,admitted,capacity=64,ticks=100000):
        self.lib=C.CDLL(str(dll));self.errors=[]
        def req(_,out):
            try:
                value=request()
                if value is None:return 0
                out[0]=Contract(*value);return 1
            except Exception as exc:self.errors.append(exc);return 0
        def routed(_,value):
            try:
                # Copy: native pending storage can change after callback returns.
                result=Result.from_buffer_copy(value.contents)
                decision=route(result)
                if decision not in (-1,0,1):raise ValueError('invalid commit disposition')
                return decision
            except Exception as exc:self.errors.append(exc);return 0
        def ack(_,identifier):
            try:admitted(identifier)
            except Exception as exc:self.errors.append(exc)
        self.callbacks=(Request(req),Route(routed),Admitted(ack))
        self.lib.nodus_mail_create.argtypes=[C.c_size_t,C.c_uint64,C.c_size_t,C.c_uint64,Request,Route,Admitted,C.c_void_p]
        self.lib.nodus_mail_create.restype=C.c_void_p
        for name in ('nodus_mail_step','nodus_mail_destroy'):
            getattr(self.lib,name).argtypes=[C.c_void_p]
        self.lib.nodus_mail_destroy.restype=None
        self.lib.nodus_mail_retire.argtypes=[C.c_void_p,C.c_uint64]
        # Native host owns the connected socket after this transfer.
        fd=connection.detach()
        self.handle=self.lib.nodus_mail_create(fd,engine,capacity,ticks,*self.callbacks,None)
        if not self.handle:raise RuntimeError('native Nodus service initialization failed')
    def add_participant(self,connection,participant):
        if not self.handle:raise RuntimeError("native host closed")
        function=self.lib.nodus_mail_add_participant
        function.argtypes=[C.c_void_p,C.c_size_t,C.c_uint64]
        if not function(self.handle,connection.detach(),participant):raise RuntimeError("native participant refused")
    def step(self):
        if not self.handle or not self.lib.nodus_mail_step(self.handle):raise RuntimeError('native graph step failed')
    def retire(self,identifier):return bool(self.lib.nodus_mail_retire(self.handle,identifier))
    def close(self):
        if self.handle:self.lib.nodus_mail_destroy(self.handle);self.handle=None

