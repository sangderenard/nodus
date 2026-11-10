"""Retained payload side channel for native uint64 contract handles.

Not a scheduler or authorization service. Use a separately scoped store for each
writer role inside an already-authorized job directory, shared by parent/worker.
One writer owns admission/release; readers never delete. Native FIFO determines
ordering and engine validation determines commit. No credentials in payloads.
"""
import hashlib
import json
from pathlib import Path
import secrets

class PayloadStore:
    def __init__(self,directory,*,max_bytes=32*1024*1024,max_items=8):
        if max_bytes<=0 or max_items<=0:raise ValueError('invalid payload bounds')
        self.directory=Path(directory).resolve()
        self.directory.mkdir(parents=True,exist_ok=True)
        self.max_bytes,self.max_items=max_bytes,max_items

    def _path(self,handle):
        if type(handle) is not int or not 0<handle<2**64:raise ValueError('invalid opaque handle')
        return self.directory / f'{handle:016x}.json'

    def publish(self,identity,payload):
        if not isinstance(identity,dict) or not identity:raise ValueError('explicit identity required')
        raw=json.dumps({'identity':identity,'payload':payload},allow_nan=False,separators=(',',':'),sort_keys=True).encode()
        if len(raw)>self.max_bytes:raise ValueError('payload exceeds byte budget')
        if len(list(self.directory.glob('*.json')))>=self.max_items:raise BufferError('retained payload store full')
        # Exclusive reserved handle; no partially written final file is visible.
        for _ in range(8):
            handle=secrets.randbits(64) or 1
            path=self._path(handle);temporary=path.with_suffix('.pending')
            if path.exists():continue
            try:
                with temporary.open('xb') as stream:stream.write(raw)
            except FileExistsError:continue
            digest=hashlib.sha256(raw).hexdigest()
            # Digest is returned via caller's retained registry, not treated as auth.
            temporary.rename(path)
            return handle,digest
        raise RuntimeError('unable to reserve payload handle')

    def read(self,handle,identity,*,digest=None):
        path=self._path(handle)
        # Bounded read even if another process replaces an unexpectedly large file.
        with path.open('rb') as stream:raw=stream.read(self.max_bytes+1)
        if len(raw)>self.max_bytes:raise ValueError('payload exceeds byte budget')
        if digest is not None and hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('payload digest differs')
        value=json.loads(raw,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite payload')))
        if not isinstance(value,dict) or set(value)!={'identity','payload'} or value['identity']!=identity:
            raise ValueError('payload identity differs from retained contract')
        return value['payload']

    def release(self,handle):
        """Owner only, AFTER commit/rejection/cancel is reconciled, never on receipt."""
        self._path(handle).unlink()
