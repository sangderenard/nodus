
from __future__ import annotations



from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional, List, Union, Callable, Dict, Deque, NamedTuple, Iterable, TYPE_CHECKING
import math
import time
from collections import deque

# Wire in new abstraction_methods/properties
from .abstraction_methods import properties as _properties
from .abstraction_methods import reshape as _reshape_methods

# TYPE: Faculty, DEFAULT_FACULTY should be imported from .faculty
try:
    from .faculty import Faculty, DEFAULT_FACULTY
except ImportError:
    Faculty = None  # TYPE: ignore
    DEFAULT_FACULTY = None  # TYPE: ignore

DEFAULT_DEVICE = "cpu"

# Optional dependencies
try:
    import torch
except ImportError:
    torch = None  # TYPE: ignore
try:
    import numpy as np
except ImportError:
    np = None  # TYPE: ignore
try:
    from .accelerator_backends.c_backend import CTensor
except ImportError:
    CTensor = None  # TYPE: ignore

# TYPE: register_conversion, CONVERSION_REGISTRY, DEBUG, _get_ops_for_class
from . import DEBUG
from .logger import get_tensors_logger

logger = get_tensors_logger()

# Global tensor pool ---------------------------------------------------------
try:  # pragma: no cover - optional dependency
    from .pooling.tensor_pool import TensorPool, PoolPolicy
except Exception:  # pragma: no cover - pool is optional
    TensorPool = None  # type: ignore
    PoolPolicy = None  # type: ignore


if TensorPool is not None and np is not None:
    class _PoolATAdapter:
        """Backend adapter so :class:`TensorPool` can allocate AbstractTensors."""

        def __init__(self, AT_cls):
            self.AT = AT_cls

        def empty(self, shape, dtype=None, device=None):
            t = self.AT(track_time=False, tape=None)
            try:
                t.data = t.empty_(shape, dtype=dtype, device=device)
            except Exception:
                t.data = np.empty(shape, dtype=dtype or np.float32)
            return t

        def fill0_(self, buf):
            d = getattr(buf, "data", buf)
            try:
                d[...] = 0
            except Exception:
                if isinstance(d, list):
                    def _fill(lst):
                        for i, v in enumerate(lst):
                            if isinstance(v, list):
                                _fill(v)
                            else:
                                lst[i] = 0
                    _fill(d)
                else:
                    try:
                        np.asarray(d)[...] = 0
                    except Exception:
                        pass

        def nbytes(self, buf):
            d = getattr(buf, "data", buf)
            return int(getattr(d, "nbytes", np.asarray(d).nbytes))

        def detach(self, buf):
            try:
                return buf.detach()  # type: ignore[attr-defined]
            except Exception:
                return buf

    _GLOBAL_TENSOR_POOL: TensorPool | None = None

    def _get_tensor_pool(cls):
        global _GLOBAL_TENSOR_POOL
        if _GLOBAL_TENSOR_POOL is None or _GLOBAL_TENSOR_POOL.B.AT is not cls:
            try:
                _GLOBAL_TENSOR_POOL = TensorPool(backend=_PoolATAdapter(cls), policy=PoolPolicy())
            except Exception:
                _GLOBAL_TENSOR_POOL = None
        return _GLOBAL_TENSOR_POOL
else:  # pragma: no cover - pool not available
    def _get_tensor_pool(cls):  # type: ignore[override]
        return None

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .autograd import GradTape
def register_conversion(*args, **kwargs):
    pass
CONVERSION_REGISTRY = dict()



# ---- diagnostics ------------------------------------------------------------
import os
from dataclasses import dataclass

DIAG_LEVEL = os.environ.get("ABSTRACT_TENSOR_DIAG", "auto")  # 'concise' | 'auto' | 'verbose'

@dataclass
class _Diag:
    op: str | None = None          # e.g., "broadcast_rows"
    tensor: str | None = None      # e.g., "Linear.forward(bias)"
    expected: str | None = None    # e.g., "(1, 16) or (32, 16)"
    actual: str | None = None      # e.g., "(8, 16)"
    batch_size: int | None = None
    hint: str | None = None

class TensorShapeError(ValueError):
    def __init__(self, message: str, diag: _Diag | None = None):
        self._message = message
        self._diag = diag
        super().__init__(str(self))

    def __str__(self) -> str:
        d = self._diag
        if d is None:
            return self._message
        head = self._message
        # concise core: one line with labels + shapes
        parts = []
        if d.op:     parts.append(d.op)
        if d.tensor: parts.append(d.tensor)
        prefix = f"{' in '.join(parts)}: " if parts else ""
        line1 = f"{prefix}expected {d.expected}"
        if d.batch_size is not None:
            line1 += f" for batch_size={d.batch_size}"
        line1 += f", got {d.actual}."
        # hint policy
        want_hint = (DIAG_LEVEL == "verbose") or (DIAG_LEVEL == "auto" and d.hint)
        if want_hint:
            return head + "\n" + line1 + (f"\nHint: {d.hint}" if d.hint else "")
        return head + "\n" + line1







# --- Backend Registry Pattern ---
# This registry allows dynamic discovery and decoupling of tensor backends.
# Each backend module registers itself here at import time, avoiding all circular imports.
BACKEND_REGISTRY: dict[str, type] = {}

def register_backend(name: str, backend_cls: type) -> None:
    """
    Register a tensor backend class under a given name.
    Backends should call this after their class definition.
    """
    BACKEND_REGISTRY[name] = backend_cls

# Delayed import to avoid circular dependency
class MeshGrid:
    """Tuple-like wrapper for meshgrid results."""
    def __init__(self, tensors):
        self.tensors = tuple(tensors)

    def __iter__(self):
        return iter(self.tensors)

        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx]

    def __repr__(self):
        return f"MeshGrid(len={len(self.tensors)}, shape={self.shape})"

    @property
    def shape(self):
        return self.tensors[0].shape if self.tensors else ()




def _register_all_conversions():
    NumPyTensorOperations = BACKEND_REGISTRY.get("numpy")
    PyTorchTensorOperations = BACKEND_REGISTRY.get("torch")
    JAXTensorOperations = BACKEND_REGISTRY.get("jax")
    PurePythonTensorOperations = BACKEND_REGISTRY.get("pure_python")
class AbstractTensor:
    inf: float = float('inf')
    ninf: float = float('-inf')  
    nan: float = float('nan')

    @staticmethod
    def nan_to_num(x, nan: float = 0.0, posinf: float = inf, neginf: float = ninf) -> "AbstractTensor":
        """Return a tensor with NaN and infinities replaced.

        The function accepts either an :class:`AbstractTensor` instance or any
        array-like object.  Replacements mirror ``numpy.nan_to_num`` with the
        default behaviour matching NumPy's API.
        """
        if not isinstance(x, AbstractTensor):
            x = AbstractTensor.tensor(x)

        x = AbstractTensor.where(AbstractTensor.isnan(x), nan, x)
        x = AbstractTensor.where(x == AbstractTensor.inf, posinf, x)
        x = AbstractTensor.where(x == AbstractTensor.ninf, neginf, x)
        return x

    def __index__(self):
        

        """Allow use in slice contexts by returning a scalar ``int``.

        Non-scalar tensors raise ``TypeError`` mirroring PyTorch/NumPy."""

        try:
            n = self.numel()
        except Exception:
            n = 1
        if n != 1:
            raise TypeError("Only scalar tensors can be converted to int")

        item_fn = getattr(self, "item_")
        try:
            import inspect
            if len(inspect.signature(item_fn).parameters) == 0:
                value = item_fn()
            else:
                value = item_fn(self.data)
        except Exception:
            value = item_fn(self.data)
        return int(value)
    def argwhere(self) -> "AbstractTensor":
        """Return the indices where condition is True. Like np.argwhere, always returns a 2D array of indices."""
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.argwhere_()
        return result

    def argwhere_(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement argwhere_()")
    @staticmethod
    def _normalize_shape_args(*shape):
        """
        Accepts: reshape(-1), reshape(2,3), reshape([2,3]), reshape((2,3))
        Returns a tuple shape (e.g., (-1,), (2,3))
        """
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            return tuple(int(s) for s in shape[0])
        if len(shape) == 1 and isinstance(shape[0], int):
            return (int(shape[0]),)
        return tuple(int(s) for s in shape)

    def empty_(self, size: Tuple[int, ...], dtype: Any = None, device: Any = None):
        """Create an uninitialized tensor of the given shape (backend hook)."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement empty_()")
    @staticmethod
    def backend_class_from_backend_data(data):
        """
        Given a backend-native data object (e.g., torch.Tensor, np.ndarray),
        return the registered backend class that wraps this data type.
        """
        # Ensure registry is populated

        for backend_cls in BACKEND_REGISTRY.values():
            if type(data) == backend_cls:
                return backend_cls
            
            tensor_type = backend_cls.tensor_type_
            
            if tensor_type is not None and isinstance(data, tensor_type):
                return backend_cls
        return None
    # Redirect backward to autograd implementation for user convenience

    def backward(self, *args, **kwargs):
        from .autograd import backward as _autograd_backward
        return _autograd_backward(self, *args, **kwargs)
    """Abstraction layer for tensor operations."""
    # --- Sentinel dtypes for use before backend is set ---
    float_dtype_ = 'float32'  # Default sentinel, can be replaced by backend
    long_dtype_ = 'int64'     # Default sentinel, can be replaced by backend
    bool_dtype_ = 'bool'      # Default sentinel, can be replaced by backend
    # --- Unary operators ---


    def __neg__(self):
        return self._apply_operator("neg", self, None)

    def __pos__(self):
        # +tensor is a no-op, return self
        return self

    def sign(self):
        return self._apply_operator("sign", self, None)

    def abs(self):
        return self._apply_operator("abs", self, None)

    def __abs__(self):
        return self._apply_operator("abs", self, None)


    def __invert__(self):
        return self._apply_operator("invert", self, None)


    def __round__(self, n=None):
        raise NotImplementedError(f"{self.__class__.__name__} must implement __round__()")


    def __trunc__(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement __trunc__()")


    def __floor__(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement __floor__()")


    def __ceil__(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement __ceil__()")

    # --- Backend hooks for unary ops (must be implemented by backends) ---
    def neg_(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement neg_()")

    def abs_(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement abs_()")

    def invert_(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement invert_()")

    def round_(self, n=None):
        raise NotImplementedError(f"{self.__class__.__name__} must implement round_()")

    def trunc_(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement trunc_()")

    def floor_(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement floor_()")

    def ceil_(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement ceil_()")
    def max_(self, dim=None, keepdim: bool = False):
        raise NotImplementedError(f"{self.__class__.__name__} must implement max_() with keepdim.")

    def argmax_(self, dim=None, keepdim: bool = False):
        raise NotImplementedError(f"{self.__class__.__name__} must implement argmax_() with keepdim.")

    # --- Clamping ---
    def clamp(self, min: float | None = None, max: float | None = None) -> "AbstractTensor":
        """Return ``self`` clamped between ``min`` and ``max`` with autograd support."""
        finalize = AbstractTensor._pre_autograd("clamp", [self], params={"min": min, "max": max})
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.clamp_(min_val=min, max_val=max)
        return finalize(result)

    def clamp_(self, min_val: float | None = None, max_val: float | None = None):
        raise NotImplementedError(f"{self.__class__.__name__} must implement clamp_()")

    def clamp_min(self, min_val: float) -> "AbstractTensor":
        """Clamp values below ``min_val`` up to ``min_val``."""
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.clamp_min_(min_val)
        return result

    def clamp_min_(self, min_val: float):
        raise NotImplementedError(f"{self.__class__.__name__} must implement clamp_min_()")

    def clamp_max(self, max_val: float) -> "AbstractTensor":
        """Clamp values above ``max_val`` down to ``max_val``."""
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.clamp_max_(max_val)
        return result

    def clamp_max_(self, max_val: float):
        raise NotImplementedError(f"{self.__class__.__name__} must implement clamp_max_()")

    # --- API compatibility ---
    def clip(
        self,
        min: float | None = None,
        max: float | None = None,
        *,
        a_min: float | None = None,
        a_max: float | None = None,
    ) -> "AbstractTensor":
        """Alias for :meth:`clamp` accepting NumPy-style parameters."""
        if (min is not None or max is not None) and (a_min is not None or a_max is not None):
            raise TypeError("Specify either min/max or a_min/a_max, not both")
        if min is None and max is None:
            min, max = a_min, a_max
        return self.clamp(min=min, max=max)

    def greater_(self, value):
        raise NotImplementedError(f"{self.__class__.__name__} must implement greater_()")
    def greater_equal_(self, value):
        raise NotImplementedError(f"{self.__class__.__name__} must implement greater_equal_()")

    def less_(self, value):
        raise NotImplementedError(f"{self.__class__.__name__} must implement less_()")

    def less_equal_(self, value):
        raise NotImplementedError(f"{self.__class__.__name__} must implement less_equal_()")

    def equal_(self, value):
        raise NotImplementedError(f"{self.__class__.__name__} must implement equal_()")
    def not_equal_(self, value):
        raise NotImplementedError(f"{self.__class__.__name__} must implement not_equal_()")

    # --- Logical ---
    def logical_not(self) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.logical_not_()
        return result

    def logical_not_(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement logical_not_()")

    # --- Unary math ---
    def sqrt(self) -> "AbstractTensor":
        if isinstance(self, AbstractTensor):
            result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        else:
            result = AbstractTensor.get_tensor(self.data, track_time=False)
        result.data = self.sqrt_()
        return result

    def sqrt_(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement sqrt_()")

    def exp(self) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.exp_()
        return result

    def exp_(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement exp_()")

    def log(self) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.log_()
        return result

    def log_(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement log_()")

    @staticmethod
    def real(x) -> "AbstractTensor":
        """Return the real part of a complex tensor."""
        if not isinstance(x, AbstractTensor):
            x = AbstractTensor.tensor(x)
        result = type(x)(track_time=x.track_time, tape=getattr(x, "_tape", None))
        result.data = x.real_()
        return result

    def real_(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement real_()")

    @staticmethod
    def imag(x) -> "AbstractTensor":
        """Return the imaginary part of a complex tensor."""
        if not isinstance(x, AbstractTensor):
            x = AbstractTensor.tensor(x)
        result = type(x)(track_time=x.track_time, tape=getattr(x, "_tape", None))
        result.data = x.imag_()
        return result

    def imag_(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement imag_()")

    # --- Softmax utilities ---
    def softmax(self, dim: int = -1) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.softmax_(dim)
        return result

    def softmax_(self, dim):
        raise NotImplementedError(f"{self.__class__.__name__} must implement softmax_()")

    def log_softmax(self, dim: int = -1) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.log_softmax_(dim)
        return result

    def log_softmax_(self, dim):
        raise NotImplementedError(f"{self.__class__.__name__} must implement log_softmax_()")

    # --- Basic layout ---
    def mean(self, dim=None, keepdim: bool = False):
        """Return the mean of the tensor along the specified dimension(s)."""
        finalize = AbstractTensor._pre_autograd(
            "mean", [self], params={"axis": dim, "keepdim": keepdim}
        )
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.mean_(dim=dim, keepdim=keepdim)
        result = finalize(result)
        if getattr(result.data, "shape", ()) == ():
            return AbstractScalar(result)
        return result

    def sum(self, dim=None, keepdim: bool = False):
        """Return the sum of the tensor along the specified dimension(s)."""
        finalize = AbstractTensor._pre_autograd(
            "sum", [self], params={"axis": dim, "keepdim": keepdim}
        )
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.sum_(dim=dim, keepdim=keepdim)
        result = finalize(result)
        if getattr(result.data, "shape", ()) == ():
            return AbstractScalar(result)
        return result

    def cumsum(self, dim: int = 0) -> "AbstractTensor":
        """Return the cumulative sum of the tensor along a dimension."""
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.cumsum_(dim)
        return result

    def min(self, dim=None, keepdim: bool = False):
        """Return the minimum of the tensor along the specified dimension(s)."""
        finalize = AbstractTensor._pre_autograd(
            "min", [self], params={"axis": dim, "keepdim": keepdim}
        )
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.min_(dim=dim, keepdim=keepdim)
        result = finalize(result)
        if getattr(result.data, "shape", ()) == ():
            return AbstractScalar(result)
        return result

    # --- Backend hooks for reductions (must be implemented by backends) ---
    def mean_(self, dim=None, keepdim: bool = False):
        raise NotImplementedError(f"{self.__class__.__name__} must implement mean_() with keepdim.")

    def sum_(self, dim=None, keepdim: bool = False):
        raise NotImplementedError(f"{self.__class__.__name__} must implement sum_() with keepdim.")

    def cumsum_(self, dim: int = 0):
        raise NotImplementedError(f"{self.__class__.__name__} must implement cumsum_()")

    def min_(self, dim=None, keepdim: bool = False):
        raise NotImplementedError(f"{self.__class__.__name__} must implement min_() with keepdim.")

    def argmin_(self, dim=None, keepdim: bool = False):
        raise NotImplementedError(f"{self.__class__.__name__} must implement argmin_() with keepdim.")

    def tolist_(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement tolist_() for conversion to list.")

    # --- Fourier / FFT backend hooks --------------------------------------
    def fft_(self, n: int | None = None, axis: int = -1, norm: str | None = None):
        raise NotImplementedError(f"{self.__class__.__name__} must implement fft_()")

    def ifft_(self, n: int | None = None, axis: int = -1, norm: str | None = None):
        raise NotImplementedError(f"{self.__class__.__name__} must implement ifft_()")

    def rfft_(self, n: int | None = None, axis: int = -1, norm: str | None = None):
        raise NotImplementedError(f"{self.__class__.__name__} must implement rfft_()")

    def irfft_(self, n: int | None = None, axis: int = -1, norm: str | None = None):
        raise NotImplementedError(f"{self.__class__.__name__} must implement irfft_()")

    def rfftfreq_(self, n: int, d: float = 1.0):
        raise NotImplementedError(f"{self.__class__.__name__} must implement rfftfreq_()")

    def fftfreq_(self, n: int, d: float = 1.0):
        raise NotImplementedError(f"{self.__class__.__name__} must implement fftfreq_()")
    def _AbstractTensor__unwrap(self, obj=None):
        """Return the underlying tensor data for this AbstractTensor or for another AbstractTensor instance."""
        if obj is None:
            return self.data
        if isinstance(obj, AbstractTensor):
            return obj.data
        return obj
    def __init__(self, track_time: bool = False, requires_grad: bool = False, tape: "GradTape" | None = None):
        """Create a new tensor wrapper.

        Parameters
        ----------
        track_time:
            Unused placeholder retained for API compatibility.
        tape:
            Optional :class:`GradTape` the tensor should attach to. If omitted,
            the global ``autograd`` tape is used and the tensor registers itself
            as a root node on that graph.
        """

        self.track_time = track_time
        if tape is None and requires_grad:
            from . import autograd as _autograd  # local import to avoid cycle
            tape = _autograd.autograd.tape
        if requires_grad:
            self._tape = tape
            self._tape.create_tensor_node(self)


    def tensor_from_list_(self, data, dtype=None, device=None):
        """
        Create a tensor from a Python list using the first suitable backend in BACKEND_REGISTRY (including torch),
        or try to import numpy/pure_python as fallback. Use the first available backend found.
        """
        if isinstance(data, AbstractTensor):
            return data.data

        # 1. Try any already-registered backend
        for name, backend_cls in BACKEND_REGISTRY.items():
            if backend_cls is not None:
                inst = backend_cls(track_time=False)
                return inst.tensor_from_list_(data, dtype, device)

        # 2. Try to import and register numpy backend
        backend_cls = BACKEND_REGISTRY.get("numpy")
        if backend_cls is None:
            try:
                from . import numpy_backend  # noqa: F401
                backend_cls = BACKEND_REGISTRY.get("numpy")
            except Exception:
                backend_cls = None
        if backend_cls is not None:
            inst = backend_cls(track_time=False)
            return inst.tensor_from_list_(data, dtype, device)

        # 3. Try to import and register pure_python backend
        backend_cls = BACKEND_REGISTRY.get("pure_python")
        if backend_cls is None:
            try:
                from . import pure_backend  # noqa: F401
                backend_cls = BACKEND_REGISTRY.get("pure_python")
            except Exception:
                backend_cls = None
        if backend_cls is not None:
            inst = backend_cls(track_time=False)
            return inst.tensor_from_list_(data, dtype, device)

        # 4. If all else fails, raise an error
        raise RuntimeError("No suitable tensor backend is available to create a tensor from list.")

    @classmethod
    def _tensor_from_list(
        cls,
        data,
        dtype=None,
        device=None,
        tape: "GradTape" | None = None,
        *,
        like: "AbstractTensor" | None = None,
        requires_grad: bool = False,
    ):
        """Centralized list->tensor creation.

        - If called on AbstractTensor, choose backend from ``like`` or registry.
        - If called on a backend subclass, use it directly.
        - Always attaches to ``tape`` if provided and sets requires_grad when requested.
        """
        # Resolve target backend
        target_cls = cls
        if cls is AbstractTensor:
            if like is not None:
                target_cls = type(like)
            else:
                target_cls = AbstractTensor.check_or_build_registry()

        inst = target_cls(track_time=False, tape=tape)
        inst.data = inst.tensor_from_list_(data, dtype, device)
        if requires_grad:
            try:
                inst.requires_grad_(True)
            except Exception:
                try:
                    inst._requires_grad = True
                except Exception:
                    pass
        return inst

    # --- Tensor creation and manipulation methods ---

    def full_(
        self,
        size: Tuple[int, ...],
        fill_value: Any,
        dtype: Any = None,
        device: Any = None,
    ):
        raise NotImplementedError(f"{self.__class__.__name__} must implement full_()")

    def zeros_(self, size: Tuple[int, ...], dtype: Any = None, device: Any = None):
        raise NotImplementedError(f"{self.__class__.__name__} must implement zeros_()")

    def ones_(self, size: Tuple[int, ...], dtype: Any = None, device: Any = None):
        raise NotImplementedError(f"{self.__class__.__name__} must implement ones_()")

    def zeros_like_(self, dtype: Any = None, device: Any = None):
        raise NotImplementedError(f"{self.__class__.__name__} must implement zeros_like_()")

    def ones_like_(self, dtype: Any = None, device: Any = None):
        raise NotImplementedError(f"{self.__class__.__name__} must implement ones_like_()")

    def full_like_(
        self,
        fill_value: Any,
        dtype: Any = None,
        device: Any = None,
    ):
        raise NotImplementedError(f"{self.__class__.__name__} must implement full_like_()")

    def clone(self) -> "AbstractTensor":
        finalize = AbstractTensor._pre_autograd("clone", [self])
        result = AbstractTensor.get_tensor(like=self)
        result.data = self.clone_()
        if getattr(self, "requires_grad", False):
            try:
                result.requires_grad_(True)
            except Exception:
                result._requires_grad = True
        return finalize(result)

    # copy is an alias of clone for API compatibility
    def copy(self) -> "AbstractTensor":
        return self.clone()

    def to_device(self, device: Any = None) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.to_device_(device)
        return result

    def get_device(self) -> Any:
        return self.get_device_()

    def get_dtype(self) -> Any:
        return self.get_dtype_()


    # --- Properties and methods from abstraction_methods/properties.py ---
    numel = _properties.numel
    item = _properties.item
    shape = _properties.shape
    shape_ = _properties.shape_
    ndim = _properties.ndim
    dim = _properties.dim
    ndims = _properties.ndims
    datastring = _properties.datastring
    __str__ = _properties.__str__
    __repr__ = _properties.__repr__
    __len__ = _properties.__len__

    @staticmethod
    def check_or_build_registry():
        cls = None
        if not BACKEND_REGISTRY:
            try:
                from . import numpy_backend  # noqa: F401
            except Exception:
                pass
            try:
                from . import torch_backend  # noqa: F401
            except Exception:
                pass
            try:
                from . import pure_backend  # noqa: F401
            except Exception:
                pass

        for backend_name in ("numpy", "torch", "pure_python"):
            backend_cls = BACKEND_REGISTRY.get(backend_name)
            if backend_cls is not None:
                cls = backend_cls
                break
        if cls is None:
            raise RuntimeError("No tensor backend available for tensor creation.")
        return cls

    @classmethod
    def tensor(
        cls,
        data=None,
        *,
        dtype=None,
        device=None,
        track_time: bool = False,
        faculty: "Faculty" = None,
        requires_grad: bool = False,
        tape: "GradTape" | None = None,
    ) -> "AbstractTensor":
        """
        Create an AbstractTensor from `data`.

        - If called on AbstractTensor: auto-select backend via get_tensor(...).
        - If called on a backend subclass: use that backend directly.
        - dtype/device are applied best-effort after wrapping.
        """
        if faculty is not None:
            print("Faculty is depreciated and unused")
        if cls is AbstractTensor:
            cls = cls.check_or_build_registry()

        # Use the specific backend class
        inst = cls(track_time=track_time, requires_grad=requires_grad, tape=tape)
        if data is None:
            # Initialize a concrete empty tensor on the selected backend so the
            # object has a valid `.data` payload. This avoids leaking
            # uninitialized "backend handles" into code paths that expect real
            # tensors (numel/item/dtype/device).
            try:  # backends implement empty_(shape, dtype, device)
                empty_buf = inst.empty_((0,), dtype=dtype, device=device)
                try:
                    inst.data = empty_buf  # type: ignore[attr-defined]
                except Exception:
                    pass
            except Exception:
                # If backend lacks empty_ or assignment fails, fall back to
                # returning the instance without data; callers must avoid
                # truthiness/numel/item on it.
                pass
            return inst

        out = inst.ensure_tensor(data)
        if dtype is not None:
            try:
                out = out.to_dtype(dtype)
            except Exception:
                pass
        if device is not None:
            try:
                out = out.to_device(device)
            except Exception:
                pass
        return out

    @classmethod
    def from_nested(cls, data, *, dtype=None, device=None):
        """
        Recursively pack arbitrarily nested sequences of leaves (scalars, numpy/torch tensors,
        AbstractTensor instances) into a single AbstractTensor by stacking bottom-up.

        This is the safe replacement for passing a nested list to .tensor(...).
        """
        from .nested_pack import pack_nested_to_tensor

        return pack_nested_to_tensor(data, dtype=dtype, device=device, cls=cls)

    @staticmethod
    def get_tensor(
        data=None,
        *,
        dtype=None,
        device=None,
        cls=None,
        like=None,
        track_time=False,
        tape: "GradTape" | None = None,
        requires_grad: bool = False,
    ) -> "AbstractTensor":
        """
        Get the tensor data from this AbstractTensor or create a new one if data is provided.
        If data is None, return self.
        """
        if cls is None:
            if like is not None:
                cls = like.__class__
            else:
                cls = AbstractTensor.check_or_build_registry()

        # Inherit tape/gradient intent from ``like`` but do not override an
        # explicit requires_grad=True request.
        if like is not None:
            if tape is None:
                tape = getattr(like, "_tape", None)
            if not requires_grad and getattr(like, "requires_grad", False):
                requires_grad = True

        # If gradients are requested and no tape has been supplied yet, fall
        # back to the global tape when available.
        if requires_grad and tape is None:
            try:
                from . import autograd as _autograd  # local import to avoid cycles
                tape = _autograd.autograd.tape
            except Exception:
                tape = None
        pool = _get_tensor_pool(cls)
        if data is not None and pool is not None and np is not None:
            if not isinstance(data, AbstractTensor):
                try:
                    arr = np.asarray(data, dtype=dtype if dtype is not None else None)
                except Exception:
                    try:
                        pool.observe(np.shape(data), dtype=dtype, device=device)
                    except Exception:
                        pass
                else:
                    shape = arr.shape
                    try:
                        buf = pool.acquire(shape, dtype=arr.dtype, device=device)
                        try:
                            buf.data[...] = arr
                        except Exception:
                            try:
                                buf.data = arr.tolist()
                            except Exception:
                                buf.data = arr
                        buf.track_time = track_time
                        if requires_grad:
                            buf._tape = tape
                            try:
                                buf.requires_grad_(True)
                            except Exception:
                                buf._requires_grad = True  # type: ignore[attr-defined]
                        return buf
                    except Exception:
                        try:
                            pool.observe(shape, dtype=arr.dtype, device=device)
                        except Exception:
                            pass
            else:
                try:
                    shape = getattr(data, "shape", ())
                    dtype_key = getattr(getattr(data, "data", data), "dtype", None)
                    pool.observe(shape, dtype=dtype_key, device=device)
                except Exception:
                    pass

        t = cls.tensor(
            data,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            track_time=track_time,
            tape=tape,
        )
        return t

    def tensor_like(self, data=None, *, dtype=None, device=None, cls=None) -> "AbstractTensor":
        """
        Get the tensor data from this AbstractTensor or create a new one if data is provided.
        If data is None, return self.
        """
        if cls is not None:
            return cls.tensor(data, dtype=dtype, device=device, track_time=self.track_time)
        return self.tensor(data, dtype=dtype, device=device, track_time=self.track_time)

    @staticmethod
    def range(start, end=None, step=1, *, dtype=None, device=None, cls=None, tape: "GradTape" | None = None):
        return AbstractTensor.arange(start, end, step, dtype=dtype, device=device, cls=cls, tape=tape)

    @staticmethod
    def arange(
        start,
        end=None,
        step=1,
        *,
        dtype=None,
        device=None,
        cls=None,
        tape: "GradTape" | None = None,
    ) -> "AbstractTensor":
        """
        Create an arange tensor using the best available backend if cls is None.

        Forms:
        arange(end)                -> [0, 1, ..., end-1]
        arange(start, end)         -> [start, ..., end-step]
        arange(start, end, step)   -> general form

        Notes:
        - step must be nonzero.
        - dtype/device are forwarded to the backend.
        """
        # Normalize one-argument form
        if end is None:
            start, end = 0, start

        if step == 0:
            raise ValueError("arange step must be nonzero")

        # Backend selection (attempt to register common backends)
        if cls is None:
            test_tensor = AbstractTensor.get_tensor(0)
            cls = type(test_tensor)

        inst = cls(track_time=False, tape=tape)  # Assuming default track_time
        inst.data = inst.arange_(start, end, step, dtype=dtype, device=device)
        return inst

    def select_by_indices(
        self, indices_dim0: Any = None, indices_dim1: Any = None
    ) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.select_by_indices_(indices_dim0, indices_dim1)
        return result

    def log_softmax(self, dim: int = -1) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.log_softmax_(dim)
        return result

    def pad(self, pad: Tuple[int, ...] = (0, 0), value: float = 0) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.pad_(pad, value)
        return result

    # --- 2D spatial helpers -------------------------------------------------
    def pad2d(self, pad: Tuple[int, int, int, int], value: float = 0.0) -> "AbstractTensor":
        """Pad a 4D NCHW tensor with constant values.

        Parameters
        ----------
        pad:
            Tuple ``(pad_left, pad_right, pad_top, pad_bottom)``.
        value:
            Constant fill value for padded regions.
        """
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.pad2d_(pad, value)
        return result

    def pad2d_(self, pad: Tuple[int, int, int, int], value: float = 0.0):
        """Backend hook for :meth:`pad2d`."""
        # Default implementation delegates to generic ``pad``.
        return self.pad_(pad, value)

    @staticmethod
    def concatenate(*args, **kwargs):
        print("Warning: AbstractTensor.concatenate is deprecated, use AbstractTensor.cat instead.")
        return AbstractTensor.cat(*args, **kwargs)

    @staticmethod
    def cat(tensors: List[Any], dim: int = 0) -> "AbstractTensor":
        """Concatenate ``tensors`` along dimension ``dim``.

        Accepts raw backend tensors or ``AbstractTensor`` instances and
        dispatches to the underlying backend implementation via ``cat_``.
        """
        if not tensors:
            raise ValueError("cat requires at least one tensor")
        first = AbstractTensor.get_tensor(tensors[0])
        tensors = [first.ensure_tensor(t) for t in tensors]
        result = first.__class__(track_time=first.track_time)
        result.data = first.cat_(tensors, dim)
        return result

    @staticmethod
    def pad_cat(
        tensors: List[Any], dim: int = 0, pad_value: float = 0
    ) -> "AbstractTensor":
        """Concatenate tensors, padding mismatched dimensions with ``pad_value``."""

        if not tensors:
            raise ValueError("pad_cat requires at least one tensor")
        first = AbstractTensor.get_tensor(tensors[0])
        tensors = [first.ensure_tensor(t) for t in tensors]
        result = first.__class__(track_time=first.track_time)
        result.data = first.pad_cat_(tensors, dim, pad_value)
        return result

    @staticmethod
    def outer(a: Any, b: Any) -> "AbstractTensor":
        """Return the outer product ``a ⊗ b`` as an :class:`AbstractTensor`.

        The operation flattens both inputs to 1D, adds singleton dimensions,
        and multiplies them using standard broadcasting.  This mirrors
        ``numpy.outer``/``torch.outer`` semantics while reusing existing
        reshape and multiply primitives so autograd can trace the computation
        without additional backend hooks.

        Parameters
        ----------
        a, b:
            Tensor-like objects.  If plain arrays are provided they will be
            converted to tensors using ``AbstractTensor.get_tensor`` and
            ``ensure_tensor`` so both arguments share the same backend and
            gradient tape.
        """

        at = AbstractTensor.get_tensor(a)
        bt = at.ensure_tensor(b)
        return at.flatten().unsqueeze(-1) * bt.flatten().unsqueeze(-2)

    class _TopKResult(NamedTuple):
        values: "AbstractTensor"
        indices: Any

    @staticmethod
    def topk(tensor: Any, k: int = 1, dim: int = -1) -> "AbstractTensor._TopKResult":
        """Return the top ``k`` elements and their indices along ``dim``.

        Mirrors ``torch.topk`` by returning a named tuple with ``values`` and
        ``indices`` fields.
        """
        tensor = AbstractTensor.get_tensor(tensor)
        values, idxs = tensor.topk_(k, dim)
        result = tensor.__class__(track_time=tensor.track_time)
        result.data = values
        return AbstractTensor._TopKResult(result, idxs)

    def _prefix_broadcastable(self, shapes, dim: int):
        """
        Raise ValueError if any pair of prefixes ([:dim]) in `shapes`
        is not broadcastable (equal or 1).
        """
        if dim <= 0 or not shapes:
            return
        # Left-pad each prefix with 1s to the same length
        maxlen = max(len(s) for s in shapes)
        pads = []
        for s in shapes:
            pref = s[:dim] if len(s) >= dim else s
            # left-pad to `dim` with 1s
            pref = (1,) * max(0, dim - len(pref)) + tuple(pref[:dim])
            pads.append(pref)
        # pairwise check
        ref = pads[0]
        for cur in pads[1:]:
            for a, b in zip(ref, cur):
                if (a != b) and (a != 1) and (b != 1):
                    raise ValueError(
                        f"Prefix dims before axis {dim} are not broadcastable: {ref} vs {cur}"
                    )

    def gather_and(
        self,
        dim: int,
        indices,
        fn_specs,
        param_tensor,
        *,
        other=None,
    ):
        """
        Gather from `base` (=other or self) along `dim` with `indices`,
        then run a function chain with per-fn params sliced from `param_tensor`.

        Call Signatures
        ---------------
        ``gather_and(dim, indices, fn_specs, param_tensor, *, other=None)``
            Standard form with ``dim`` as the first positional argument.
        ``gather_and(indices, fn_specs, param_tensor, *, dim, other=None)``
            ``dim`` may also be supplied via keyword, in which case the other
            arguments remain positional.

        Parameters
        ----------
        dim : int
            Axis to gather along.
        indices : tensor-like (int)
            Indices for gather.
        fn_specs : Sequence[Tuple[Callable, Any]]
            List of (fn, slice_spec). Each `fn` is called as `fn(y, p_slice)`.
        param_tensor : tensor-like
            Single tensor holding all parameters; slices are drawn from this.
        other : tensor-like, optional
            If provided, `other` is the gather source; otherwise `self`.

        Returns
        -------
        Any
            Output of the final function in the chain.
        """
        base  = self.ensure_tensor(other) if other is not None else self
        idx   = base.ensure_tensor(indices)
        p_all = base.ensure_tensor(param_tensor)

        # --- prefix broadcastability: shapes before `dim` must be equal or 1
        bshape   = tuple(getattr(base, "shape", ()))
        ishape   = tuple(getattr(idx,  "shape", ()))
        pshape   = tuple(getattr(p_all, "shape", ()))
        # Note: indices participates in the forward shape too; include it here
        self._prefix_broadcastable([bshape, ishape, pshape], dim)

        # --- gather
        y = base.gather(idx, dim)
        axis = dim if dim >= 0 else len(bshape) + dim
        axis_after = axis + len(ishape) - 1

        # --- apply function chain
        for spec in fn_specs or ():
            if not isinstance(spec, (tuple, list)) or len(spec) < 2:
                raise TypeError("fn_specs must be (fn, slice_spec) pairs")
            fn, sl = spec[0], spec[1]
            p_slice = p_all[sl]  # autograd-friendly slicing
            if getattr(p_slice, "ndim", 0) < getattr(y, "ndim", 0):
                lead = axis_after - (p_slice.ndim - 1)
                trail = y.ndim - lead - p_slice.ndim
                p_slice = p_slice.reshape(
                    *(1,) * max(0, lead),
                    *p_slice.shape,
                    *(1,) * max(0, trail),
                )
            y = fn(y, p_slice)
        return y


    def scatter_and(
        self,
        dim: int,
        index,
        src,
        fn_specs,
        param_tensor,
        *,
        other=None,
    ):
        """
        Scatter `src` into `base` (=other or self) along `dim` with `index`,
        then run a function chain with per-fn params sliced from `param_tensor`.

        Parameters
        ----------
        dim : int
            Axis to scatter along.
        index : tensor-like (int)
            Destinations for scatter (same shape as `src` along `dim`).
        src : tensor-like
            Values to scatter.
        fn_specs : Sequence[Tuple[Callable, Any]]
            List of (fn, slice_spec). Each `fn` is called as `fn(y, p_slice)`.
        param_tensor : tensor-like
            Single tensor holding all parameters; slices are drawn from this.
        other : tensor-like, optional
            If provided, `other` is the scatter destination; otherwise `self`.

        Returns
        -------
        Any
            Output of the final function in the chain.
        """
        base  = self.ensure_tensor(other) if other is not None else self
        idx   = base.ensure_tensor(index)
        val   = base.ensure_tensor(src)
        p_all = base.ensure_tensor(param_tensor)

        # --- prefix broadcastability
        bshape = tuple(getattr(base, "shape", ()))
        ishape = tuple(getattr(idx,  "shape", ()))
        vshape = tuple(getattr(val,  "shape", ()))
        pshape = tuple(getattr(p_all, "shape", ()))
        self._prefix_broadcastable([bshape, ishape, vshape, pshape], dim)

        # --- scatter
        y = base.scatter(idx, val, dim)

        # --- apply function chain
        for spec in fn_specs or ():
            if not isinstance(spec, (tuple, list)) or len(spec) < 2:
                raise TypeError("fn_specs must be (fn, slice_spec) pairs")
            fn, sl = spec[0], spec[1]
            p_slice = p_all[sl]
            y = fn(y, p_slice)
        return y


    @staticmethod
    def stack(tensors: List[Any], dim: int = 0) -> "AbstractTensor":
        """Stack ``tensors`` along a new dimension ``dim`` with autograd support.

        - Records a "stack" node so gradients are properly distributed back to
          each input via the registered backward rule (unstack along ``dim``).
        - No explicit ``requires_grad`` flag: the result requires grad when any
          input does; standard autograd semantics.
        """
        if not tensors:
            raise ValueError("stack requires at least one tensor")
        first = AbstractTensor.get_tensor(tensors[0])
        ts = [first.ensure_tensor(t) for t in tensors]
        out = first.__class__(track_time=first.track_time, tape=getattr(first, "_tape", None))
        out.data = first.stack_(ts, dim)
        finalize = AbstractTensor._pre_autograd("stack", ts, params={"dim": dim})
        return finalize(out)

    @staticmethod
    def unstack(tensor: Any, dim: int = 0) -> tuple:
        """Split ``tensor`` into a tuple of slices along dimension ``dim``.

        Utility used by backward rules; does not record autograd nodes.
        """
        t = AbstractTensor.get_tensor(tensor)
        shape = t.shape
        if dim < 0:
            dim = len(shape) + dim
        n = shape[dim]
        parts = []
        for i in range(n):
            idx = [slice(None)] * len(shape)
            idx[dim] = i
            parts.append(t[tuple(idx)])
        return tuple(parts)

    @staticmethod
    def split(tensor: Any, sizes: List[int], dim: int = 0) -> tuple:
        """Split ``tensor`` into chunks with ``sizes`` along ``dim``.

        Utility used by backward rules; does not record autograd nodes.
        """
        t = AbstractTensor.get_tensor(tensor)
        shape = t.shape
        if dim < 0:
            dim = len(shape) + dim
        parts = []
        offset = 0
        for s in sizes:
            idx = [slice(None)] * len(shape)
            idx[dim] = slice(offset, offset + s)
            parts.append(t[tuple(idx)])
            offset += s
        return tuple(parts)

    @staticmethod
    def copyto(
        dst: Any,
        src: Any,
        *,
        where: Any | None = None,
        casting: str = "same_kind",
    ) -> "AbstractTensor":
        """Copy ``src`` into ``dst`` with NumPy ``copyto`` semantics."""
        dst_tensor = dst if isinstance(dst, AbstractTensor) else AbstractTensor.get_tensor(dst)
        src_tensor = dst_tensor.ensure_tensor(src)
        where_tensor = dst_tensor.ensure_tensor(where) if where is not None else None
        dst_tensor.data = dst_tensor.copyto_(
            src_tensor, where=where_tensor, casting=casting
        )
        return dst_tensor

    @staticmethod
    def diag(tensor: Any, offset: int = 0) -> "AbstractTensor":
        """Extract or construct a diagonal.

        Mirrors ``numpy.diag`` behaviour. If ``tensor`` is 1‑D, a square
        matrix is returned with ``tensor`` on the ``offset`` diagonal. If
        ``tensor`` is 2‑D, the specified diagonal is extracted.

        Args:
            tensor: Input array or ``AbstractTensor``.
            offset: Which diagonal to consider. ``0`` selects the main
                diagonal, positive values move up, negative move down.

        Returns:
            ``AbstractTensor`` holding the resulting diagonal data.
        """
        t = AbstractTensor.get_tensor(tensor)
        result = t.__class__(track_time=t.track_time)
        result.data = t.diag_(offset)
        return result


    # --- Broadcasting helpers (abstract, backend-agnostic) ---
    def expand(self, *shape) -> "AbstractTensor":
        """Backend-agnostic expand/broadcast_to (view when possible).

        Accepts either a single iterable ``shape`` or multiple integer
        dimensions, mirroring ``torch.Tensor.expand``.  All provided values are
        collapsed into a single ``tuple`` before delegating to
        :meth:`expand_`.

        Also records an autograd node (as "broadcast_to") so gradients flow
        back through broadcasted views via the standard unbroadcast rule.
        """

        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        else:
            shape = tuple(shape)

        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.expand_(shape)
        # Record on the tape so that grads flow back to the source tensor.
        # Pass the target shape as a parameter for the backward rule signature (g, x, shape).
        finalize = AbstractTensor._pre_autograd("broadcast_to", [self], params={"shape": shape})
        return finalize(result)
    # in AbstractTensor (abstraction.py)
    def broadcast_rows(self, n: int, *, label: str | None = None) -> "AbstractTensor":
        """
        Make a (1, D) bias behave as (n, D) via view-style expand when possible.
        If already (n, D) return self.
        If neither 1 nor n rows, raise a labeled TensorShapeError.
        """
        rows, cols = self.shape[0], self.shape[1]
        if rows == n:
            return self
        if rows == 1:
            try:
                return self.expand((n, cols))
            except Exception:
                # backend lacks true view-expand; fall back to repeat
                return self.repeat_interleave(repeats=n, dim=0)

        # Build an iff-applicable hint (only when it's the classic "multi-row bias" mistake)
        hint = None
        if rows > 1 and cols >= 1:
            # This really looks like an accidental (N, D) bias; propose a safe collapse.
            hint = (
                "Bias must be a single row broadcast across the batch, or match the batch exactly. "
                "If you accidentally created an (N, D) bias, collapse it, e.g.: "
                "bias = bias.sum(dim=0, keepdim=True)  # -> (1, D)"
            )

        raise TensorShapeError(
            "Broadcast error",
            _Diag(
                op="broadcast_rows",
                tensor=label,
                expected=f"(1, {cols}) or ({n}, {cols})",
                actual=f"({rows}, {cols})",
                batch_size=n,
                hint=hint,  # only shown in 'auto' if we actually built one
            ),
        )



    # Backend hooks -----------------------------------------------------------
    def expand_(self, shape):  # pragma: no cover - backend required
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement expand_()"
        )

    def repeat_interleave_(
        self, repeats: int = 1, dim: Optional[int] = None
    ):  # pragma: no cover - backend required
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement repeat_interleave_()"
        )

    def copyto_(
        self,
        src: Any,
        *,
        where: Any | None = None,
        casting: str = "same_kind",
    ):  # pragma: no cover - backend required
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement copyto_()"
        )

    def view_flat(self) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.view_flat_()
        return result

    def assign_at_indices(
        self,
        indices_dim0: Any = None,
        indices_dim1: Any = None,
        values_to_assign: Any = None,
    ) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.assign_at_indices_(
            indices_dim0, indices_dim1, values_to_assign
        )
        return result

    def increment_at_indices(self, mask: Any = None) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.increment_at_indices_(mask)
        return result

    # Only keep the upper, guarded to_backend (already present above)

    

    def boolean_mask_select(self, mask: Any = None) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.boolean_mask_select_(mask)
        return result

    def tobytes(self) -> bytes:
        """Serialize tensor data to bytes by serializing each item in the list."""
        items = self.tolist()
        # If items is a list of tensors, recursively call tobytes; otherwise, bytes(items)
        if hasattr(items, '__iter__') and not isinstance(items, (bytes, bytearray)):
            # Flatten the list and convert each item to bytes
            flat = []
            def _flatten(x):
                if isinstance(x, (list, tuple)):
                    for y in x:
                        _flatten(y)
                else:
                    flat.append(x)
            _flatten(items)
            # Try to convert each item to bytes, if possible
            result = b''
            for v in flat:
                if hasattr(v, 'tobytes'):
                    result += v.tobytes()
                elif isinstance(v, (int, float, bool)):
                    import struct
                    # Use double for float, long long for int, ? for bool
                    if isinstance(v, float):
                        result += struct.pack('d', v)
                    elif isinstance(v, int):
                        result += struct.pack('q', v)
                    elif isinstance(v, bool):
                        result += struct.pack('?', v)
                elif isinstance(v, bytes):
                    result += v
                else:
                    # fallback: try str encoding
                    result += str(v).encode('utf-8')
            return result
        else:
            # Not a list, just try to convert directly
            if hasattr(items, 'tobytes'):
                return items.tobytes()
            elif isinstance(items, (int, float, bool)):
                import struct
                if isinstance(items, float):
                    return struct.pack('d', items)
                elif isinstance(items, int):
                    return struct.pack('q', items)
                elif isinstance(items, bool):
                    return struct.pack('?', items)
            elif isinstance(items, bytes):
                return items
            else:
                return str(items).encode('utf-8')

    def tolist(self) -> List[Any]:
        return self.tolist_()

    def nbytes(self) -> int:
        """Return the number of bytes consumed by the tensor's data buffer.

        Delegates to the backend hook ``nbytes_``. This is a read-only query
        and does not participate in autograd.
        """
        return int(self.nbytes_())

    
    def pi(self=None) -> float:
        """Return pi as a numeric scalar.

        Works when called as ``AbstractTensor.pi()`` or on a tensor instance.
        Prefers a backend-provided constant ``_pi`` when available; otherwise
        returns a high-precision float fallback.
        """
        # Allow class-call without an instance by delegating to a default backend
        if self is None:
            base = AbstractTensor.get_tensor(0)
            return base.pi()
        # Try backend-provided constant (may be a property returning a float)
        try:
            if hasattr(self, "_pi"):
                val = getattr(self, "_pi")
                # If it's a property/value, return it; if it's a callable, call it
                return AbstractTensor.get_tensor(val() if callable(val) else val)
        except Exception:
            pass
        # Fallback: numeric constant
        return AbstractTensor.get_tensor(3.14159265358979323846264338327950288419716939937510)

    def long_pi(self=None):
        if self is None:
            base = AbstractTensor.get_tensor(0)
            return base.long_pi()
        # Pi to 50 decimal places
        return AbstractTensor.tensor(
            3.14159265358979323846264338327950288419716939937510,
            dtype=self.float_dtype, device=self.get_device()
        )

    def index_select(self, dim: int = 0, indices: Any = None) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.index_select_(dim, indices)
        return result

    def argmin(self, dim: Optional[int] = None) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.argmin_(dim)
        return result

    def interpolate(self, size: Tuple[int, ...] = None) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.interpolate_(size)
        return result

    def save(self, filepath: str = None) -> None:
        self.save_(filepath)

    def load(
        self, filepath: str, dtype: Any = None, device: Any = None
    ) -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.load_(filepath, dtype, device)
        return result

    def to_dtype(self, dtype: str = "float") -> "AbstractTensor":
        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self.to_dtype_(dtype)
        return result

    # --- Pure-abstraction utilities ---------------------------------------
    @staticmethod
    def searchsorted(a: Any, v: Any, side: str = "left") -> "AbstractTensor":
        """Return insertion indices for ``v`` in sorted 1-D sequence ``a``.

        Implemented in terms of existing elementwise comparisons and reductions
        to preserve autograd tape semantics without backend-specific code.

        - side='left': count of elements strictly less than v
        - side='right': count of elements less than or equal to v
        """
        seq = AbstractTensor.get_tensor(a)
        vals = seq.ensure_tensor(v)

        # Flatten for a simple (N, M) broadcasted comparison, then reshape back
        seq1 = seq.view_flat()
        vals1 = vals.view_flat()

        # Build broadcasted compare matrix: (N, 1) vs (1, M) -> (N, M)
        A = seq1.unsqueeze(1)
        B = vals1.unsqueeze(0)
        if side == "left":
            cmp = A < B
        elif side == "right":
            cmp = A <= B
        else:
            raise ValueError("side must be 'left' or 'right'")

        # Sum boolean comparisons along the sequence axis to get insertion idx
        counts = cmp.to_dtype(seq.long_dtype).sum(dim=0)
        # Restore original vals shape
        try:
            vshape = vals.get_shape()
            counts = counts.reshape(vshape)
        except Exception:
            pass

        finalize = AbstractTensor._pre_autograd("searchsorted", [seq, vals], params={"side": side})
        return finalize(counts)
    @staticmethod
    def concat(*args, **kwargs):
        return AbstractTensor.cat(*args, **kwargs)
    # --- Order statistics --------------------------------------------------
    def percentile(
        self,
        n: float,
        dim: int | None = None,
        *,
        which: str | None = None,
        group: str | None = None,
        inclusive: bool = True,
        e: float | None = None,
        return_mask: bool = False,
    ):
        """Percentile utilities: values and percentile-defined selection.

        Value mode (when ``group is None``):
        - When ``dim is None`` returns a Python ``float`` (global percentile).
        - When ``dim`` is provided, returns an ``AbstractTensor`` reduced over
          that dimension.
        - ``which`` chooses the return relative to the target percentile:
          "value" (default), "under" (lower bracket), "over" (upper bracket),
          or "margin" (over-under).

        Selection mode (when ``group`` provided):
        - Returns a boolean mask (``return_mask=True``) or selected elements
          (``False``, default) corresponding to a percentile-defined slice.
        - ``group`` one of: "under"/"0-n" (<= n%), "over"/"n-100" (>= n%),
          "band"/"margin" (between n-e and n+e percent, requires ``e``).
        - ``inclusive`` controls boundary inclusion.

        Implementation is backend-agnostic and composed from existing
        high-level tensor operations in this module:
        - ``flatten`` to 1D
        - ``min``/``max`` fast paths for 0%/100%
        - ``topk`` + ``min`` on the negated array to compute order statistics
        - linear interpolation between adjacent order statistics

        Notes:
        - For empty inputs, returns ``float('nan')`` (or a nan tensor/mask of
          all False in selection mode).
        - For single-element inputs, returns that element (or a mask selecting it).
        - Interpolation uses NumPy's traditional "linear" scheme with
          rank = p/100 * (N-1).
        """
        # Parse percentile and behaviour flags
        try:
            p = float(n)
        except Exception:
            raise TypeError("percentile requires a numeric percentile value")
        which = (which or "value").lower()
        if which not in ("value", "under", "over", "margin"):
            raise ValueError("which must be one of: 'value', 'under', 'over', 'margin'")

        # If selection mode requested, normalise group alias and branch later
        if group is not None:
            g = (group or "under").lower()
            if g in ("0-n", "below"): g = "under"
            if g in ("n-100", "above"): g = "over"
            if g in ("window", "margin"): g = "band"
            if g not in ("under", "over", "band"):
                raise ValueError("group must be one of: 'under', 'over', 'band'")
        else:
            g = None

        # Axis/global setup
        if dim is None and g is None:
            x = self.flatten()
            try:
                N = int(x.numel())
            except Exception:
                N = 0
            if N == 0:
                # Global empty -> NaN float
                return float("nan")
            if N == 1:
                return float(x.item())

            if p <= 0.0:
                return float(x.min().item())
            if p >= 100.0:
                return float(x.max().item())

            rank = (p / 100.0) * (N - 1)
            lo_i = int(math.floor(rank))
            hi_i = int(math.ceil(rank))
            alpha = rank - lo_i

            def kth_smallest_1d(vec: "AbstractTensor", k: int) -> "AbstractTensor":
                y = -vec
                y_topk = AbstractTensor.topk(y, k=k, dim=0).values  # (k,)
                y_k = y_topk.min()  # scalar
                return -y_k

            v_lo = kth_smallest_1d(x, lo_i + 1)
            v_hi = kth_smallest_1d(x, hi_i + 1)

            if which == "under":
                return float(v_lo.item())
            if which == "over":
                return float(v_hi.item())
            if which == "margin":
                return float((v_hi - v_lo).item())

            if lo_i == hi_i:
                return float(v_lo.item())
            out = (1.0 - alpha) * v_lo + alpha * v_hi
            return float(out.item())

        # Dimension-aware value mode, or any selection mode
        x = self
        nd = int(x.ndim) if hasattr(x, "ndim") else len(getattr(x, "shape", ()))
        axis = dim if dim >= 0 else (nd + dim)

        try:
            shape = x.shape
            N = int(shape[axis])
        except Exception:
            # Fallback: attempt to reduce over dim; if fails, treat as empty
            N = 0

        if g is None:
            # Value mode along axis
            if N == 0:
                out_shape = tuple(shape[i] for i in range(len(shape)) if i != axis) if isinstance(shape, tuple) else ()
                return AbstractTensor.full(out_shape, float("nan"), dtype=float)
            if p <= 0.0:
                return x.min(dim=axis)
            if p >= 100.0:
                return x.max(dim=axis)

            rank = (p / 100.0) * (N - 1)
            lo_i = int(math.floor(rank))
            hi_i = int(math.ceil(rank))
            alpha = rank - lo_i

            def kth_smallest_along(vec: "AbstractTensor", k: int, d: int) -> "AbstractTensor":
                y = -vec
                y_topk = AbstractTensor.topk(y, k=k, dim=d).values  # replace axis d by k
                y_k = y_topk.min(dim=d)  # reduce the k-axis
                return -y_k

            v_lo = kth_smallest_along(x, lo_i + 1, axis)
            v_hi = kth_smallest_along(x, hi_i + 1, axis)

            if which == "under":
                return v_lo
            if which == "over":
                return v_hi
            if which == "margin":
                return v_hi - v_lo

            if lo_i == hi_i:
                return v_lo
            return (1.0 - alpha) * v_lo + alpha * v_hi

        # Selection mode (group present): build mask then optionally compact
        # Compute thresholds per group
        if g == "band":
            if e is None:
                raise ValueError("percentile(group='band') requires e (half-width)")
            lo_p = max(0.0, p - float(e))
            hi_p = min(100.0, p + float(e))
            t_lo = self.percentile(lo_p, dim=dim)
            t_hi = self.percentile(hi_p, dim=dim)
        else:
            t = self.percentile(p, dim=dim)

        # Build mask with broadcasting over axis
        if dim is None:
            if g == "under":
                mask = self.less_equal(t) if inclusive else self.less(t)
            elif g == "over":
                mask = self.greater_equal(t) if inclusive else self.greater(t)
            else:  # band
                m1 = self.greater_equal(t_lo) if inclusive else self.greater(t_lo)
                m2 = self.less_equal(t_hi) if inclusive else self.less(t_hi)
                mask = (m1.to_dtype(self.long_dtype) * m2.to_dtype(self.long_dtype)).to_dtype(self.bool_dtype)
        else:
            if g == "band":
                t_lo = t_lo.unsqueeze(axis)
                t_hi = t_hi.unsqueeze(axis)
                m1 = self.greater_equal(t_lo) if inclusive else self.greater(t_lo)
                m2 = self.less_equal(t_hi) if inclusive else self.less(t_hi)
                mask = (m1.to_dtype(self.long_dtype) * m2.to_dtype(self.long_dtype)).to_dtype(self.bool_dtype)
            else:
                t = t.unsqueeze(axis)
                if g == "under":
                    mask = self.less_equal(t) if inclusive else self.less(t)
                else:
                    mask = self.greater_equal(t) if inclusive else self.greater(t)

        return mask if return_mask else self.boolean_mask_select(mask)

    # --- Dtype helpers ---
    @property
    def long_dtype(self) -> Any:
        return self.long_dtype_

    @property
    def bool_dtype(self) -> Any:
        return self.bool_dtype_

    @property
    def float_dtype(self) -> Any:
        return self.float_dtype_

    @property
    def tensor_type(self) -> type:
        return self.tensor_type_

    # Lightweight helper to coerce arbitrary input to this backend's tensor type


    def to_backend(self, target_ops):
        conv_func = CONVERSION_REGISTRY.get((type(self), type(target_ops)))
        if conv_func is None:
            converted = default_to_backend(self, self, target_ops)
        else:
            converted = conv_func(self, self, target_ops)

        if isinstance(converted, AbstractTensor):
            if type(converted) is type(target_ops):
                return converted
            # No progress? Inject raw data into the target and bail.
            if type(converted) is type(self):
                out = type(target_ops)(track_time=self.track_time, tape=getattr(target_ops, "_tape", None))
                out.data = converted.data
                return out
            return converted.to_backend(target_ops)

        new_tensor = type(target_ops)(track_time=self.track_time, tape=getattr(target_ops, "_tape", None))
        new_tensor.data = converted
        return new_tensor

    def ensure_tensor(self, tensor: Any) -> "AbstractTensor":
        """Return ``tensor`` wrapped as an ``AbstractTensor`` instance."""
        if not isinstance(self, AbstractTensor):
            raise TypeError(f"ensure_tensor called on non-AbstractTensor instance: {type(self)}")
        if tensor is None:
            raise ValueError("ensure_tensor called with tensor=None")
        backend_cls = self.__class__
        if isinstance(tensor, AbstractTensor):
            return tensor.to_backend(self)
        if isinstance(tensor, self.tensor_type):
            result = backend_cls(track_time=self.track_time, tape=getattr(self, "_tape", None))
            result.data = tensor
            return result
        if torch is not None and isinstance(tensor, torch.Tensor):
            try:
                from .torch_backend import PyTorchTensorOperations
                torch_ops = AbstractTensor.get_tensor(cls=PyTorchTensorOperations, tape=getattr(self, "_tape", None))
                tmp = torch_ops.__class__(track_time=self.track_time, tape=getattr(self, "_tape", None))
                tmp.data = tensor
                return tmp.to_backend(self)
            except Exception:
                pass
        if np is not None and isinstance(tensor, np.ndarray):
            from .numpy_backend import NumPyTensorOperations
            numpy_ops = AbstractTensor.get_tensor(cls=NumPyTensorOperations, tape=getattr(self, "_tape", None))
            numpy_tensor = numpy_ops.__class__(track_time=self.track_time, tape=getattr(self, "_tape", None))
            numpy_tensor.data = tensor
            return numpy_tensor.to_backend(self)
        if isinstance(tensor, (list, tuple)):
            # Mixed or nested sequences are routed through the nested packer
            if any(isinstance(elem, (list, tuple, AbstractTensor)) for elem in tensor):
                # Ensure nested creation attaches to this tensor's tape
                from . import autograd as _autograd
                prev = _autograd.autograd.tape
                try:
                    if getattr(self, "_tape", None) is not None:
                        _autograd.autograd.tape = getattr(self, "_tape", None)
                    return self.__class__.from_nested(tensor)
                finally:
                    _autograd.autograd.tape = prev
            try:
                from . import autograd as _autograd
                prev = _autograd.autograd.tape
                try:
                    if getattr(self, "_tape", None) is not None:
                        _autograd.autograd.tape = getattr(self, "_tape", None)
                    return type(self)._tensor_from_list(tensor, dtype=None, device=None, tape=getattr(self, "_tape", None))
                finally:
                    _autograd.autograd.tape = prev
            except Exception:
                # numpy/pure backends may choke on ragged lists; fall back to nested pack
                from . import autograd as _autograd
                prev = _autograd.autograd.tape
                try:
                    if getattr(self, "_tape", None) is not None:
                        _autograd.autograd.tape = getattr(self, "_tape", None)
                    return self.__class__.from_nested(tensor)
                finally:
                    _autograd.autograd.tape = prev
        if hasattr(tensor, "tolist"):
            return self.ensure_tensor(tensor.tolist())
        from . import autograd as _autograd
        prev = _autograd.autograd.tape
        try:
            if getattr(self, "_tape", None) is not None:
                _autograd.autograd.tape = getattr(self, "_tape", None)
            return type(self)._tensor_from_list([tensor], dtype=None, device=None, tape=getattr(self, "_tape", None))
        finally:
            _autograd.autograd.tape = prev

    # --- Operator routing ---
    @staticmethod
    def _pre_autograd(op: str, inputs: Iterable[Any], params: Dict[str, Any] | None = None):
        """Return a callback that records ``op`` on the autograd tape.

        Parameters
        ----------
        op:
            Name of the operator about to be executed.
        inputs:
            Sequence of operands participating in the operation.
        """
        from . import autograd as _autograd

        inputs = list(inputs)
        # Promote non-tensor operands to tensors so backward rules always see wrappers.
        first = next((t for t in inputs if isinstance(t, AbstractTensor)), None)
        if first is not None:
            inputs = [x if isinstance(x, AbstractTensor) else first.ensure_tensor(x) for x in inputs]

        tape = None
        for t in inputs:
            if isinstance(t, AbstractTensor):
                tape = getattr(t, "_tape", None)
                if tape is not None:
                    break
        if tape is None:
            tape = _autograd.autograd.tape

        requires = any(
            getattr(x, "requires_grad", False) for x in inputs if isinstance(x, AbstractTensor)
        )
        track = any(
            getattr(x, "track_time", False) for x in inputs if isinstance(x, AbstractTensor)
        )

        if requires or track or getattr(AbstractTensor.autograd, "capture_all", False):
            start = time.perf_counter() if track else None

            def finalize(result: Any):
                end = time.perf_counter() if track else None
                if requires:
                    try:
                        result._requires_grad = True  # type: ignore[attr-defined]
                    except Exception:
                        pass
                if getattr(AbstractTensor.autograd, "_no_grad_depth", 0) == 0:
                    AbstractTensor.autograd.record(
                        op, inputs, result, start=start, end=end, params=params
                    )
                return result

            return finalize

        return lambda result: result

    def matmul(self, other: AbstractTensor) -> AbstractTensor:
        """Matrix multiplication."""
        return self._apply_operator("matmul", self, other)

    def _apply_operator(self, op: str, left: Any, right: Any):
        """
        Arithmetic with bool tensors:
        - if mixing with floats/complex -> cast bool to float
        - if mixing with ints          -> cast bool to int (0/1)
        - for true division            -> cast bool to float
        Promotion happens BEFORE unwrap; backends never see bool arithmetic.
        """
        # Coerce list-like operands into tensors so that operator logic remains
        # backend-agnostic. This ensures raw Python lists interoperate with
        # tensors without requiring callers to explicitly convert them.
        # In AbstractTensor._apply_operator(...)
        if isinstance(left, AbstractTensor) and isinstance(right, (list, tuple)):
            right = left.ensure_tensor(right)      # instead of get_tensor(... Faculty.NUMPY)
        elif isinstance(right, AbstractTensor) and isinstance(left, (list, tuple)):
            left = right.ensure_tensor(left)       # instead of get_tensor(... Faculty.NUMPY)

        # Record first using the original operand wrappers so parameter identity
        # is preserved on the tape (ids match actual model params).
        finalize = AbstractTensor._pre_autograd(op, [x for x in (left, right) if x is not None])

        # Optional belt-and-suspenders: align mixed backends AFTER recording so
        # backend execution receives compatible operands without altering tape inputs.
        if isinstance(left, AbstractTensor) and isinstance(right, AbstractTensor) and (type(left) is not type(right)):
            try:
                right = right.to_backend(left)
            except Exception:
                try:
                    left = left.to_backend(right)
                except Exception:
                    pass

        arithmetic_ops = {
            "add","sub","mul","truediv","floordiv","mod","pow",
            "iadd","isub","imul","itruediv","ifloordiv","imod","ipow",
            "radd","rsub","rmul","rtruediv","rfloordiv","rmod","rpow",
            "neg","abs","invert",
        }
        div_ops = {"truediv","rtruediv","itruediv"}

        def kind(x):
            if isinstance(x, AbstractTensor):
                try:
                    dt = x.get_dtype()
                    if dt == x.bool_dtype: return "bool"
                    s = str(dt).lower()
                    if "complex" in s: return "complex"
                    if "float" in s or "half" in s or "bfloat" in s: return "float"
                    if "int" in s or "long" in s: return "int"
                except Exception:
                    return "unknown"
                return "unknown"
            if isinstance(x, bool):  return "bool"
            if isinstance(x, float): return "float"
            if isinstance(x, int):   return "int"
            return "unknown"

        def cast_bool_like(x, target_kind):
            if target_kind in ("float","complex"):
                return x.to_dtype("float")
            else:
                return x.long()

        if op in arithmetic_ops:
            lk, rk = kind(left), kind(right)
            if op in div_ops:
                if isinstance(left, AbstractTensor) and lk == "bool":
                    left = left.to_dtype("float")
                if isinstance(right, AbstractTensor) and rk == "bool":
                    right = right.to_dtype("float")
            else:
                target = "float" if ("float" in (lk, rk) or "complex" in (lk, rk)) else "int"
                if isinstance(left, AbstractTensor) and lk == "bool":
                    left = cast_bool_like(left, target)
                if isinstance(right, AbstractTensor) and rk == "bool":
                    right = cast_bool_like(right, target)

            # Handle raw Python bools symmetrically (e.g., 1 - True)
            if isinstance(left, bool):
                left = 1.0 if (op in div_ops or "float" in (rk,)) else 1
            if isinstance(right, bool):
                right = 1.0 if (op in div_ops or "float" in (lk,)) else 1

        if op == "matmul":
            lk, rk = kind(left), kind(right)
            if "float" in (lk, rk) and "int" in (lk, rk):
                if isinstance(left, AbstractTensor) and lk == "int":
                    left = left.to_dtype("float")
                if isinstance(right, AbstractTensor) and rk == "int":
                    right = right.to_dtype("float")

        if op == "matmul" and max(list(left.shape) + list(right.shape)) > 4096:
            from .batched_matmul import matmul_chunked
            return matmul_chunked(left, right, Kt=1024, Nt=512)

        # unwrap AFTER promotion
        l = left._AbstractTensor__unwrap() if isinstance(left, AbstractTensor) else left
        r = right._AbstractTensor__unwrap() if isinstance(right, AbstractTensor) else right

        result = type(self)(track_time=self.track_time, tape=getattr(self, "_tape", None))
        result.data = self._apply_operator__(op, l, r)
        return finalize(result)

    # inside AbstractTensor
    def __array__(self, dtype=None):
        """Return a ``numpy`` view of this tensor's data.

        The conversion only materialises when the current backend is not the
        NumPy implementation.  In that case we delegate to ``to_backend`` to
        obtain a NumPy-backed tensor and expose its underlying ``ndarray``.
        """

        from .numpy_backend import NumPyTensorOperations

        tensor = self
        if not isinstance(self, NumPyTensorOperations):
            tensor = self.to_backend(
                AbstractTensor.get_tensor(0, cls=NumPyTensorOperations)
            )

        arr = tensor.data
        return arr if dtype is None else arr.astype(dtype)

    @property
    def __array_interface__(self):
        return self.__array__().__array_interface__


    def __add__(self, other):
        return self._apply_operator("add", self, other)

    def __sub__(self, other):
        return self._apply_operator("sub", self, other)

    def __mul__(self, other):
        return self._apply_operator("mul", self, other)

    def __truediv__(self, other):
        return self._apply_operator("truediv", self, other)

    def __floordiv__(self, other):
        return self._apply_operator("floordiv", self, other)

    def __mod__(self, other):
        return self._apply_operator("mod", self, other)

    def __pow__(self, other):
        return self._apply_operator("pow", self, other)

    def __matmul__(self, other):
        return self._apply_operator("matmul", self, other)

    def __eq__(self, other):
        return self.equal(other)

    def __ne__(self, other):
        return self.not_equal(other)

    def __lt__(self, other):
        return self.less(other)

    def __le__(self, other):
        return self.less_equal(other)

    def __gt__(self, other):
        return self.greater(other)

    def __ge__(self, other):
        return self.greater_equal(other)

    # Reverse operators
    def __radd__(self, other):
        return self._apply_operator("radd", other, self)

    def __rsub__(self, other):
        return self._apply_operator("rsub", other, self)

    def __rmul__(self, other):
        return self._apply_operator("rmul", other, self)

    def __rtruediv__(self, other):
        return self._apply_operator("rtruediv", other, self)

    def __rfloordiv__(self, other):
        return self._apply_operator("rfloordiv", other, self)

    def __rmod__(self, other):
        return self._apply_operator("rmod", other, self)

    def __rpow__(self, other):
        return self._apply_operator("rpow", other, self)

    def __rmatmul__(self, other):
        return self._apply_operator("rmatmul", other, self)

    # In-place operators
    def __iadd__(self, other):
        return self._apply_operator("iadd", self, other)

    def __isub__(self, other):
        return self._apply_operator("isub", self, other)

    def __imul__(self, other):
        return self._apply_operator("imul", self, other)

    def __itruediv__(self, other):
        return self._apply_operator("itruediv", self, other)

    def __ifloordiv__(self, other):
        return self._apply_operator("ifloordiv", self, other)

    def __imod__(self, other):
        return self._apply_operator("imod", self, other)

    def __ipow__(self, other):
        return self._apply_operator("ipow", self, other)

    def __imatmul__(self, other):
        return self._apply_operator("imatmul", self, other)

    # --- Indexing helpers ---
    def __getitem__(self, idx):
        """Return an indexed view wrapped as an AbstractTensor when possible.

        Accepts either standard Python index types or another ``AbstractTensor``
        instance as ``idx``. When given an ``AbstractTensor`` the underlying
        value is extracted via ``__unwrap`` so all backends behave consistently
        for tensor-based indexing.
        """
        if DEBUG:
            print(f"__getitem__ called with idx={idx} on {self.__class__.__name__}")

        data = self.data
        if data is None:
            raise ValueError("__getitem__ called on empty tensor")

        # ---- unwrap any AbstractTensor indices ----
        if isinstance(idx, tuple):
            index = tuple(
                (item._AbstractTensor__unwrap() if isinstance(item, AbstractTensor) else item)
                for item in idx
            )
        elif isinstance(idx, AbstractTensor):
            index = idx._AbstractTensor__unwrap()
        else:
            index = idx
        if DEBUG:
            print(f"Unwrapped index: {index}")
            print(f"Data type: {type(data)}")
        logger.debug("getitem idx=%s tensor_id=%s", index, id(self))

        # ---- prefer backend-specific get_item_ if available ----
        # Try to locate the ops/backend object (name may vary in your codebase)

        finalize = AbstractTensor._pre_autograd(
            "slice", [self], params={"slices": index}
        )

        if hasattr(self, "get_item_"):
            result = self.get_item_(data, index)
        else:
            # generic fallback: direct indexing on the raw tensor/array
            result = data[index]

        # ---- wrap result if it's the native tensor type for this backend ----
        if isinstance(result, self.tensor_type):
            wrapped = type(self)(track_time=getattr(self, "track_time", False), tape=getattr(self, "_tape", None))
            wrapped.data = result
            return finalize(wrapped)
        return result



    def __setitem__(self, idx, value):
        """Assign to the underlying tensor using Python indexing.

        Like ``__getitem__``, ``idx`` may itself be an ``AbstractTensor``.  The
        raw value is extracted before performing the assignment so tensor-based
        indices work across all backends.
        """
        if DEBUG:
            print(
                f"__setitem__ called with idx={idx}, value={value} on {self.__class__.__name__}"
            )
        data = self.data
        if data is None:
            raise ValueError("__setitem__ called on empty tensor")
        if CTensor is not None and isinstance(data, CTensor):
            raise NotImplementedError("__setitem__ not implemented for CTensor backend")
        if isinstance(idx, tuple):
            index = tuple(
                item._AbstractTensor__unwrap() if isinstance(item, AbstractTensor) else item
                for item in idx
            )
        elif isinstance(idx, AbstractTensor):
            index = idx._AbstractTensor__unwrap()
        else:
            index = idx

        if getattr(AbstractTensor.autograd, "_no_grad_depth", 0) > 0:
            raw_value = value.data if isinstance(value, AbstractTensor) else value
            data[index] = raw_value
            logger.debug("setitem idx=%s tensor_id=%s", index, id(self))
            return

        finalize = AbstractTensor._pre_autograd("index_set", [self, value], params={"idx": index})

        raw_value = value.data if isinstance(value, AbstractTensor) else value
        data[index] = raw_value
        finalize(self)
        logger.debug("setitem idx=%s tensor_id=%s", index, id(self))

    def __bool__(self):
        try:
            n = int(self.numel())
        except Exception:
            return bool(self.item())
        if n != 1:
            raise ValueError("The truth value of a tensor with more than one element is ambiguous.")
        return bool(self.item())

    def numpy(self):
        """Convert the tensor to a NumPy array."""
        return np.array(self.data)

    @staticmethod
    def benchmark(
        fn: Callable, *args, repeat: int = 1, warmup: int = 0, **kwargs
    ) -> "TapeProfiler":
        """Run ``fn`` repeatedly and profile recorded operations.

        The callable is executed ``warmup`` times without recording to allow the
        runtime to stabilise.  A fresh :class:`GradTape` is then installed on the
        global autograd engine and ``fn`` is executed ``repeat`` additional
        times.  The returned :class:`TapeProfiler` exposes statistics for all
        operations that were recorded on the tape.
        """

        from . import autograd as _autograd

        for _ in range(max(warmup, 0)):
            fn(*args, **kwargs)

        tape = _autograd.GradTape()
        _autograd.autograd.tape = tape
        for _ in range(max(repeat, 0)):
            fn(*args, **kwargs)
        return _autograd.TapeProfiler(tape)

    def data_or(self, obj: Any = None) -> Any:
        """Return self.data if no argument is passed, otherwise return the argument unchanged."""
        if obj is None:
            return self.data
        return obj

    @abstractmethod
    def get_shape(self) -> Tuple[int, ...]:
        """Return the shape of ``self`` as a tuple."""
        pass

    @abstractmethod
    def get_ndims(self) -> int:
        """Return the number of dimensions of ``self``."""
        pass

    def repeat(self, repeats: Any = None, dim: int = 0) -> "AbstractTensor":
        """Repeat ``self`` along ``dim`` ``repeats`` times."""
        return self.repeat_(repeats, dim)




# --- Added backend hook methods/properties ---
    def reshape_(self, shape):
        raise NotImplementedError(f"{self.__class__.__name__} must implement reshape_()")

    def transpose_(self, dim0, dim1):
        raise NotImplementedError(f"{self.__class__.__name__} must implement transpose_()")

    def squeeze_(self, dim: int | None = None):
        raise NotImplementedError(f"{self.__class__.__name__} must implement squeeze_()")

    def unravel_index_(self, shape):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement unravel_index_()"
        )

    def nbytes_(self) -> int:
        """Backend hook for nbytes query."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement nbytes_()")


class AbstractScalar(AbstractTensor):
    """Marker mixin for zero-dimensional tensors produced by reductions."""

    def __new__(cls, tensor: "AbstractTensor"):
        if getattr(getattr(tensor, "data", None), "shape", ()) != ():
            raise ValueError("AbstractScalar requires a zero-dimensional tensor")
        if not isinstance(tensor, AbstractScalar):
            scalar_cls = type(
                f"{tensor.__class__.__name__}Scalar",
                (tensor.__class__, cls),
                {},
            )
            tensor.__class__ = scalar_cls
        return tensor


class AbstractF:
    """
    Functional API for advanced tensor operations (e.g., interpolation).
    Decides the best backend and dispatches accordingly.
    """

    @staticmethod
    def interpolate(
        tensor,
        size=None,
        scale_factor=None,
        mode="bilinear",
        batch_dim=0,
        channel_dim=1,
        backend: str = None,
        align_corners=False,
    postprocess: str = None,  # 'round', 'floor', 'ceil', or None
    **kwargs,
    ):
        """
        Interpolate a tensor to a new size using the best available backend.
        - tensor: AbstractTensor or raw data
        - size: tuple of ints (new spatial size)
        - scale_factor: float or tuple (optional)
        - mode: interpolation mode (e.g., 'bilinear', 'nearest')
        - batch_dim: which dim is batch (default 0)
        - channel_dim: which dim is channel (default 1)
        - backend: 'torch', 'numpy', etc. (optional, auto if None)
        - align_corners: passed to torch if used
        """
        # Convert to AbstractTensor if needed
        tensor = AbstractTensor.get_tensor(tensor)
        orig_dtype = None
        try:
            import torch  # type: ignore
        except Exception:  # torch may be unavailable
            torch = None  # type: ignore
        if hasattr(tensor, 'data'):
            arr_data = tensor.data
        else:
            arr_data = tensor
        if torch is not None and hasattr(arr_data, 'dtype'):
            orig_dtype = arr_data.dtype
        # Backend selection
        chosen = None
        if backend == "torch":
            chosen = "torch"
        elif backend == "numpy":
            chosen = "numpy"
        else:
            try:
                import torch
                import torch.nn.functional as F

                chosen = "torch"
            except ImportError:
                chosen = "numpy"
        if chosen == "torch":
            import torch
            import torch.nn.functional as F

            from .torch_backend import PyTorchTensorOperations
            arr = tensor.to_backend(
                AbstractTensor.get_tensor(cls=PyTorchTensorOperations)
            )
            data = arr.data if hasattr(arr, "data") else arr
            # Ensure shape is (N, C, H, W) or (N, 1, H, W)
            nd = data.dim() if hasattr(data, "dim") else len(data.shape)
            if nd == 2:
                data = data.unsqueeze(0).unsqueeze(0)
            elif nd == 3:
                # Assume (C, H, W) or (N, H, W)
                if batch_dim == 0:
                    data = data.unsqueeze(0)
                else:
                    data = data.unsqueeze(1)
            # Convert to float if needed (required by F.interpolate for most modes)
            was_int = not torch.is_floating_point(data)
            if was_int:
                data = data.float()
            out = F.interpolate(
                data,
                size=size,
                scale_factor=scale_factor,
                mode=mode,
                align_corners=(
                    align_corners
                    if mode in ("linear", "bilinear", "bicubic", "trilinear")
                    else None
                ),
            )
            # Postprocess if requested
            if postprocess is not None:
                if postprocess == 'round':
                    out = torch.round(out)
                elif postprocess == 'floor':
                    out = torch.floor(out)
                elif postprocess == 'ceil':
                    out = torch.ceil(out)
                else:
                    raise ValueError(f"Unknown postprocess: {postprocess}")
            # Cast back to original dtype if it was int
            if was_int and orig_dtype is not None:
                out = out.to(orig_dtype)
            # Remove added batch/channel dims if needed
            if nd == 2:
                out = out[0, 0]
            elif nd == 3:
                out = out[0] if batch_dim == 0 else out[:, 0]
            return AbstractTensor.get_tensor(out)
        else:
            # Numpy fallback: use PIL for images, or scipy.ndimage.zoom if available
            from .numpy_backend import NumPyTensorOperations
            arr = tensor.to_backend(
                AbstractTensor.get_tensor(cls=NumPyTensorOperations)
            )
            data = arr.data if hasattr(arr, "data") else arr
            import numpy as np

            if data.ndim == 2:
                from PIL import Image

                img = Image.fromarray((data * 255).astype(np.uint8))
                img = img.resize(
                    size[::-1], Image.BILINEAR if mode == "bilinear" else Image.NEAREST
                )
                out = np.array(img) / 255.0
            else:
                # Use scipy.ndimage.zoom for nD
                try:
                    from scipy.ndimage import zoom

                    zoom_factors = [size[i] / data.shape[i] for i in range(len(size))]
                    out = zoom(data, zoom_factors, order=1 if mode == "bilinear" else 0)
                except ImportError:
                    raise RuntimeError(
                        "No suitable interpolation backend available (need torch or scipy)"
                    )
            return AbstractTensor.get_tensor(out)

    @staticmethod
    def filtered_poisson(
        rhs,
        *,
        iterations: int = 50,
        filter_strength: float = 0.0,
        mode: str | None = None,
        adjacency=None,
        boundary_mask=None,
        boundary_flux=None,
        normalization: str = "none",
        tol: float | None = None,
    ):
        """Solve a Poisson problem with optional RHS smoothing."""
        from .filtered_poisson import (
            filtered_poisson as _filtered_poisson,
        )

        rhs = AbstractTensor.get_tensor(rhs)
        if adjacency is not None:
            adjacency = AbstractTensor.get_tensor(adjacency, like=rhs)
        if boundary_mask is not None:
            boundary_mask = AbstractTensor.get_tensor(boundary_mask, like=rhs)
        if boundary_flux is not None:
            boundary_flux = AbstractTensor.get_tensor(boundary_flux, like=rhs)
        return _filtered_poisson(
            rhs,
            iterations=iterations,
            filter_strength=filter_strength,
            mode=mode,
            adjacency=adjacency,
            boundary_mask=boundary_mask,
            boundary_flux=boundary_flux,
            normalization=normalization,
            tol=tol,
        )

# Attach to AbstractTensor
AbstractTensor.F = AbstractF

# --- Abstraction method assignments ---------------------------------------
from .abstraction_methods.creation import (
    linspace,
    meshgrid,
    zeros as create_zeros,
    ones as create_ones,
    full as create_full,
    zeros_like as create_zeros_like,
    ones_like as create_ones_like,
    full_like as create_full_like,
    eye_like,
    random_tensor,
    randoms,
    rand_like,
    randn,
    randint,
    randint_like,
    empty as create_empty,
    hanning,
)
from .abstraction_methods.reduction import (
    max as reduction_max,
    argmax as reduction_argmax,
    argmin as reduction_argmin,
    prod as reduction_prod,
)
from .abstraction_methods.indexing import (
    unravel_index as indexing_unravel_index,
    gather,
    scatter,
)


def scatter_row(x: "AbstractTensor", index: Any, row_value: Any, dim: int = 0):
    """Replace row(s) of ``x`` at ``index`` with ``row_value``.

    The update is expressed in terms of :func:`gather` and :func:`scatter` so the
    operation remains differentiable. When ``row_value`` is already an
    ``AbstractTensor`` we use it directly to preserve autograd tape connections;
    otherwise it is promoted to a tensor via :meth:`x.ensure_tensor`.
    """
    from .abstraction import AbstractTensor as AT  # local import to avoid cycles

    if not isinstance(row_value, AT):
        row_value = x.ensure_tensor(row_value)
    current = gather(x, index, dim=dim)
    delta = row_value - current
    return scatter(x, index, delta, dim=dim)
from .abstraction_methods.type_ops import (
    to as type_to,
    astype as type_astype,
    long_cast as type_long_cast,
    float as type_float,
    double as type_double,
    int as type_int,
    long as type_long,
    bool as type_bool,
    cpu as type_cpu,
    cuda as type_cuda,
)
from .abstraction_methods.comparison import (
    argwhere as comp_argwhere,
    greater as comp_greater,
    greater_equal as comp_greater_equal,
    less as comp_less,
    less_equal as comp_less_equal,
    equal as comp_equal,
    not_equal as comp_not_equal,
    any as comp_any,
    where as comp_where,
    nonzero as comp_nonzero,
    isnan as comp_isnan,
    isinf as comp_isinf,
    all as comp_all,
    isfinite as comp_isfinite,
    allclose as comp_allclose,
)
from .abstraction_methods.trigonometry import (
    sin as trig_sin,
    cos as trig_cos,
    tan as trig_tan,
    asin as trig_asin,
    acos as trig_acos,
    atan as trig_atan,
    sinh as trig_sinh,
    cosh as trig_cosh,
    tanh as trig_tanh,
    asinh as trig_asinh,
    acosh as trig_acosh,
    atanh as trig_atanh,
    sec as trig_sec,
    csc as trig_csc,
    cot as trig_cot,
    sech as trig_sech,
    csch as trig_csch,
    coth as trig_coth,
    sinc as trig_sinc,
    deg2rad as trig_deg2rad,
    rad2deg as trig_rad2deg,
)
from .abstraction_methods.fourier import (
    fft as fourier_fft,
    ifft as fourier_ifft,
    rfft as fourier_rfft,
    irfft as fourier_irfft,
    rfftfreq as fourier_rfftfreq,
    fftfreq as fourier_fftfreq,
)
from .abstraction_methods.fold import (
    fold2d as fold2d_ref,
    fold3d as fold3d_ref,
    unfold2d as unfold2d_ref,
    unfold3d as unfold3d_ref,
)
from .abstraction_methods.properties import (
    numel as prop_numel,
    item as prop_item,
    shape as prop_shape,
    shape_ as prop_shape_,
    ndim as prop_ndim,
    dim as prop_dim,
    ndims as prop_ndims,
    device as prop_device,
    dtype as prop_dtype,
    datastring as prop_datastring,
    __str__ as prop_str,
    __format__ as prop_format,
    __repr__ as prop_repr,
    __len__ as prop_len,
)

# --- Autograd method assignments ------------------------------------------
from . import autograd as _autograd_methods
from .backward import BACKWARD_REGISTRY

AbstractTensor.requires_grad_ = _autograd_methods.requires_grad_
AbstractTensor.requires_grad  = _autograd_methods.requires_grad
AbstractTensor.backward       = _autograd_methods.backward
AbstractTensor.grad           = _autograd_methods.grad
AbstractTensor.detach         = _autograd_methods.detach
AbstractTensor.is_leaf        = _autograd_methods.is_leaf
AbstractTensor.retain_grad    = _autograd_methods.retain_grad
AbstractTensor.grad_fn        = _autograd_methods.grad_fn
AbstractTensor.zero_grad      = _autograd_methods.zero_grad



# grads() removed. Use AbstractTensor.grad for all gradient access and computation.

# Register backward algorithms and expose a helper for building pipelines
_BACKWARD = BACKWARD_REGISTRY.register_from_module(_autograd_methods)


def get_backward_tool(ops: Iterable[str]):
    """Return a callable executing backward rules for ``ops`` in order."""
    return _BACKWARD.build(ops)


AbstractTensor.get_backward_tool = staticmethod(get_backward_tool)
AbstractTensor.register_hook  = _autograd_methods.register_hook

# --- Bindings to AbstractTensor -------------------------------------------

AbstractTensor.linspace = staticmethod(linspace)
AbstractTensor.meshgrid = staticmethod(meshgrid)
AbstractTensor.zeros = staticmethod(create_zeros)
AbstractTensor.ones = staticmethod(create_ones)
AbstractTensor.full = staticmethod(create_full)
AbstractTensor.empty = staticmethod(create_empty)
AbstractTensor.hanning = staticmethod(hanning)
from .abstraction_methods.random import Random as _RandomClass
AbstractTensor.random = _RandomClass()
AbstractTensor.randoms = staticmethod(randoms)
AbstractTensor.randint = staticmethod(randint)
AbstractTensor.unravel_index = staticmethod(indexing_unravel_index)
AbstractTensor.gather = gather
AbstractTensor.scatter = scatter
AbstractTensor.scatter_row = scatter_row
AbstractTensor.random_tensor = staticmethod(random_tensor)

# --- Creation helpers: tape + requires_grad + record ------------------------
def _attach_requires_and_record(result, *, op: str, requires_grad: bool, params: dict | None = None):
    from . import autograd as _autograd
    tape = getattr(result, "_tape", None) or _autograd.autograd.tape
    if requires_grad:
        try:
            result.requires_grad_(True)
        except Exception:
            try:
                result._requires_grad = True  # type: ignore[attr-defined]
            except Exception:
                pass
    if getattr(AbstractTensor.autograd, "_no_grad_depth", 0) == 0:
        AbstractTensor.autograd.record(op, [], result, start=None, end=None, params=params or {})
    return result


class _UseTape:
    def __init__(self, tape):
        from . import autograd as _autograd
        self._autograd = _autograd
        self._new = tape
        self._prev = None
    def __enter__(self):
        self._prev = self._autograd.autograd.tape
        self._autograd.autograd.tape = self._new
        return self._new
    def __exit__(self, exc_type, exc, tb):
        self._autograd.autograd.tape = self._prev


def _wrap_creation_fn(op_name: str, raw_fn):
    """
    Wrap a top-level creation function so it accepts:
      - requires_grad: bool = False
      - tape: Optional[GradTape] = None
    and records the op.
    """
    def wrapped(*args, requires_grad: bool = False, tape=None, **kwargs):
        from . import autograd as _autograd
        # If requires_grad=True, force use of the global tape regardless of provided tape
        desired_tape = _autograd.autograd.tape if requires_grad else tape
        if desired_tape is None:
            result = raw_fn(*args, **kwargs)
        else:
            prev = _autograd.autograd.tape
            _autograd.autograd.tape = desired_tape
            try:
                result = raw_fn(*args, **kwargs)
            finally:
                _autograd.autograd.tape = prev

        if not isinstance(result, AbstractTensor):
            result = AbstractTensor.get_tensor(result, tape=desired_tape)

        params = {}
        try:
            params["shape"] = tuple(getattr(result, "shape", ()))
        except Exception:
            pass
        return _attach_requires_and_record(result, op=op_name, requires_grad=requires_grad, params=params)
    return wrapped


def _wrap_meshgrid_fn(raw_fn):
    """Special wrapper for ``meshgrid`` that preserves tuple output."""

    def wrapped(*args, requires_grad: bool = False, tape=None, **kwargs):
        from . import autograd as _autograd

        desired_tape = _autograd.autograd.tape if requires_grad else tape
        if desired_tape is not None:
            prev = _autograd.autograd.tape
            _autograd.autograd.tape = desired_tape
            try:
                result = raw_fn(*args, **kwargs)
            finally:
                _autograd.autograd.tape = prev
        else:
            result = raw_fn(*args, **kwargs)

        tensors = [AbstractTensor.get_tensor(r, tape=desired_tape) for r in result]
        return tuple(tensors)

    return wrapped


# Expose the tape context
AbstractTensor.use_tape = staticmethod(lambda tape: _UseTape(tape))

# Rebind top-level creations so they accept tape=/requires_grad= and record ops.
try:
    AbstractTensor.zeros = staticmethod(_wrap_creation_fn("zeros", AbstractTensor.zeros))
except Exception:
    pass
try:
    AbstractTensor.ones  = staticmethod(_wrap_creation_fn("ones",  AbstractTensor.ones))
except Exception:
    pass
try:
    AbstractTensor.full  = staticmethod(_wrap_creation_fn("full",  AbstractTensor.full))
except Exception:
    pass
try:
    AbstractTensor.empty = staticmethod(_wrap_creation_fn("empty", AbstractTensor.empty))
except Exception:
    pass
try:
    AbstractTensor.linspace = staticmethod(_wrap_creation_fn("linspace", AbstractTensor.linspace))
except Exception:
    pass
try:
    AbstractTensor.meshgrid = staticmethod(_wrap_meshgrid_fn(AbstractTensor.meshgrid))
except Exception:
    pass
try:
    AbstractTensor.hanning = staticmethod(_wrap_creation_fn("hanning", AbstractTensor.hanning))
except Exception:
    pass
try:
    AbstractTensor.randint = staticmethod(_wrap_creation_fn("randint", AbstractTensor.randint))
except Exception:
    pass
try:
    AbstractTensor.randoms = staticmethod(_wrap_creation_fn("randoms", AbstractTensor.randoms))
except Exception:
    pass
try:
    # Ensure arange/range also support requires_grad + tape and force global tape when requested
    AbstractTensor.arange = staticmethod(_wrap_creation_fn("arange", AbstractTensor.arange))
except Exception:
    pass
try:
    AbstractTensor.range = staticmethod(_wrap_creation_fn("range", AbstractTensor.range))
except Exception:
    pass
try:   # Ensure random also supports requires_grad + tape and force global tape when requested
    AbstractTensor.random_tensor = staticmethod(_wrap_creation_fn("random_tensor", AbstractTensor.random_tensor))
except Exception:
    pass
def _wrap_with_autograd(name: str, func: Callable) -> Callable:
    def wrapped(self, *args, **kwargs):
        tensor_args = [a for a in args if isinstance(a, AbstractTensor)]
        tensor_args += [v for v in kwargs.values() if isinstance(v, AbstractTensor)]
        finalize = AbstractTensor._pre_autograd(name, [self] + tensor_args)
        result = func(self, *args, **kwargs)
        return finalize(result)
    return wrapped


def _bind_and_wrap(mapping: Dict[str, Callable]) -> None:
    for _name, _func in mapping.items():
        setattr(AbstractTensor, _name, _wrap_with_autograd(_name, _func))


_bind_and_wrap({
    "reshape": _reshape_methods.reshape,
    "view": _reshape_methods.view,
    "flatten": _reshape_methods.flatten,
    "transpose": _reshape_methods.transpose,
    "permute": _reshape_methods.permute,
    "unsqueeze": _reshape_methods.unsqueeze,
    "squeeze": _reshape_methods.squeeze,
    "swapaxes": _reshape_methods.swapaxes,
    "repeat": _reshape_methods.repeat,
    "repeat_interleave": _reshape_methods.repeat_interleave,
    "eye_like": eye_like,
    "zeros_like": create_zeros_like,
    "ones_like": create_ones_like,
    "full_like": create_full_like,
    "rand_like": rand_like,
    "randn": randn,
    "randint_like": randint_like,
    "max": reduction_max,
    "argmax": reduction_argmax,
    "argmin": reduction_argmin,
    "prod": reduction_prod,
    "to": type_to,
    "astype": type_astype,
    "long_cast": type_long_cast,
    "float": type_float,
    "double": type_double,
    "int": type_int,
    "long": type_long,
    "bool": type_bool,
    "cpu": type_cpu,
    "cuda": type_cuda,
    "greater": comp_greater,
    "greater_equal": comp_greater_equal,
    "less": comp_less,
    "less_equal": comp_less_equal,
    "equal": comp_equal,
    "not_equal": comp_not_equal,
    #"where": comp_where, this is now handled by the elementwise set
    "all": comp_all,
    "any": comp_any,
    "nonzero": comp_nonzero,
    "isfinite": comp_isfinite,
    "isnan": comp_isnan,
    "isinf": comp_isinf,
    "isinfinite": comp_isinf,
    "allclose": comp_allclose,
    "argwhere": comp_argwhere,
    "sin": trig_sin,
    "cos": trig_cos,
    "tan": trig_tan,
    "asin": trig_asin,
    "acos": trig_acos,
    "atan": trig_atan,
    "sinh": trig_sinh,
    "cosh": trig_cosh,
    "tanh": trig_tanh,
    "asinh": trig_asinh,
    "acosh": trig_acosh,
    "atanh": trig_atanh,
    "sec": trig_sec,
    "csc": trig_csc,
    "cot": trig_cot,
    "sech": trig_sech,
    "csch": trig_csch,
    "coth": trig_coth,
    "sinc": trig_sinc,
    "deg2rad": trig_deg2rad,
    "rad2deg": trig_rad2deg,
})

def T(self) -> "AbstractTensor":
    """Matrix transpose convenience alias.

    Returns ``self`` transposed over its last two dimensions.  Vectors and
    scalars are promoted to 2D before transposition to match NumPy's
    ``.T`` semantics.
    """
    ndim = getattr(self, "ndim", len(getattr(self, "shape", ())))
    if ndim < 2:
        if ndim == 0:
            base = self.reshape((1, 1))
        else:
            base = self.reshape((1, self.shape[0]))
        return base.transpose(-2, -1)
    return self.transpose(-2, -1)

AbstractTensor.reshape = _reshape_methods.reshape
AbstractTensor.view = _reshape_methods.view
AbstractTensor.flatten = _reshape_methods.flatten
AbstractTensor.transpose = _reshape_methods.transpose
AbstractTensor.permute = _reshape_methods.permute
AbstractTensor.unsqueeze = _reshape_methods.unsqueeze
AbstractTensor.squeeze = _reshape_methods.squeeze
AbstractTensor.swapaxes = _reshape_methods.swapaxes

AbstractTensor.T = T

AbstractTensor.numel   = prop_numel
AbstractTensor.item    = prop_item
AbstractTensor.shape   = property(prop_shape)
AbstractTensor.shape_  = prop_shape_
AbstractTensor.ndim    = property(prop_ndim)
AbstractTensor.dim     = prop_dim
AbstractTensor.ndims   = prop_ndims
AbstractTensor.device  = property(prop_device)
AbstractTensor.dtype   = property(prop_dtype)
AbstractTensor.datastring = prop_datastring
AbstractTensor.__str__    = prop_str
AbstractTensor.__format__ = prop_format
AbstractTensor.__repr__   = prop_repr
AbstractTensor.__len__    = prop_len

# Reference fold implementations (pure AbstractTensor). Bound as staticmethods.
AbstractTensor.fold2d = staticmethod(fold2d_ref)
AbstractTensor.fold3d = staticmethod(fold3d_ref)
# Bind unfold as instance methods so `self` is passed and signatures match
AbstractTensor.unfold2d = unfold2d_ref
AbstractTensor.unfold3d = unfold3d_ref
from .linalg import (
    dot as linalg_dot,
    norm as linalg_norm,
    cross as linalg_cross,
    trace as linalg_trace,
    det as linalg_det,
    solve as linalg_solve,
    inv as linalg_inv,
    eye as linalg_eye,
    eigh as linalg_eigh,
    cholesky as linalg_cholesky,
)

class _LinalgNS:
    pass

if not hasattr(AbstractTensor, "linalg"):
    AbstractTensor.linalg = _LinalgNS()

# namespace (numpy-like)
AbstractTensor.linalg.dot   = staticmethod(linalg_dot)
AbstractTensor.linalg.norm  = staticmethod(linalg_norm)
AbstractTensor.linalg.cross = staticmethod(linalg_cross)
AbstractTensor.linalg.trace = staticmethod(linalg_trace)
AbstractTensor.linalg.det   = staticmethod(linalg_det)
AbstractTensor.linalg.solve = staticmethod(linalg_solve)
AbstractTensor.linalg.inv   = staticmethod(linalg_inv)
AbstractTensor.linalg.eye   = staticmethod(linalg_eye)
AbstractTensor.linalg.eigh  = staticmethod(linalg_eigh)
AbstractTensor.linalg.cholesky = staticmethod(linalg_cholesky)

# optional top-level shorthands used by your code already
AbstractTensor.dot   = staticmethod(linalg_dot)
AbstractTensor.norm  = staticmethod(linalg_norm)
AbstractTensor.cross = staticmethod(linalg_cross)
AbstractTensor.trace = staticmethod(linalg_trace)
AbstractTensor.det   = staticmethod(linalg_det)
AbstractTensor.solve = staticmethod(linalg_solve)
AbstractTensor.inv   = staticmethod(linalg_inv)
AbstractTensor.eye   = staticmethod(linalg_eye)
AbstractTensor.inverse = staticmethod(linalg_inv)
AbstractTensor.eigh  = staticmethod(linalg_eigh)
AbstractTensor.cholesky = staticmethod(linalg_cholesky)

# --- FFT methods -----------------------------------------------------------
AbstractTensor.fft = fourier_fft
AbstractTensor.ifft = fourier_ifft
AbstractTensor.rfft = fourier_rfft
AbstractTensor.irfft = fourier_irfft
AbstractTensor.rfftfreq = fourier_rfftfreq
AbstractTensor.fftfreq = fourier_fftfreq

def _cbrt(x):
    import numpy as np
    if isinstance(x, AbstractTensor):
        result = type(x)(track_time=x.track_time)
        result.data = np.cbrt(x.data)
        return result
    return np.cbrt(x)

AbstractTensor.cbrt = staticmethod(_cbrt)

def _einsum(equation: str, *tensors: "AbstractTensor") -> "AbstractTensor":
    cls = type(tensors[0])
    data = [t.data if isinstance(t, AbstractTensor) else t for t in tensors]
    result = cls(track_time=False)
    result.data = cls.einsum_(equation, *data)
    return result

AbstractTensor.einsum = staticmethod(_einsum)

def _sparse_coo_tensor(indices, values, size):
    from .coo_matrix import COOMatrix
    return COOMatrix(indices, values, size)

AbstractTensor.sparse_coo_tensor = staticmethod(_sparse_coo_tensor)


def _get_shape(data):
    if not isinstance(data, list):
        return ()
    if not data:
        return (0,)
    return (len(data),) + _get_shape(data[0])


def _flatten(data):
    if not isinstance(data, list):
        return [data]
    return [item for sublist in data for item in _flatten(sublist)]


def default_to_backend(source_ops, tensor, target_ops):
    if type(source_ops) is type(target_ops):
        # Preserve the original tensor when no backend conversion is needed.
        # Cloning here detached parameters from the autograd tape, yielding
        # ``None`` gradients for FluxSpring weights during training.
        return tensor

    data = tensor.tolist()
    dtype = None
    device = None
    try:
        dtype = source_ops.get_dtype(tensor)
    except Exception:
        dtype = None
    if isinstance(dtype, str):
        dtype = None
    try: device = source_ops.get_device(tensor)
    except: pass
    # Works if backend exposes either a classmethod, staticmethod, or instance method:
    import inspect
    cls = type(target_ops)
    raw = inspect.getattr_static(cls, "_tensor_from_list", None)
    tape = getattr(target_ops, "_tape", None)
    if isinstance(raw, classmethod):
        try:
            return raw.__func__(cls, data, dtype, device, tape)
        except TypeError:
            return raw.__func__(cls, data, dtype, device)
    if isinstance(raw, staticmethod):
        try:
            return raw.__func__(data, dtype, device, tape)
        except TypeError:
            return raw.__func__(data, dtype, device)
    return target_ops._tensor_from_list(data, dtype=dtype, device=device, tape=tape)


def get_tensor_operations(
    faculty: Faculty | None = None, *, track_time: bool = False
) -> AbstractTensor:
    """[REMOVED] Use AbstractTensor.get_tensor instead."""
    raise RuntimeError(
        "get_tensor_operations is obsolete. Use AbstractTensor.get_tensor instead."
    )



# --- Delayed backend registration to avoid circular imports ---
from .abstraction_methods.elementwise import (
    __eq__ as elementwise_eq,
    __ne__ as elementwise_ne,
    __lt__ as elementwise_lt,
    __le__ as elementwise_le,
    __gt__ as elementwise_gt,
    __ge__ as elementwise_ge,
    __and__ as elementwise_and,
    __or__ as elementwise_or,
    __xor__ as elementwise_xor,
    __invert__ as elementwise_invert,
    where as elementwise_where,
    sign as elementwise_sign,
    maximum as elementwise_maximum,
    minimum as elementwise_minimum,
    _as_scalar, _scalar_kernel,
    _v1_valuewise, _v2_valuewise, _v3_valuewise
)

# --- Elementwise operator assignments (from abstraction_methods/elementwise.py) ---
AbstractTensor.__eq__    = elementwise_eq
AbstractTensor.__ne__    = elementwise_ne
AbstractTensor.__lt__    = elementwise_lt
AbstractTensor.__le__    = elementwise_le
AbstractTensor.__gt__    = elementwise_gt
AbstractTensor.__ge__    = elementwise_ge
AbstractTensor.__and__   = elementwise_and
AbstractTensor.__or__    = elementwise_or
AbstractTensor.__xor__   = elementwise_xor
AbstractTensor.__invert__= elementwise_invert
AbstractTensor.where     = staticmethod(elementwise_where)
AbstractTensor.sign      = elementwise_sign
AbstractTensor.maximum   = elementwise_maximum
AbstractTensor.minimum   = elementwise_minimum

AbstractTensor._as_scalar   = staticmethod(_as_scalar)
AbstractTensor._scalar_kernel = staticmethod(_scalar_kernel)
AbstractTensor._v1_valuewise  = _v1_valuewise
AbstractTensor._v2_valuewise  = _v2_valuewise
AbstractTensor._v3_valuewise  = _v3_valuewise
