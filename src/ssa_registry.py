from enum import Enum
from typing import Dict, Callable, List, Any
from collections import defaultdict


class Handler(Enum):
    """
    Enumeration of all SSA operations used.
    Values correspond to SSA `Instr.op` names.
    """
    # Arithmetic
    Add           = "Add"
    Sub           = "Sub"
    Mul           = "Mul"
    Div           = "Div"
    Mod           = "Mod"
    Pow           = "Pow"
    Neg           = "Neg"
    Abs           = "Abs"

    # Bitwise
    And           = "And"
    Or            = "Or"
    Xor           = "Xor"
    Not           = "Not"
    Shl           = "Shl"
    Shr           = "Shr"

    # Logical
    LAnd          = "LAnd"
    LOr           = "LOr"
    LNot          = "LNot"

    # Comparison
    Eq            = "Eq"
    Ne            = "Ne"
    Lt            = "Lt"
    Le            = "Le"
    Gt            = "Gt"
    Ge            = "Ge"

    # Memory & Indexing
    Load          = "Load"
    Store         = "Store"
    Alloca        = "Alloca"
    GetElementPtr = "GetElementPtr"

    # Casts & Conversions
    Cast          = "Cast"
    Trunc         = "Trunc"
    ZExt          = "ZExt"
    SExt          = "SExt"
    FpToSi        = "FpToSi"
    FpToUi        = "FpToUi"
    SiToFp        = "SiToFp"
    UiToFp        = "UiToFp"

    # Control Flow
    Phi           = "Phi"
    Br            = "Br"
    CondBr        = "CondBr"
    Ret           = "Ret"
    Call          = "Call"

    # Misc
    Select        = "Select"
    Const         = "Const"  # literal constants

    def __str__(self) -> str:
        return self.value


# -----------------------------------------------------------------------------
# Sympy → SSA base name map
# -----------------------------------------------------------------------------
sympy_ssa_name_map: Dict[str, Handler] = {
    # Symbols & Variables
    'symbol':              Handler.Load,
    'var':                 Handler.Load,

    # Literals / Constants
    'integer':             Handler.Const,
    'float':               Handler.Const,
    'rational':            Handler.Const,
    'half':                Handler.Const,
    'pi':                  Handler.Const,
    'e':                   Handler.Const,
    'i':                   Handler.Const,
    'imaginaryunit':       Handler.Const,
    'true':                Handler.Const,
    'false':               Handler.Const,

    # Arithmetic
    'add':                 Handler.Add,
    'sub':                 Handler.Sub,
    'mul':                 Handler.Mul,
    'div':                 Handler.Div,
    'mod':                 Handler.Mod,
    'pow':                 Handler.Pow,
    'neg':                 Handler.Neg,
    'abs':                 Handler.Abs,

    # Bitwise
    'bitwise_and':         Handler.And,
    'bitwise_or':          Handler.Or,
    'bitwise_xor':         Handler.Xor,
    'invert':              Handler.Not,

    # Logical
    'and':                 Handler.LAnd,
    'or':                  Handler.LOr,
    'not':                 Handler.LNot,
    'xor':                 Handler.Xor,

    # Comparison
    'eq':                  Handler.Eq,
    'equality':            Handler.Eq,
    'ne':                  Handler.Ne,
    'unequality':          Handler.Ne,
    'lt':                  Handler.Lt,
    'strictlessthan':      Handler.Lt,
    'le':                  Handler.Le,
    'lessthanorequal':     Handler.Le,
    'gt':                  Handler.Gt,
    'strictgreaterthan':   Handler.Gt,
    'ge':                  Handler.Ge,
    'greaterthanorequal':  Handler.Ge,

    # Memory & Indexing
    'load':                Handler.Load,
    'store':               Handler.Store,
    'alloca':              Handler.Alloca,
    'getelementptr':       Handler.GetElementPtr,
    'idx':                 Handler.GetElementPtr,
    'indexed':             Handler.Load,
    'indexedbase':         Handler.Alloca,
    'matrixelement':       Handler.Load,
    'matrixsymbol':        Handler.Alloca,

    # Casts & Conversions
    'cast':                Handler.Cast,
    'trunc':               Handler.Trunc,
    'zext':                Handler.ZExt,
    'sext':                Handler.SExt,
    'fptosi':              Handler.FpToSi,
    'fptoui':              Handler.FpToUi,
    'sitofp':              Handler.SiToFp,
    'uitofp':              Handler.UiToFp,

    # Selection / Piecewise
    'select':              Handler.Select,
    'piecewise':           Handler.Select,
    'exprcondpair':        Handler.Select,

    # Control Flow
    'phi':                 Handler.Phi,
    'br':                  Handler.Br,
    'condbr':              Handler.CondBr,
    'ret':                 Handler.Ret,

    # Function-Calls (catch-all externals)
    'call':                Handler.Call,
    'sin':                 Handler.Call,
    'cos':                 Handler.Call,
    'tan':                 Handler.Call,
    'exp':                 Handler.Call,
    'log':                 Handler.Call,
    'sqrt':                Handler.Call,
    'floor':               Handler.Call,
    'ceiling':             Handler.Call,
    'round':               Handler.Call,
    'max':                 Handler.Call,
    'min':                 Handler.Call,
    'sum':                 Handler.Call,
    'matrix':              Handler.Call,
    'transpose':           Handler.Call,
    'inverse':             Handler.Call,
    'trace':               Handler.Call,
    'function':            Handler.Call,
}


# -----------------------------------------------------------------------------
# Placeholder for developer-resolved disambiguation
# -----------------------------------------------------------------------------
sympy_ssa_disambig: Dict[str, Dict[str, Any]] = {
    # e.g. 'bitwise_and': {'bitwidth': 32, 'signed': False},
}


class SSARegistry:
    """
    Holds the authoritative SSA name map, disambiguation parameters,
    and helper-function registry for SSA emission.
    """
    name_map: Dict[str, Handler] = sympy_ssa_name_map
    disambig_map: Dict[str, Dict[str, Any]] = sympy_ssa_disambig
    ssa_helpers: Dict[Handler, Callable[..., Any]] = {}

    @classmethod
    def detect_ambiguous(cls) -> Dict[Handler, List[str]]:
        """Return handlers mapped from multiple SymPy node names."""
        rev = defaultdict(list)
        for sym, h in cls.name_map.items():
            rev[h].append(sym)
        return {h: syms for h, syms in rev.items() if len(syms) > 1}

    @classmethod
    def interactive_disambiguate(cls):
        """
        Prompt developer to disambiguate multi-mapped SymPy nodes by
        collecting extra parameters (e.g., bitwidth, signed).
        """
        amb = cls.detect_ambiguous()
        print("Ambiguous Handler mappings detected:")
        for handler, syms in amb.items():
            print(f"\nHandler {handler}:")
            for sym in syms:
                print(f"  SymPy node `{sym}`:")
                params: Dict[str, Any] = {}
                bw = input("    bitwidth (e.g. 1,8,32)? ")
                sd = input("    signed? (y/n) ")
                params['bitwidth'] = int(bw)
                params['signed']   = (sd.lower() == 'y')
                cls.disambig_map[sym] = params

    @classmethod
    def generate_disambig_code(cls) -> str:
        """
        Emit a Python code snippet for the `sympy_ssa_disambig` dict.
        """
        lines = ["sympy_ssa_disambig = {" ]
        for sym, args in cls.disambig_map.items():
            lines.append(f"    {sym!r}: {args},")
        lines.append("}")
        return "\n".join(lines)

    @classmethod
    def register_helper(cls, handler: Handler):
        """Decorator to register an SSA-emission helper for a Handler."""
        def decorator(fn: Callable[..., Any]):
            cls.ssa_helpers[handler] = fn
            return fn
        return decorator

    @classmethod
    def emit_ssa(cls, handler: Handler, builder, operands: List[Any], **kwargs) -> Any:
        """
        Dispatch to the registered helper for `handler`.
        Raises KeyError if none is found.
        """
        fn = cls.ssa_helpers.get(handler)
        if not fn:
            raise KeyError(f"No SSA helper registered for {handler}")
        return fn(builder, operands, **kwargs)
