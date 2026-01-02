import sys, traceback
try:
    import torch
    prefixes = getattr(getattr(torch, 'utils', None), 'cmake_prefix_path', '')
    if isinstance(prefixes, (list, tuple)):
        prefixes = '\n'.join(prefixes)
    out = []
    out.append('PYEXE:' + sys.executable)
    out.append('TV:' + str(getattr(torch, '__version__', '')))
    out.append('TFILE:' + str(getattr(torch, '__file__', '')))
    out.append('PREFIX:' + str(prefixes))
    out.append('CUDA:' + str(getattr(getattr(torch, 'version', None), 'cuda', None)))
    print('\n'.join(out))
except Exception:
    traceback.print_exc()
    sys.exit(1)
