import sys

def arch(path):
    try:
        with open(path, 'rb') as f:
            f.seek(0x3C)
            pe_off = int.from_bytes(f.read(4), 'little')
            f.seek(pe_off + 4)
            machine = int.from_bytes(f.read(2), 'little')
            if machine == 0x8664:
                return 'x64'
            if machine == 0x014c:
                return 'x86'
            return hex(machine)
    except Exception as e:
        return f'ERROR: {e}'

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: pe_arch.py <file> [<file> ...]')
        sys.exit(2)
    for p in sys.argv[1:]:
        print(p, arch(p))
