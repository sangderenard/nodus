import sys
try:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Delete dev_bundle/include/torch-src? [y/N]: "
    # Print the prompt to stderr immediately so CMake configure output shows it
    print(prompt, file=sys.stderr)
    sys.stderr.flush()
    ans = input()
    if ans.strip().lower() in ('y', 'yes'):
        print('YES')
        sys.exit(0)
    else:
        print('NO')
        sys.exit(2)
except Exception as e:
    print('ERROR:', e, file=sys.stderr)
    sys.exit(3)
