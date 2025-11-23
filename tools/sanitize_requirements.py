import re
from pathlib import Path

def sanitize_line(line: str) -> str:
    line = line.strip()
    if not line or line.startswith('#'):
        return line
    # If the line uses 'name @ file://...' keep only the name
    if ' @ ' in line:
        name = line.split(' @ ', 1)[0].strip()
        return name.lower()
    return line

def main():
    src = Path('requirements.txt')
    dst = Path('requirements-pypi.txt')
    if not src.exists():
        print('requirements.txt not found in current directory')
        return
    lines = src.read_text(encoding='utf-8').splitlines()
    out = []
    for l in lines:
        out.append(sanitize_line(l))
    dst.write_text('\n'.join(out) + '\n', encoding='utf-8')
    print(f'Wrote sanitized requirements to {dst}')

if __name__ == '__main__':
    main()
