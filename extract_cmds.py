import json
import re
from pathlib import Path

p = Path('/home/vscode/.vscode-server/data/User/workspaceStorage/e3b4030e53a51b3faef6141698ef9f85/GitHub.copilot-chat/transcripts/2111a887-28b7-4280-b113-662460efcbf9.jsonl')
cmds = set()
patterns = [
    r'run_pipeline\.py\s+--\S+',
    r'evaluate_gold\.py\s+--\S+'
]

def find_commands(text):
    for pat in patterns:
        for match in re.finditer(pat, text):
            # Extract the full line containing the match
            start = text.rfind('\n', 0, match.start()) + 1
            end = text.find('\n', match.end())
            if end == -1: end = len(text)
            line = text[start:end].strip()
            # Clean up JSON noise if any
            line = line.replace('\\"', '"').replace('\\n', ' ')
            if line:
                cmds.add(line)

if p.exists():
    with p.open('r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            try:
                data = json.loads(line)
                content = data.get('content', '')
                find_commands(content)
            except:
                continue

for cmd in sorted(cmds):
    print(cmd)
