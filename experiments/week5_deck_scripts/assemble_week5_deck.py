"""Assemble experiments/Week5_GAW_detailed.html from week5_body.html, reusing the Week 4 deck's stylesheet.
usage: python assemble_week5_deck.py week5_body.html EXPERIMENTS_DIR"""
import re, sys
from pathlib import Path
body = open(sys.argv[1], encoding='utf-8').read(); EXP = Path(sys.argv[2])
w4 = open(EXP / 'Week4_GAW_detailed.html', encoding='utf-8').read()
head = re.sub(r'<title>.*?</title>', '<title>Week 5 · BC Hotspot Forecasting · GAW</title>', w4[:w4.index('</style></head>') + 8])
i = body.index('</style>') + 8
open(EXP / 'Week5_GAW_detailed.html', 'w', encoding='utf-8', newline='\n').write(head + '\n' + body[:i] + '</head><body>' + body[i:].lstrip())
print('written')
