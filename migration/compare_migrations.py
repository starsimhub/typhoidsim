"""
Compare the performance of the different migrations
"""

import sciris as sc
import matplotlib.pyplot as plt

# Set folders
original = sc.path('original')
final = sc.path('final')
folders = [
    'migrated_new1',
    'migrated_new2',
    'migrated_tests_new1',
    'migrated_tests_new2',
]
folders = [sc.path(f) for f in folders]
files = sc.getfilelist(folder=original, nopath=True)
files = [f for f in files if f] # Removes ''

def run(dir1, dir2, file):
    f1 = dir1 / file
    f2 = dir2 / file
    cmd = f'git diff --no-index {f1} {f2}'
    raw = sc.runcommand(cmd).splitlines()
    return raw


#%% Build up index of changes
target_diffs = []
for file in files:
    print(f'Working on {file}...')
    raw = run(original, final, file)
    is_old = False
    is_new = False
    old_lines = []
    new_lines = []
    finished = False
    for line in raw:
        if len(line) > 5: # Avoid formatting
            finished = False
            if line[0] != '-':
                is_old = False
            if line [0] != '+':
                if is_new:
                    finished = True
                is_new = False
            if line[0] == '-' and line[:3] != '---':
                is_old = True
                old_lines.append(line)
            if line[0] == '+' and line[:3] != '+++':
                is_new = True
                new_lines.append(line)

            if finished:
                entry = sc.dictobj(file=file, old=sc.newlinejoin(old_lines), new=sc.newlinejoin(new_lines))
                target_diffs.append(entry)
                old_lines = []
                new_lines = []

print(f'Found {len(target_diffs)} target diffs')

# diffs =


