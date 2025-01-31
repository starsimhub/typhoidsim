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
    'migrated_new_tests',
    'migrated_new_tests2',
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

def find_diffs(dir1, dir2, verbose=True):
    diffs = []
    for file in files:
        if verbose: print(f'Working on {file}...')
        raw = run(dir1, dir2, file)
        # is_old = False
        is_new = False
        old_lines = []
        new_lines = []
        finished = False
        for line in raw:
            if len(line) > 5: # Avoid formatting
                finished = False
                # if line[0] != '-':
                    # is_old = False
                if line [0] != '+':
                    if is_new:
                        finished = True
                    is_new = False
                if line[0] == '-' and line[:3] != '---':
                    # is_old = True
                    old_lines.append(line)
                if line[0] == '+' and line[:3] != '+++':
                    is_new = True
                    new_lines.append(line)

                if finished:
                    entry = sc.dictobj(file=file, old=sc.newlinejoin(old_lines), new=sc.newlinejoin(new_lines))
                    diffs.append(entry)
                    old_lines = []
                    new_lines = []
    return diffs


#%% Build up index of changes
target_diffs = find_diffs(original, final)


print(f'Found {len(target_diffs)} target diffs')

sc.heading('Finding all diffs ...')
diffs = sc.objdict()
for f1 in folders:
    for f2 in folders:
        fn1 = str(f1)
        fn2 = str(f2)
        if fn1 != fn2:
            key = f'{fn1} ... {fn2}'
            print(f'Working on {key}')
            diffs[key] = find_diffs(f1, f2, verbose=False)

print('Found these numbers of diffs:')
for key,d in diffs.items():
    print(key, len(d))


sc.heading('Quantifying performance...')
perf = sc.dictobj()
all_targets = [entry['old'] for entry in target_diffs]
for key,odiff in diffs.items():
    count = 0
    n_target = len(target_diffs)
    for entry in odiff:
        if entry['old'] in all_targets:
            count += 1
    perf[key] = count

print('Performance')
print(perf)
