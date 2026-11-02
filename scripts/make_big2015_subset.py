import csv, random
from collections import defaultdict

SEED = 620
CAP = 250          # max samples per family
PREFIX = "train\\" # match the archive listing; use "" if names are bare

by_class = defaultdict(list)
with open("extracted/trainLabels.csv", newline="") as f:
    for row in csv.DictReader(f):
        by_class[row["Class"]].append(row["Id"])

rng = random.Random(SEED)
chosen = []
for cls in sorted(by_class):
    ids = sorted(by_class[cls])
    chosen += ids if len(ids) <= CAP else rng.sample(ids, CAP)

with open("subset_list.txt", "w") as f:
    for i in chosen:
        f.write(f"{PREFIX}{i}.bytes\n")
print(len(chosen), "files selected")