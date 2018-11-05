import time
import datetime
import json

f = 1136098799
l = 1550818801
o = json.load(open('vectors.old.json'))

m = {}

with open('dates.txt') as fh:
    for line in fh:
        r = time.mktime(datetime.datetime.strptime(line.split(',')[1].strip(), "%Y-%m-%d").timetuple())
        m[line.split(',')[0]] = (r - f) / (l - f)

for k, v in o.items():
    o[k].append(m[k.split('_')[1]])

print(json.dumps(o))
