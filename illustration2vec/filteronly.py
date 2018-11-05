import sqlite3
import re


def must(tag, threshold):
    return lambda tags: tag in tags and tags[tag] > threshold


def must_not(tag, threshold):
    return lambda tags: tag not in tags or tags[tag] < threshold


filters = [
    must('safe', 0.5),
    must('solo', 0.5),
    must('white background', 0.1),
    must_not('chibi', 0.1),
    must_not('photo', 0.1),
    must_not('no humans', 0.1),
    must_not('monochrome', 0.1)

]

conn = sqlite3.connect('metadata-combined.db')

res = conn.execute('select * from tags')

imgs = {}

for row in res:
    imageid, tag, score = row
    if imageid not in imgs:
        imgs[imageid] = {
            tag: score
        }
    else:
        imgs[imageid][tag] = score

for img in imgs:
    tags = imgs[img]
    keep = True
    for filt in filters:
        if not filt(tags):
            keep = False
            break
    if keep:
        print('cp combined/' + img + ' combinedfiltered')
    else:
        print('cp combined/' + img + ' rejected')
