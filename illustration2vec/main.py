import faulthandler
import i2v
from PIL import Image
import sqlite3
import os
faulthandler.enable()

illust2vec = i2v.make_i2v_with_chainer(
    "illust2vec_tag_ver200.caffemodel", "tag_list.json")

print('Loaded i2v')
conn = sqlite3.connect('metadata.db')
cur = conn.cursor()
cur.execute('''
CREATE TABLE IF NOT EXISTS `tags` (
`imageid`    TEXT NOT NULL,
`tag`    TEXT NOT NULL,
`score`    REAL,
PRIMARY KEY(`imageid`,`tag`)
);
''')
cur.execute('''
CREATE INDEX IF NOT EXISTS ix_tag ON tags(tag);
''')
conn.commit()

imdir = '../malfc'

batch_size = 100
imids = []
imgs = []
itags = []
files = os.listdir(imdir)
files.sort()
i = 0

skip = 0
for r in cur.execute("SELECT count(distinct(imageid)) from tags"):
    skip = r[0]

print('skipping', skip)

for im in files:
    i += 1
    if i <= skip:
        continue

    img = Image.open(imdir + "/" + im)
    imids.append(im)
    imgs.append(img)
    if i % batch_size == 0:
        for tags in illust2vec.estimate_plausible_tags(imgs, threshold=0.1):
            itags.append(tags)
        for ip in imgs:
            ip.close()
        imgs = []
        j = 0
        for tags in itags:
            for tag, score in tags['general']:
                cur.execute("INSERT INTO tags (imageid, tag, score) VALUES (?,?,?)",
                            (imids[j], tag, score))
            for tag, score in tags['rating']:
                cur.execute("INSERT INTO tags (imageid, tag, score) VALUES (?,?,?)",
                            (imids[j], tag, score))
            j += 1
        conn.commit()
        itags = []
        imids = []
        print('pred', i)

for tags in illust2vec.estimate_plausible_tags(imgs, threshold=0.1):
    itags.append(tags)
for ip in imgs:
    ip.close()
imgs = []
j = 0
for tags in itags:
    for tag, score in tags['general']:
        cur.execute("INSERT INTO tags (imageid, tag, score) VALUES (?,?,?)",
                    (imids[j], tag, score))
    for tag, score in tags['rating']:
        cur.execute("INSERT INTO tags (imageid, tag, score) VALUES (?,?,?)",
                    (imids[j], tag, score))
    j += 1

print('skipping', skip)

conn.commit()
