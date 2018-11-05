import sqlite3
import re
import json

def must_solo(tags):
    return "solo" in tags


def must_safe(tags):
    return "safe" in tags


def must_whitebg(tags):
    return "white background" in tags


def must_lookingviewer(tags):
    return "looking at viewer" in tags

def vec(tag):
    def v(tags):
        return [1. if tag in tags else 0]
    return v

def vec_female(tags):
    return [1. if "1girl" in tags else 0]

def vec_male(tags):
    return [1. if "1boy" in tags or "male" in tags else 0]

def vec_eyecolor(color):
    def vecec(tags):
        return [tags[color + " eyes"] if color + " eyes" in tags else 0]
    return vecec

def vec_hair(color):
    def vecec(tags):
        return [tags[color + " hair"] if color + " hair" in tags else 0]
    return vecec

def normalize(funcs):
    def norm(tags):
        ret = []
        for fn in funcs:
            for x in fn(tags):
                ret.append(x)

        total = sum(ret)
        if total != 0:
            ret = [x / total for x in ret]
        return ret

    return norm

filters = [
    must_safe,
    must_solo,
    must_whitebg,
    must_lookingviewer
]

vec_gen = [
    vec_male,
    vec_female,
    normalize([
        vec_eyecolor('blue'),
        vec_eyecolor('brown'),
        vec_eyecolor('purple'),
        vec_eyecolor('red'),
        vec_eyecolor('green'),
        vec_eyecolor('yellow'),
        vec_eyecolor('pink'),
        vec_eyecolor('aqua'),
        vec_eyecolor('black'),
        vec_eyecolor('orange'),
        vec_eyecolor('closed')
    ]),
    normalize([
        vec_hair('very long'),
        vec_hair('long'),
        vec_hair('short'),
    ]),
    normalize([
        vec_hair('brown'),
        vec_hair('black'),
        vec_hair('blonde'),
        vec_hair('blue'),
        vec_hair('purple'),
        vec_hair('silver'),
        vec_hair('pink'),
        vec_hair('red'),
        vec_hair('white'),
        vec_hair('green'),
        vec_hair('orange'),
        vec_hair('grey') # 28
    ]),
    vec('smile'),
    vec('blush'), # 30
    vec('breasts'),
    vec('school uniform'),
    vec('skirt'),
    vec('bow'),
    vec('gloves'), # 35
    vec('jewelry'),
    vec('hair ribbon'),
    vec('open mouth'),
    vec('necktie'),
    vec('bare shoulders'), # 40
    vec('long sleeves'),
    vec('shirt'),
    vec('weapon'),
    vec('glasses')
]

conn = sqlite3.connect('metadata.db')

res = conn.execute('select * from tags')

imgs = {}

vecs = {}
for row in res:
    imageid, tag, score = row
    if not imageid in imgs:
        imgs[imageid] = {
            tag: score
        }
    else:
        imgs[imageid][tag] = score

for img in imgs:
    tags = imgs[img]
    for filt in filters:
        if not filt(tags):
            continue

    vec = []
    for gen in vec_gen:
        for x in gen(tags):
            vec.append(x)
    vecs[img] = vec

with open('../vectors.json', 'w') as fh:
    fh.write(json.dumps(vecs))
