from pymongo import MongoClient
import gridfs
import sys
import os

db = MongoClient('mongo.sacred')
fs = gridfs.GridFS(db.experiments)
runid = int(sys.argv[1])

runs = db.experiments.runs

run_entry = runs.find_one({'_id': runid})

for artifact in run_entry['artifacts']:
    outpf = 'artifacts/' + str(runid) + '_' + str(artifact['name'])
    print(artifact)
    if not os.path.exists(outpf):
        with open(outpf, 'wb') as fh:
            fh.write(fs.get(artifact['file_id']).read())
