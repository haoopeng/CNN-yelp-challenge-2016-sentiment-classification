# -*- coding:utf-8 -*-
# /bin/python

import json
import csv

outfile = open("review.csv", 'wb')
sfile = csv.writer(outfile, delimiter =",", quoting=csv.QUOTE_MINIMAL)
sfile.writerow(['stars', 'text'])

with open('yelp_academic_dataset_review.json') as f:
    for line in f:
        row = json.loads(line)
        # some special char must be encoded in 'utf-8'
        sfile.writerow([row['stars'], (row['text']).encode('utf-8')])
outfile.close()
