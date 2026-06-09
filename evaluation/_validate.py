import json
from collections import defaultdict

d = json.load(open('evaluation/dataset_200.json', encoding='utf-8'))['queries']
chunks = json.load(open('data/indexes/chunks.json', encoding='utf-8'))
csrc = {c['source_file'] for c in chunks}
stext = defaultdict(str)
for c in chunks:
    stext[c['source_file']] += ' ' + (c.get('text') or '').lower()

err = []
ids = set()
for q in d:
    if q['id'] in ids:
        err.append('dup ' + q['id'])
    ids.add(q['id'])
    for f in ['id', 'query', 'query_bn', 'category', 'expected_sources',
              'expected_answer_keywords', 'must_refuse']:
        if f not in q:
            err.append(q['id'] + ' missing ' + f)
    if q['category'] in ('in_corpus_factual', 'in_corpus_procedural', 'dialect_bn'):
        if not q['expected_sources'] or q['expected_sources'][0]['source_file'] not in csrc:
            err.append(q['id'] + ' bad src')

kf = kc = 0
for q in d:
    if int(q['id'][1:]) > 60 and q['category'] in ('in_corpus_factual', 'in_corpus_procedural', 'dialect_bn') and q['expected_answer_keywords']:
        kc += 1
        txt = stext[q['expected_sources'][0]['source_file']]
        if sum(1 for k in q['expected_answer_keywords'] if k.lower() in txt) == 0:
            kf += 1

print('total', len(d), '| unique', len(ids), '| errors', len(err), err[:8])
print('grounding:', kc, 'checked,', kf, 'with zero kw in gold source')
print('\n=== kept dialect_bn (new) ===')
for q in d:
    if q['category'] == 'dialect_bn' and int(q['id'][1:]) > 60:
        print('  {}: {}  [src={}]'.format(q['id'], q['query'], q['expected_sources'][0]['source_file']))
