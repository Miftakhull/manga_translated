import json
d = json.load(open('run_full.json', encoding='utf-8'))
print(d.get('provider'), d.get('model'))
for stem, p in d['pages'].items():
    print('==', stem)
    for r in p['regions']:
        print('  ', r['idx'], r['font'], repr(r['src']), '->', repr(r['en']))
    ca = p['clean_audit']
    print('   dirty', ca['dirty_px_total'], 'comp', ca['components_total'])
