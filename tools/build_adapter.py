"""Rebuild the median box adapter of protocol A and write thirtytwo/settings/adapter.json.

As in the notebook's protocol A cell (cell 28): U.fit_box_adapter(SETS['dis_train']['images'], TEETH, A_TRAIN), where
TEETH are the clean tooth detector's boxes (teeth_clean.pkl; cell 20 makes TEETH = TEETH_E before protocol A runs).
Needs DENTEX (training labels) and the assets; the deployed app only reads the JSON.
"""
import json
import pickle

from common import C, U, keys, sets


def fit():
    teeth = pickle.loads((C.ASSETS / 'teeth_clean.pkl').read_bytes())
    return U.fit_box_adapter(sets()['dis_train']['images'], teeth, keys('A_train'))


def table(adapter, counts):
    return [dict(disease=d, name=U.DIS[d], upper=up, jaw='upper' if up else 'lower', offsets=list(m),
                 pairs=counts.get((d, up), 0)) for (d, up), m in sorted(adapter.items())]


if __name__ == '__main__':
    adapter, counts = fit()
    doc = dict(
        description='Median box adapter of protocol A: for each disease and jaw, the median offset of each edge of '
                    'the annotated disease box from the found tooth box, in tooth widths (dx1, dx2) and heights '
                    '(dy1, dy2). Both tooth classifiers place their boxes with it (box source picked on validation).',
        fitted_with='dentex_utils.fit_box_adapter(DENTEX training records, assets/teeth_clean.pkl, '
                    'assets/keys/A_train.txt), min_count=20',
        table=table(adapter, counts))
    p = C.SETTINGS / 'adapter.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1) + '\n')
    print(f'wrote {p}: {len(adapter)} entries')
    for r in doc['table']:
        print(f"  {r['name']:18s} {r['jaw']:5s} pairs {r['pairs']:4d}  " + ' '.join(f'{v:+.4f}' for v in r['offsets']))
