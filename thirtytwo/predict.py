"""python -m thirtytwo.predict xray.jpg --mode full|quick   -> prints the result as JSON"""
import argparse
import contextlib
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m thirtytwo.predict',
                                 description='Find caries, deep caries, periapical lesions and impacted teeth on a '
                                             'panoramic X-ray (JPEG, PNG or DICOM).')
    ap.add_argument('image')
    ap.add_argument('--mode', choices=('full', 'quick'), default='full')
    ap.add_argument('--device', default=None, help="e.g. 'cpu' or 'cuda:0' (default: a GPU if there is one)")
    a = ap.parse_args(argv)
    with contextlib.redirect_stdout(sys.stderr):      # stdout carries only the JSON
        from .pipeline import ThirtyTwo, to_json
        result = ThirtyTwo(device=a.device).predict(a.image, a.mode)
    json.dump(to_json(result), sys.stdout, indent=1)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
