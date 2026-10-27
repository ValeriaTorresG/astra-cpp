"""minimal fits binary table (bintable) writer for test data without astropy."""
from pathlib import Path

import numpy as np

_TFORM = {'u1': 'B', 'i2': 'I', 'i4': 'J', 'i8': 'K', 'f4': 'E', 'f8': 'D'}


def _card(key, value=None, comment=''):
    if value is None:
        text = f'{key:<8}'
    else:
        if isinstance(value, bool):
            v = f'{"T" if value else "F":>20}'
        elif isinstance(value, str):
            v = f"'{value:<8}'"
        else:
            v = f'{value:>20}'
        text = f'{key:<8}= {v}' + (f' / {comment}' if comment else '')
    return text[:80].ljust(80).encode('ascii')


def _block(cards):
    raw = b''.join(cards + [_card('END')])
    return raw + b' ' * (-len(raw) % 2880)


def write_bintable(path: Path, columns: dict[str, np.ndarray]) -> None:
    names = list(columns)
    n = len(columns[names[0]])
    fields, tforms = [], []
    for name in names:
        col = np.asarray(columns[name])
        if col.dtype.kind == 'S':
            fields.append((name, col.dtype))
            tforms.append(f'{col.dtype.itemsize}A')
        else:
            fields.append((name, col.dtype.newbyteorder('>')))
            tforms.append(_TFORM[col.dtype.str[1:]])
    rec = np.empty(n, dtype=fields)
    for name in names:
        rec[name] = columns[name]
    primary = _block([_card('SIMPLE', True), _card('BITPIX', 8), _card('NAXIS', 0), _card('EXTEND', True)])
    cards = [_card('XTENSION', 'BINTABLE'), _card('BITPIX', 8), _card('NAXIS', 2),
             _card('NAXIS1', rec.dtype.itemsize), _card('NAXIS2', n), _card('PCOUNT', 0),
             _card('GCOUNT', 1), _card('TFIELDS', len(names))]
    for i, (name, tform) in enumerate(zip(names, tforms), 1):
        cards += [_card(f'TTYPE{i}', name), _card(f'TFORM{i}', tform)]
    data = rec.tobytes()
    with open(path, 'wb') as fh:
        fh.write(primary + _block(cards) + data + b'\0' * (-len(data) % 2880))
