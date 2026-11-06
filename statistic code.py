from __future__ import annotations
import csv
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
import sys
from docx import Document

ORDER = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Exchange']
HORIZONS = ['96', '192', '336', '720']
METRICS = ['MSE', 'MAE']


def midrank(values: dict[str, Decimal]) -> dict[str, float]:
    result = {}
    for method, v in values.items():
        smaller = sum(x < v for x in values.values())
        tied = sum(x == v for x in values.values())
        result[method] = 1.0 + smaller + (tied - 1) / 2.0
    return result


def extract(path: Path):
    doc = Document(str(path))
    candidates = []
    for tab in doc.tables:
        if len(tab.rows) != 14 or len(tab.columns) != 22:
            continue
        h = [c.text.strip() for c in tab.rows[0].cells]
        mh = [c.text.strip() for c in tab.rows[1].cells]
        if h[0:2] == ['Methods', 'Methods'] and mh[2:4] == ['MSE','MAE']:
            candidates.append(tab)
    if len(candidates) != 2:
        raise RuntimeError(f'Expected 2 Table 4/5 candidates, found {len(candidates)}')
    methods = [candidates[0].rows[0].cells[col].text.strip() for col in range(2,22,2)]
    raw = {}  # (metric,dataset,horizon)-> method:value
    for t in candidates:
        this_methods = [t.rows[0].cells[col].text.strip() for col in range(2,22,2)]
        if this_methods != methods:
            raise ValueError('Inconsistent method headers across tables')
        for row in t.rows[2:]:
            dataset, horizon = row.cells[0].text.strip(), row.cells[1].text.strip()
            if dataset not in ORDER or horizon not in HORIZONS:
                raise ValueError((dataset,horizon))
            for metric, offset in [('MSE',0),('MAE',1)]:
                key=(metric,dataset,horizon)
                if key in raw: raise ValueError(f'Duplicate task: {key}')
                raw[key]={method:Decimal(row.cells[2+2*k+offset].text.strip()) for k,method in enumerate(methods)}
    if len(raw)!=2*6*4: raise ValueError(f'Expected 48 task×metric combinations, got {len(raw)}')
    return methods, raw


def write_csv(path, rows):
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        csv.writer(f).writerows(rows)


def build(manuscript: Path, outdir: Path):
    outdir.mkdir(parents=True,exist_ok=True)
    methods, raw = extract(manuscript)
    per_task = {k:midrank(v) for k,v in raw.items()}
    summaries = {}
    for metric in METRICS:
        matrix = [['Dataset','Horizon'] + methods]
        for dataset in ORDER:
            for horizon in HORIZONS:
                r=per_task[(metric,dataset,horizon)]
                matrix.append([dataset,horizon]+[r[m] for m in methods])
        write_csv(outdir / f'RankMatrix_{metric}_24tasks.csv',matrix)

        summary = [['Dataset'] + methods]
        for dataset in ORDER:
            averages={method:round(sum(per_task[(metric,dataset,h)][method] for h in HORIZONS)/4,4) for method in methods}
            summary.append([dataset]+[averages[m] for m in methods])
        write_csv(outdir / f'Table7_{metric}_MeanRanks.csv',summary)
        summaries[metric]=summary
        

if __name__=='__main__':
    manuscript=Path(sys.argv[1] if len(sys.argv)>1 else 'Manuscript (clean).docx')
    outdir=Path(sys.argv[2] if len(sys.argv)>2 else 'reviewer7_descriptive_results')
    methods,summaries,rank_matrix=build(manuscript,outdir)
    print('Methods:',methods)
    print('ETTh1 MSE Proposed, individual horizons:', [rank_matrix[('MSE','ETTh1',h)]['Proposed'] for h in HORIZONS])
    for met in METRICS:
        print('\n',met)
        for row in summaries[met]: print(row)
