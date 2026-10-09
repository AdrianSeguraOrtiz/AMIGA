#!/usr/bin/env python3
"""Verify vector output and optionally rebuild from sources alone, offline."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import fitz

ROOT=Path(__file__).resolve().parent


def digest(p:Path)->str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify(directory:Path)->dict:
    cfg=json.loads((ROOT/'figure_config.json').read_text())
    stem=cfg['output_basename']
    checks=json.loads((directory/'layout_checks.json').read_text())
    for filename,expected in checks['artifacts'].items():
        if digest(directory/filename)!=expected:
            raise AssertionError(f'Figure artifact changed after its build: {filename}')
    for filename,expected in checks['sources_sha256'].items():
        if digest(ROOT/filename)!=expected:
            raise AssertionError(f'Figure source changed after its build: {filename}')
    if checks['text_box_overlaps'] or checks['text_outside_regions']:
        raise AssertionError('Layout checks show overlapping or out-of-bounds text')
    with fitz.open(directory/f'{stem}.pdf') as doc:
        assert len(doc)==1,'Unexpected PDF page count'
        assert not doc[0].get_images(full=True),'Raster image embedded in PDF'
        assert doc[0].get_drawings(),'PDF has no drawing primitives'
        page_info={'pages':len(doc),'raster_images':len(doc[0].get_images(full=True)),
                   'vector_drawings':len(doc[0].get_drawings())}
    for suffix in ['.svg','_outlined.svg']:
        root=ET.parse(directory/f'{stem}{suffix}').getroot()
        tags=[el.tag.split('}')[-1] for el in root.iter()]
        assert 'image' not in tags, 'SVG includes a raster/image element'
        if suffix=='_outlined.svg': assert 'text' not in tags,'Outlined SVG still contains text'
        else: assert 'text' in tags,'Editable SVG has no text'
    return {'vector_check':'passed','layout_check':'passed','pdf':page_info,
            'text_elements':checks['text_elements'],'source_only_rebuild':None}


def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,default=ROOT)
    p.add_argument('--rebuild',action='store_true')
    p.add_argument('--font-dir',type=Path)
    a=p.parse_args(); directory=a.directory.resolve()
    report=verify(directory)
    if a.rebuild:
        with tempfile.TemporaryDirectory(prefix='amiga-figure-verify-') as name:
            clean=Path(name)
            source_names=['build_figure.py','vector_engine.py','figure_text.json','figure_config.json']
            for source in source_names: shutil.copyfile(ROOT/source,clean/source)
            command=[sys.executable,str(clean/'build_figure.py'),'--out-dir',str(clean),'--dpi','300']
            if a.font_dir: command += ['--font-dir',str(a.font_dir.resolve())]
            subprocess.run(command,check=True,capture_output=True,text=True,timeout=120)
            verify(clean)
            cfg=json.loads((ROOT/'figure_config.json').read_text())
            stem=cfg['output_basename']
            compared={}
            for suffix in ['.pdf','.svg','_outlined.svg','.png','_preview.png']:
                filename=stem+suffix
                expected=digest(directory/filename); actual=digest(clean/filename)
                compared[filename]={'identical':actual==expected,'delivered_sha256':expected,'rebuilt_sha256':actual}
            report['source_only_rebuild']={'inputs':source_names,'uses_reference_image':False,
                'all_five_artifacts_byte_identical':all(v['identical'] for v in compared.values()),
                'compared_files':compared,
                'qualification':'Binary identity is environment- and font-version-dependent.'}
    (directory/'verification_report.json').write_text(json.dumps(report,indent=2)+'\n')
    inventory=sorted(p for p in directory.iterdir() if p.is_file() and p.name!='SHA256SUMS')
    (directory/'SHA256SUMS').write_text(''.join(f'{digest(path)}  {path.name}\n' for path in inventory))
    print(json.dumps(report,indent=2))
    if a.rebuild and not report['source_only_rebuild']['all_five_artifacts_byte_identical']:
        print('Byte differences detected. Compare font hashes and dependency versions.',file=sys.stderr)
        return 1
    return 0

if __name__=='__main__': raise SystemExit(main())
