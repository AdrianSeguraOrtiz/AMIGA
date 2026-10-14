#!/usr/bin/env python3
"""Verify vector exports, source identities and offline reconstruction."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import fitz

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = ('build_figure.py', 'vector_engine.py', 'figure_text.json',
                'figure_config.json', 'design_snapshot.json')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob('*') if p.is_file()
                  and p.name != 'SHA256SUMS' and '__pycache__' not in p.parts
                  and p.suffix != '.pyc')


def checked_path(root: Path, name: str) -> Path:
    target = (root / name).resolve()
    if Path(name).is_absolute() or '..' in Path(name).parts or not target.is_relative_to(root):
        raise AssertionError(f'Invalid checksum path: {name}')
    return target


def verify(root: Path, rebuild: bool, font_dir: Path | None, *, refresh_checksums: bool = False) -> dict:
    root = root.resolve()
    cfg = json.loads((ROOT / 'figure_config.json').read_text())
    stem = cfg['output_basename']
    checks = json.loads((root / 'layout_checks.json').read_text())
    for name, expected in checks['sources_sha256'].items():
        if digest(checked_path(ROOT, name)) != expected:
            raise AssertionError(f'Source changed after generation: {name}')
    for name, expected in checks['artifacts'].items():
        if digest(checked_path(root, name)) != expected:
            raise AssertionError(f'Artifact changed after generation: {name}')
    if checks['text_box_overlaps'] or checks['text_outside_regions']:
        raise AssertionError('Text layout checks failed')
    pdfs = [root / (stem + '.pdf'), *sorted((root / 'phase_details').glob('*.pdf'))]
    pdf_info = {}
    for path in pdfs:
        with fitz.open(path) as doc:
            if len(doc) != 1 or doc[0].get_images(full=True) or not doc[0].get_drawings():
                raise AssertionError(f'Expected a one-page vector PDF: {path.name}')
            pdf_info[path.relative_to(root).as_posix()] = dict(
                pages=1, raster_images=0, vector_drawings=len(doc[0].get_drawings()))
    svg_info = {}
    for suffix in ('.svg', '_outlined.svg'):
        name = stem + suffix
        tags = [e.tag.split('}')[-1] for e in ET.parse(root / name).iter()]
        if 'image' in tags or ('text' in tags) == (suffix == '_outlined.svg'):
            raise AssertionError(f'Unexpected raster/text content: {name}')
        svg_info[name] = dict(raster_images=0, text_nodes=tags.count('text'))
    result = dict(pdf=pdf_info, svg=svg_info, text_elements=checks['text_elements'],
                  text_box_overlap_check='passed', text_region_bounds_check='passed',
                  verified_generation_inputs=len(checks['sources_sha256']),
                  verified_artifacts=len(checks['artifacts']),
                  scope='Vector and text-box checks; visual inspection is also required after edits.')
    manifest = root / 'SHA256SUMS'
    if not refresh_checksums:
        recorded = {}
        for line in manifest.read_text().splitlines():
            expected, name = line.split('  ', 1)
            if name in recorded or digest(checked_path(root, name)) != expected:
                raise AssertionError(f'Checksum mismatch: {name}')
            recorded[name] = expected
        if set(recorded) != {p.relative_to(root).as_posix() for p in inventory(root)}:
            raise AssertionError('Checksum inventory differs from the delivered files')
        result['manifest_files_verified'] = len(recorded)
    if rebuild:
        with tempfile.TemporaryDirectory(prefix='amiga-design-verify-') as temporary:
            work = Path(temporary)
            for name in SOURCE_FILES:
                shutil.copyfile(ROOT / name, work / name)
            command = [sys.executable, str(work / 'build_figure.py'), '--out-dir', str(work),
                       '--dpi', str(cfg.get('png_dpi', 300))]
            if font_dir:
                command += ['--font-dir', str(font_dir.resolve())]
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
            comparisons = {}
            for name in checks['artifacts']:
                expected, actual = digest(root / name), digest(work / name)
                comparisons[name] = dict(identical=expected == actual,
                                         delivered_sha256=expected, rebuilt_sha256=actual)
            identical = all(row['identical'] for row in comparisons.values())
            result['source_only_rebuild'] = dict(inputs=list(SOURCE_FILES), uses_reference_image=False,
                all_artifacts_byte_identical=identical, artifacts=comparisons,
                qualification='Byte identity requires matching dependencies, settings and exact font versions.')
            if not identical:
                raise AssertionError('Rebuilt exports differ; inspect the rendering environment and fonts')
    (root / 'verification_report.json').write_text(json.dumps(result, indent=2) + '\n')
    manifest.write_text(''.join(f'{digest(p)}  {p.relative_to(root).as_posix()}\n' for p in inventory(root)))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT)
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--font-dir', type=Path)
    parser.add_argument('--refresh-checksums', action='store_true',
                        help='Refresh the file inventory after an intentional source or documentation update.')
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.rebuild, args.font_dir,
                            refresh_checksums=args.refresh_checksums), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
