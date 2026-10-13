#!/usr/bin/env python3
"""Deterministic SVG/PDF drawing primitives for the AMIGA graphical abstract.

Every visible element is drawn from geometric primitives and text.
No image reference, network connection, external API or generative image model is used.

Usage:
    python build_figure.py
    python build_figure.py --out-dir /tmp/figure --dpi 600
    python build_figure.py --font-dir /path/to/dejavu-fonts

Python >=3.10; dependencies are listed in requirements.txt. No Graphviz,
Inkscape, LaTeX, network connection, API, or generative model is needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from contextlib import AbstractContextManager, contextmanager
from html import escape
from pathlib import Path
from typing import Any, Iterator, Sequence

import fitz
from fontTools.ttLib import TTFont as OutlineFont
from fontTools.pens.svgPathPen import SVGPathPen
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent


def blend(a: str, b: str, t: float) -> str:
    return '#' + ''.join(f'{round(int(a[i:i+2],16)*(1-t)+int(b[i:i+2],16)*t):02X}'
                         for i in (1, 3, 5))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_font(filename: str, family: str, bold: bool, font_dir: Path | None) -> Path:
    """Resolve the exact requested font; do not silently substitute a family."""
    candidates = []
    if font_dir:
        candidates.append(font_dir / filename)
    candidates += [
        Path('/usr/share/fonts/truetype/dejavu') / filename,
        Path('/usr/local/share/fonts') / filename,
        Path.home() / '.fonts' / filename,
        Path.home() / '.local/share/fonts' / filename,
        Path('/Library/Fonts') / filename,
        Path.home() / 'Library/Fonts' / filename,
        Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / filename,
    ]
    if shutil.which('fc-match'):
        proc = subprocess.run(['fc-match', '-f', '%{file}',
                               f'{family}:style={"Bold" if bold else "Book"}'],
                              check=False, capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout:
            candidates.append(Path(proc.stdout.strip()))
    for path in candidates:
        if path.is_file() and path.name.lower() == filename.lower():
            return path.resolve()
    raise RuntimeError(f'Missing font {filename}. Install the configured DejaVu fonts, '
                       'or pass --font-dir pointing to its TTF files. '
                       'No font files are distributed in this package.')


def attr(**kwargs: Any) -> str:
    return ' '.join(f'{k.replace("xml_space", "xml:space").replace("_", "-")}="{escape(str(v), quote=True)}"'
                    for k, v in kwargs.items() if v is not None)


def outline_svg(svg: str, fonts: dict[bool, Path]) -> str:
    """Replace only text with glyph paths; keep gradients and shapes vectorial.

    Converting the whole PDF through some SVG exporters rasterizes gradient
    shadings. This direct text-only conversion deliberately avoids that.
    The SVG contains paths, not distributed or embedded font files.
    """
    ns = "http://www.w3.org/2000/svg"
    xlink = "http://www.w3.org/1999/xlink"
    ET.register_namespace("", ns)
    ET.register_namespace("xlink", xlink)
    root = ET.fromstring(svg)
    defs = root.find(f"{{{ns}}}defs")
    if defs is None:
        defs = ET.SubElement(root, f"{{{ns}}}defs")
    parsed = {b: OutlineFont(str(path)) for b, path in fonts.items()}
    glyphs = {b: font.getGlyphSet() for b, font in parsed.items()}
    maps = {b: font.getBestCmap() for b, font in parsed.items()}
    declared = set()
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag != f"{{{ns}}}text":
                continue
            value = child.text or ""
            bold = child.get("font-weight") == "bold"
            font = parsed[bold]
            scale = float(child.get("font-size", "14")) / font["head"].unitsPerEm
            group = ET.Element(f"{{{ns}}}g", {
                "transform": child.get("transform", ""),
                "fill": child.get("fill", "#000000"),
                "aria-label": value,
            })
            ET.SubElement(group, f"{{{ns}}}title").text = value
            advance = 0.0
            for char in value:
                name = maps[bold].get(ord(char))
                if name is None:
                    raise ValueError(f"Font does not contain {char!r}")
                identifier = f"glyph-{'b' if bold else 'r'}-{ord(char):04x}"
                if identifier not in declared:
                    pen = SVGPathPen(glyphs[bold])
                    glyphs[bold][name].draw(pen)
                    ET.SubElement(defs, f"{{{ns}}}path", {
                        "id": identifier, "d": pen.getCommands(),
                    })
                    declared.add(identifier)
                ET.SubElement(group, f"{{{ns}}}use", {
                    f"{{{xlink}}}href": f"#{identifier}",
                    "transform": f"translate({advance:.8f},0) scale({scale:.10f},{-scale:.10f})",
                })
                advance += font["hmtx"][name][0] * scale
            index = list(parent).index(child)
            parent.remove(child)
            parent.insert(index, group)
    for font in parsed.values():
        font.close()
    return ET.tostring(root, encoding="unicode", xml_declaration=False) + "\n"


class Figure:
    """Write the same primitives to SVG and an actual vector PDF.

    Coordinates are top-left-origin design units. Fonts are subset-embedded
    in the PDF. The editable SVG retains text; a second SVG with glyph outlines
    is generated directly for font-independent viewing.
    """
    def __init__(self, cfg: dict[str, Any], out: Path, fonts: dict[bool, Path]):
        self.cfg, self.out, self.fonts = cfg, out, fonts
        self.c = cfg['palette']
        self.w, self.h = cfg['canvas']['width'], cfg['canvas']['height']
        self.scale = cfg['canvas']['physical_width_mm'] * 72 / 25.4 / self.w
        self.stem = cfg['output_basename']
        self.pdf = canvas.Canvas(str(out / f'{self.stem}.pdf'),
                                 pagesize=(self.w*self.scale, self.h*self.scale),
                                 pageCompression=1, invariant=1)
        self.pdf.setTitle(cfg['title'])
        self.pdf.setAuthor('Adrian Segura Ortiz')
        self.pdf.setSubject(cfg['provenance'])
        self.pdf.translate(0, self.h*self.scale)
        self.pdf.scale(self.scale, -self.scale)
        self.parts: list[str] = []
        self.defs: list[str] = []
        self.texts: list[dict[str, Any]] = []
        self.regions: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None
        self.ox = self.oy = 0.0
        self.primitive_count = 0
        for bold, path in fonts.items():
            pdfmetrics.registerFont(TTFont(self.font_name(bold), str(path)))
        self.rect(0, 0, self.w, self.h, self.c['white'])

    @staticmethod
    def font_name(bold: bool) -> str:
        return 'FigureBold' if bold else 'FigureRegular'

    @contextmanager
    def group(self, identifier: str, x: float = 0, y: float = 0,
              region: tuple[float, float, float, float] | None = None) -> Iterator[None]:
        old = self.current
        if region:
            xx, yy, w, h = region
            self.current = dict(id=identifier, x=xx+self.ox+x, y=yy+self.oy+y,
                                width=w, height=h)
            self.regions.append(self.current)
        self.parts.append(f'<g id="{escape(identifier)}" transform="translate({x:g},{y:g})">')
        self.pdf.saveState()
        self.pdf.translate(x, y)
        self.ox += x
        self.oy += y
        try:
            yield
        finally:
            self.ox -= x
            self.oy -= y
            self.pdf.restoreState()
            self.parts.append('</g>')
            self.current = old

    def _paint(self, fill: str | None, stroke: str | None, sw: float) -> None:
        if fill and fill != 'none': self.pdf.setFillColor(HexColor(fill))
        if stroke and stroke != 'none': self.pdf.setStrokeColor(HexColor(stroke))
        self.pdf.setLineWidth(sw)
        self.pdf.setLineCap(1)
        self.pdf.setLineJoin(1)

    def rect(self, x: float, y: float, w: float, h: float, fill: str | Sequence[str] = 'none',
             stroke: str | None = None, r: float = 0, sw: float = 1,
             dash: bool = False) -> None:
        self.primitive_count += 1
        gradient = not isinstance(fill, str)
        fill_string = str(fill)
        if gradient:
            g = f'gradient-{len(self.defs)}'
            self.defs.append(f'<linearGradient id="{g}" x1="0" y1="0" x2="1" y2="1">'
                             f'<stop offset="0" stop-color="{fill[0]}"/>'
                             f'<stop offset="1" stop-color="{fill[1]}"/></linearGradient>')
            fill_string = f'url(#{g})'
        self.parts.append('<rect '+attr(x=x,y=y,width=w,height=h,rx=r,fill=fill_string,
                                       stroke=stroke,stroke_width=sw,
                                       stroke_dasharray='4 3' if dash else None) + '/>')
        self.pdf.saveState()
        self.pdf.setDash([4,3] if dash else [])
        if gradient:
            self.pdf.saveState()
            p = self.pdf.beginPath()
            p.roundRect(x,y,w,h,r)
            self.pdf.clipPath(p, stroke=0, fill=0)
            self.pdf.linearGradient(x,y,x+w,y+h,[HexColor(fill[0]),HexColor(fill[1])])
            self.pdf.restoreState()
        self._paint(None if gradient else fill,stroke,sw)
        self.pdf.roundRect(x,y,w,h,r,stroke=int(bool(stroke)),
                           fill=int(not gradient and fill!='none'))
        self.pdf.restoreState()

    def circle(self,x:float,y:float,r:float,fill:str='none',stroke:str|None=None,sw:float=1,
               dash:bool=False) -> None:
        self.primitive_count += 1
        self.parts.append('<circle '+attr(cx=x,cy=y,r=r,fill=fill,stroke=stroke,
                                         stroke_width=sw,stroke_dasharray='4 3' if dash else None)+'/>' )
        self.pdf.saveState()
        self._paint(fill,stroke,sw)
        self.pdf.setDash([4,3] if dash else [])
        self.pdf.circle(x,y,r,stroke=int(bool(stroke)),fill=int(fill!='none'))
        self.pdf.restoreState()

    def ellipse(self,x:float,y:float,rx:float,ry:float,fill:str='none',stroke:str|None=None,
                sw:float=1) -> None:
        self.primitive_count += 1
        self.parts.append('<ellipse '+attr(cx=x,cy=y,rx=rx,ry=ry,fill=fill,
                                          stroke=stroke,stroke_width=sw)+'/>' )
        self.pdf.saveState(); self._paint(fill,stroke,sw)
        self.pdf.ellipse(x-rx,y-ry,x+rx,y+ry,stroke=int(bool(stroke)),fill=int(fill!='none'))
        self.pdf.restoreState()

    def path(self,d:str,stroke:str|None=None,sw:float=1.5,fill:str='none',dash:bool=False) -> None:
        """Absolute SVG path commands M, L, C, Q and Z are supported."""
        self.primitive_count += 1
        self.parts.append('<path '+attr(d=d,stroke=stroke,stroke_width=sw,fill=fill,
                            stroke_linecap='round',stroke_linejoin='round',
                            stroke_dasharray='4 3' if dash else None)+'/>' )
        tokens = re.findall(r'[MLCQZmlcqz]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?', d)
        p = self.pdf.beginPath(); i=0; current=(0.,0.)
        while i < len(tokens):
            op=tokens[i]; i+=1
            n={'M':2,'L':2,'C':6,'Q':4,'Z':0}.get(op)
            if n is None: raise ValueError(f'Unsupported path operation {op!r}')
            a=list(map(float,tokens[i:i+n])); i+=n
            if op=='M': p.moveTo(*a); current=tuple(a)
            elif op=='L': p.lineTo(*a); current=tuple(a)
            elif op=='C': p.curveTo(*a); current=tuple(a[-2:])
            elif op=='Q':
                x0,y0=current; x1,y1,x2,y2=a
                p.curveTo(x0+2*(x1-x0)/3,y0+2*(y1-y0)/3,
                          x2+2*(x1-x2)/3,y2+2*(y1-y2)/3,x2,y2)
                current=(x2,y2)
            else: p.close()
        self.pdf.saveState(); self._paint(fill,stroke,sw)
        self.pdf.setDash([4,3] if dash else [])
        self.pdf.drawPath(p,stroke=int(bool(stroke)),fill=int(fill!='none'))
        self.pdf.restoreState()

    def line(self,x1:float,y1:float,x2:float,y2:float,color:str,sw:float=1.5) -> None:
        self.path(f'M {x1} {y1} L {x2} {y2}',color,sw)

    def arrow(self,x1:float,y1:float,x2:float,y2:float,color:str|None=None,
              sw:float=1.7,head:float=6) -> None:
        color=color or self.c['muted']
        a=math.atan2(y2-y1,x2-x1)
        bx=x2-head*math.cos(a); by=y2-head*math.sin(a)
        self.line(x1,y1,bx,by,color,sw)
        pts=[(x2,y2),(bx-head*.58*math.sin(a),by+head*.58*math.cos(a)),
             (bx+head*.58*math.sin(a),by-head*.58*math.cos(a))]
        self.path('M '+' L '.join(f'{x:g} {y:g}' for x,y in pts)+' Z',None,0,color)

    def text(self,x:float,y:float,value:str|Sequence[str],size:float=14,
             bold:bool=False,color:str|None=None,anchor:str='start',
             maxw:float|None=None,leading:float|None=None) -> None:
        lines=[value] if isinstance(value,str) else value
        for j,s in enumerate(lines):
            yy=y+j*(leading or size*1.25)
            width=pdfmetrics.stringWidth(str(s),self.font_name(bold),size)
            sx = 1.0
            if maxw and width > maxw + 0.01:
                raise ValueError(f'Text exceeds reserved width: {s!r} ({width:.1f} > {maxw})')
            visual_w=width*sx
            xx=x-(visual_w/2 if anchor=='middle' else visual_w if anchor=='end' else 0)
            self.parts.append('<text '+attr(x=0,y=0,
                transform=f'translate({xx:g},{yy:g}) scale({sx:.8f},1)',
                font_family=self.cfg['font_family'],font_size=size,
                font_weight='bold' if bold else 'normal',
                fill=color or self.c['ink'],xml_space='preserve')+'>'+escape(str(s))+'</text>')
            self.pdf.saveState(); self.pdf.translate(xx,yy); self.pdf.scale(sx,-1)
            self.pdf.setFont(self.font_name(bold),size)
            self.pdf.setFillColor(HexColor(color or self.c['ink']))
            self.pdf.drawString(0,0,str(s)); self.pdf.restoreState()
            ascent,descent=pdfmetrics.getAscentDescent(self.font_name(bold),size)
            self.texts.append(dict(text=str(s),x=xx+self.ox,y=yy-ascent+self.oy,
                            width=visual_w,height=ascent-descent,size=size,
                            horizontal_scale=sx,region=self.current['id'] if self.current else None))

    def finish(self,dpi:int) -> dict[str,Any]:
        self.pdf.showPage(); self.pdf.save()
        svg=('<?xml version="1.0" encoding="UTF-8"?>\n'
             f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
             f'viewBox="0 0 {self.w} {self.h}">\n'
             f'<title>{escape(self.cfg["title"])}</title>\n'
             f'<desc>{escape(self.cfg["provenance"])}</desc>\n'
             '<defs>'+''.join(self.defs)+'</defs>\n'+ '\n'.join(self.parts)+'\n</svg>\n')
        (self.out/f'{self.stem}.svg').write_text(svg,encoding='utf-8')
        with fitz.open(self.out/f'{self.stem}.pdf') as doc:
            page=doc[0]
            preview=page.get_pixmap(matrix=fitz.Matrix(self.w/page.rect.width,
                                                     self.w/page.rect.width),alpha=False)
            preview.save(self.out/f'{self.stem}_preview.png')
            page.get_pixmap(dpi=dpi,alpha=False).save(self.out/f'{self.stem}.png')
            (self.out/f'{self.stem}_outlined.svg').write_text(
                outline_svg(svg, self.fonts), encoding='utf-8')
            image_count=len(page.get_images(full=True))
            drawing_count=len(page.get_drawings())
            text_span_count=sum(len(line['spans']) for b in page.get_text('dict')['blocks']
                                if 'lines' in b for line in b['lines'])
        regions={r['id']:r for r in self.regions}
        outside=[]
        for t in self.texts:
            r=regions.get(t['region'],dict(x=0,y=0,width=self.w,height=self.h))
            if (t['x']<r['x']-1 or t['y']<r['y']-1 or
                t['x']+t['width']>r['x']+r['width']+1 or
                t['y']+t['height']>r['y']+r['height']+1):
                outside.append(t)
        overlaps=[]
        for i,a in enumerate(self.texts):
            for b in self.texts[i+1:]:
                w=min(a['x']+a['width'],b['x']+b['width'])-max(a['x'],b['x'])
                h=min(a['y']+a['height'],b['y']+b['height'])-max(a['y'],b['y'])
                if w>1 and h>1:
                    overlaps.append(dict(a=a['text'],b=b['text'],intersection=[w,h]))
        checks=dict(text_elements=len(self.texts),vector_primitives=self.primitive_count,
                    pdf_drawing_objects=drawing_count,pdf_text_spans=text_span_count,
                    raster_images_in_pdf=image_count,text_outside_regions=outside,
                    text_box_overlaps=overlaps,reference_image_used_by_builder=False,
                    bbox_note='Font ascent/descent boxes; no claim of a complete geometry proof.')
        (self.out/'layout_checks.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
        (self.out/'text_geometry.json').write_text(json.dumps(self.texts,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        return checks
