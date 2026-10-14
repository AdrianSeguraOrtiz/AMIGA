#!/usr/bin/env python3
"""Rebuild the five-phase AMIGA experimental-design figure from vector primitives.

No reference image, network access, AMIGA installation or fitted model is used.
Scientific constants are in design_snapshot.json; text is in figure_text.json.
"""
from __future__ import annotations
import argparse
import copy
import importlib.metadata
import json
import math
from pathlib import Path
import platform
from typing import Any, Sequence
import xml.etree.ElementTree as ET
import fitz
from vector_engine import Figure, blend, find_font, sha256

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = ('build_figure.py', 'vector_engine.py', 'figure_text.json',
                'figure_config.json', 'design_snapshot.json')


def route(f: Figure, points: Sequence[tuple[float,float]], color: str,
          width: float = 2.5, dashed: bool = False, head: float = 9) -> None:
    """Arrow with a routed polyline; corridors are reserved outside the panels."""
    if len(points) < 2:
        raise ValueError('An arrow needs at least two points')
    (x1,y1),(x2,y2) = points[-2:]
    a=math.atan2(y2-y1,x2-x1)
    bx,by=x2-head*math.cos(a),y2-head*math.sin(a)
    f.path('M '+' L '.join(f'{x:g} {y:g}' for x,y in [*points[:-1],(bx,by)]),color,width,dash=dashed)
    f.path(f'M {x2:g} {y2:g} L {bx-head*.48*math.sin(a):g} {by+head*.48*math.cos(a):g} L {bx+head*.48*math.sin(a):g} {by-head*.48*math.cos(a):g} Z',fill=color)


def lock(f: Figure,x:float,y:float,s:float,co:str) -> None:
    f.path(f'M {x-s*.23:g} {y:g} L {x-s*.23:g} {y-s*.20:g} C {x-s*.23:g} {y-s*.49:g} {x+s*.23:g} {y-s*.49:g} {x+s*.23:g} {y-s*.20:g} L {x+s*.23:g} {y:g}',co,2.2)
    f.rect(x-s*.35,y-s*.02,s*.70,s*.48,co,r=s*.08)
    f.circle(x,y+s*.18,s*.055,'#FFFFFF')


def network(f:Figure,x:float,y:float,s:float,co:str,variant:int=0)->None:
    positions=[(-.42,-.04),(-.22,-.4),(.22,-.34),(.42,.06),(.12,.39),(-.32,.3)]
    pairs=[(1,0),(1,2),(2,3),(0,5),(5,4),(1,4),(3,4)]
    if variant % 2: pairs=[(1,0),(1,2),(2,4),(0,5),(5,3),(1,3),(3,4)]
    rad=s*.06
    for a,b in pairs:
        ax,ay=positions[a]; bx,by=positions[b]; dx=bx-ax;dy=by-ay;n=math.hypot(dx,dy)
        f.arrow(x+ax*s+dx/n*rad,y+ay*s+dy/n*rad,x+bx*s-dx/n*(rad+1),y+by*s-dy/n*(rad+1),blend(co,'#FFFFFF',.27),1.2,max(2.3,s*.05))
    for i,(xx,yy) in enumerate(positions):
        f.circle(x+xx*s,y+yy*s,rad,co if i in (1,2) else blend(co,'#FFFFFF',.30),'#FFFFFF',.9)


def tree(f:Figure,x:float,y:float,s:float,co:str)->None:
    pts=[(0,0),(-.22,.28),(.22,.28),(-.38,.62),(-.08,.62),(.08,.62),(.38,.62)]
    for a,b in [(0,1),(0,2),(1,3),(1,4),(2,5),(2,6)]:
        f.line(x+pts[a][0]*s,y+pts[a][1]*s,x+pts[b][0]*s,y+pts[b][1]*s,co,2)
    for i,(xx,yy) in enumerate(pts):
        if i<3:f.circle(x+xx*s,y+yy*s,4.5,'#FFFFFF',co,1.6)
        else:f.rect(x+xx*s-4,y+yy*s-4,8,8,co,r=1.5)


def front_card(f:Figure,x:float,y:float,w:float,h:float,co:str)->None:
    f.rect(x,y,w,h,'#FFFFFF',blend(co,'#FFFFFF',.63),9,1.3)
    f.line(x+13,y+h-14,x+w-11,y+h-14,f.c['gray'],1.3)
    f.line(x+13,y+h-14,x+13,y+13,f.c['gray'],1.3)
    pts=[(.23,.25),(.39,.31),(.56,.44),(.72,.58),(.84,.73)]
    f.path('M '+' L '.join(f'{x+u*w:g} {y+v*h:g}' for u,v in pts),blend(co,'#FFFFFF',.5),1.7)
    for u,v in pts:f.circle(x+u*w,y+v*h,3.7,'#FFFFFF',co,1.6)


def sheet(f:Figure,x:float,y:float,w:float,h:float,co:str)->None:
    f.rect(x,y,w,h,'#FFFFFF',co,4,1.3)
    for j in range(3):f.line(x+w*.23,y+h*(.28+j*.22),x+w*.77,y+h*(.28+j*.22),co,1.4)


def chip(f:Figure,x:float,y:float,w:float,h:float,label:str,co:str,size:float=18,bold:bool=True)->None:
    f.rect(x,y,w,h,blend(co,'#FFFFFF',.93),blend(co,'#FFFFFF',.66),h*.25,1)
    f.text(x+w/2,y+h*.5+size*.35,label,size,bold,co,anchor='middle',maxw=w-16)


def panel(f:Figure,i:int,box:Sequence[float],t:dict[str,Any]):
    x,y,w,h=box
    co=[f.c['blue'],f.c['teal'],f.c['purple'],f.c['gold'],f.c['teal']][i-1]
    f.rect(x,y+4,w,h,'#EBF0F4',r=20)
    f.rect(x,y,w,h,'#FFFFFF',blend(co,'#FFFFFF',.53),20,1.5)
    return f.group(f'phase-{i}',x,y,region=(0,0,w,h)),co


def phase_header(f:Figure,i:int,t:dict[str,Any],w:float,co:str)->None:
    f.circle(45,41,23,co)
    f.text(45,49,str(i),24,True,'#FFFFFF',anchor='middle')
    f.text(82,48,t['phase_titles'][i-1],27 if i<4 else 29,True,maxw=w-107)
    f.text(25,85,t['phase_subtitles'][i-1],20,color=f.c['muted'],maxw=w-50)
    f.line(25,103,w-25,103,blend(co,'#FFFFFF',.84),1)


def output_strip(f:Figure,w:float,label:str,co:str)->None:
    chip(f,24,391,w-48,43,label,co,21)


def preparation(f:Figure,t:dict[str,Any],d:dict[str,Any])->None:
    with f.group('benchmark-inputs',56,192,region=(0,0,930,201)):
        f.rect(0,0,930,201,f.c['soft'],f.c['line'],16,1.2)
        f.text(24,34,t['inputs_title'],23,True)
        # Stacked network cards represent related conditions of a topology.
        for j in (2,1,0):
            f.rect(26+j*10,65-j*6,91,78,'#FFFFFF',f.c['line'],9,1)
            network(f,72+j*10,104-j*6,58,f.c['muted'],j)
        f.text(162,86,str(d['shared_topology_groups'])+' '+t['topologies'],24,True,f.c['teal'])
        f.text(163,115,t['conditions'],18,color=f.c['muted'],leading=24)
        for y,case,co in [(61,d['cases'][0],f.c['blue']),(122,d['cases'][1],f.c['purple'])]:
            f.rect(498,y,403,54,'#FFFFFF',blend(co,'#FFFFFF',.65),10,1.2)
            front_card(f,510,y+7,51,40,co)
            f.text(577,y+25,case['name'],20,True,co)
            f.text(887,y+37,f"{case['fronts']} fronts",19,color=f.c['muted'],anchor='end')
        route(f,[(441,105),(469,105),(469,88),(491,88)],f.c['muted'],1.8,head=6)
        route(f,[(469,105),(469,149),(491,149)],f.c['muted'],1.8,head=6)
        f.text(24,186,t['separate_cases'],18,color=f.c['muted'],maxw=882)
    with f.group('nested-partitions',1020,192,region=(0,0,1364,201)):
        f.rect(0,0,1364,201,f.c['soft'],f.c['line'],16,1.2)
        f.text(24,34,t['split_title'],23,True)
        for r in range(d['outer_folds']):
            for c in range(d['outer_folds']):
                co=f.c['gold'] if c==r else f.c['blue']
                f.rect(32+c*35,59+r*17,31,13,blend(co,'#FFFFFF',.25 if c==r else .65),r=2)
        f.text(117,167,'5 outer folds',19,True,anchor='middle')
        f.arrow(237,101,283,101,f.c['muted'],2,8)
        f.rect(308,56,608,93,f.c['blue_light'],blend(f.c['blue'],'#FFFFFF',.6),12,1.2)
        f.text(329,81,t['outer_train'],21,True,f.c['blue'])
        # Three miniature inner train/validation assignments.
        for r in range(3):
            for c in range(3):
                f.rect(331+c*26,96+r*12,22,9,blend(f.c['blue'] if c!=r else f.c['teal'],'#FFFFFF',.35),r=1.5)
        f.text(436,120,'3 inner folds for phases 1–3',21,True,f.c['blue'])
        f.rect(963,56,375,93,f.c['gold_light'],blend(f.c['gold'],'#FFFFFF',.61),12,1.2)
        lock(f,992,101,31,f.c['gold'])
        f.text(1019,82,t['outer_test'],20.5,True,f.c['gold'])
        f.text(1019,108,t['test_use'],18,color=f.c['muted'],leading=22)
        f.text(309,182,t['split_note'],20,color=f.c['muted'])


def phase1(f:Figure,t:dict[str,Any],d:dict[str,Any],w:float,co:str)->None:
    phase_header(f,1,t,w,co)
    f.text(29,132,t['quality_input'],18,True,leading=23)
    for r,l in enumerate([112,88,65,38,17]):
        f.rect(34,173+18*r,121,12,blend(f.c['gold'],'#FFFFFF',.88),r=3)
        f.rect(34,173+18*r,l,12,blend(f.c['gold'],'#FFFFFF',.16+r*.12),r=3)
    f.text(93,278,t['quality_measure'],17,True,f.c['gold'],anchor='middle')
    f.arrow(174,214,215,214,co,2.2,8)
    values=[['4','3','2','1','0'],['1.00','0.75','0.50','0.25','0.00'],['high','high','mid','low','low']]
    for j,cx in enumerate([292,452,612]):
        f.text(cx,140,t['label_heads'][j],18,True,co,anchor='middle')
        for r,v in enumerate(values[j]):
            f.rect(cx-59,160+r*20,118,17,blend(co,'#FFFFFF',.79+.035*r),r=3)
            f.text(cx,173+r*20,v,13.8,True,co,anchor='middle')
        f.text(cx,280,t['label_details'][j],16.6,color=f.c['muted'],anchor='middle')
    f.rect(25,296,w-50,39,f.c['soft'],f.c['line'],9,1,dash=True)
    f.arrow(46,306,46,325,f.c['gray'],1.7,5)
    f.arrow(59,325,59,306,f.c['gray'],1.7,5)
    f.text(83,322,t['label_controls'],19,color=f.c['muted'],maxw=w-113)
    for x,family in zip([25,256,487],d['families']):chip(f,x,349,208,29,family,co,17)
    output_strip(f,w,t['label_output'],co)
    f.text(w/2,457,t['label_fixed'],18,color=f.c['muted'],anchor='middle')


def phase2(f:Figure,t:dict[str,Any],d:dict[str,Any],w:float,co:str)->None:
    phase_header(f,2,t,w,co)
    for j,(cx,family) in enumerate(zip([135,360,585],d['families'])):
        color=[f.c['blue'],f.c['teal'],f.c['purple']][j]
        f.text(cx,136,family,20,True,color,anchor='middle')
        tree(f,cx,154,59,color)
        # Explicitly schematic miniature search space; count comes from domains.
        for r in range(3):
            for c in range(6):
                f.rect(cx-69+c*24,205+r*22,20,18,blend(color,'#FFFFFF',.76+.04*((r+c)%3)),r=3)
        n=math.prod(len(v) for v in d['grid_domains'][family].values())
        f.text(cx,293,str(n),29,True,color,anchor='middle')
        f.text(cx,319,t['grid_unit'],17.5,color=f.c['muted'],anchor='middle')
    # All candidates pass the same inner-validation comparison.
    for j in range(3):
        f.rect(48+j*31,344,26,23,blend(co,'#FFFFFF',.84),blend(co,'#FFFFFF',.5),4,1)
        for k in range(2):f.line(54+j*31,351+k*8,68+j*31,351+k*8,co,1.2)
    f.text(164,364,t['grid_inner'],21,True,co)
    output_strip(f,w,t['grid_output'],co)
    f.text(w/2,457,t['grid_note'],18,color=f.c['muted'],anchor='middle')


def phase3(f:Figure,t:dict[str,Any],d:dict[str,Any],w:float,co:str)->None:
    phase_header(f,3,t,w,co)
    f.text(25,131,t['feature_importance'],18.7,color=f.c['muted'],maxw=w-50)
    for j,(cx,frac) in enumerate(zip([104,274,444,614],d['feature_fractions'])):
        # Ghost columns show that every fraction refers to the ORIGINAL table.
        for c in range(12):
            for r in range(5):
                col=blend(co,'#FFFFFF',.45+.06*((c+r)%4)) if c<round(12*frac) else '#EEF1F5'
                f.rect(cx-65+c*11,155+r*18,9,15,col,r=1.6)
        f.text(cx,272,t['feature_all'] if j==0 else str(round(frac*100))+'%',20,True,co,anchor='middle')
        if j<3:f.arrow(cx+76,200,cx+94,200,co,1.9,6)
    f.text(w/2,300,t['feature_retained'],17.5,color=f.c['muted'],anchor='middle')
    f.text(w/2,323,t['feature_refit'],18,True,co,anchor='middle',maxw=w-52)
    f.text(26,348,t['feature_select'],20,True,co,leading=26,maxw=w-52)
    output_strip(f,w,t['feature_output'],co)
    f.text(w/2,457,t['feature_note'],17.4,color=f.c['muted'],anchor='middle')


def phase4(f:Figure,t:dict[str,Any],d:dict[str,Any],w:float,co:str)->None:
    phase_header(f,4,t,w,co)
    f.rect(23,122,164,184,f.c['gold_light'],blend(co,'#FFFFFF',.62),12,1.1)
    lock(f,106,140,25,co)
    for j in (2,1,0):front_card(f,47+j*8,167-j*5,100,67,co)
    f.text(106,268,t['test_candidates'],18.5,True,co,anchor='middle',leading=23)
    f.arrow(196,217,234,217,co,2.2,8)
    f.text(257,136,t['supervised'],20.5,True)
    for r,s in enumerate(t['supervised_rows']):
        y=150+r*29
        fill=f.c['teal_light'] if r==0 else f.c['soft']
        color=f.c['teal'] if r==0 else f.c['muted']
        f.rect(250,y,409,25,fill,blend(color,'#FFFFFF',.70),6,1)
        f.circle(265,y+12.5,4.3,color)
        f.text(280,y+18,s,18,r==0,color,maxw=370)
    f.arrow(672,217,711,217,co,2.2,8)
    f.text(894,136,t['top5_title'],22,True,co,anchor='middle')
    f.text(739,171,t['best_front'],18,color=f.c['muted'])
    f.rect(739,181,322,18,blend(co,'#FFFFFF',.87),r=4)
    f.rect(739,181,296,18,co,r=4)
    f.text(739,227,t['best_top5'],18,color=f.c['muted'])
    f.rect(739,237,322,18,blend(f.c['teal'],'#FFFFFF',.88),r=4)
    f.rect(739,237,214,18,f.c['teal'],r=4)
    # Dimension bracket denotes a definition, not an observed performance result.
    f.path('M 953 258 L 953 277 L 1035 277 L 1035 204',co,1.6,dash=True)
    f.text(894,311,t['regret_label'],20,True,co,anchor='middle')
    f.rect(24,334,w-48,58,f.c['soft'],f.c['line'],10,1)
    f.text(43,358,t['objective_panel'],20,True,maxw=w-85)
    f.text(43,381,t['oracle_note'],17.2,color=f.c['muted'])
    for i in range(len(d['final_seeds'])):
        sheet(f,28+i*39,414,28,33,f.c['blue'])
    f.text(119,473,'5 final training seeds',18.5,True,f.c['blue'],anchor='middle')
    f.arrow(235,431,291,431,f.c['blue'],2,8)
    f.rect(316,410,w-340,73,f.c['gold_light'],blend(co,'#FFFFFF',.65),12,1.1)
    f.text(338,438,t['topology_weight'],22,True,co)
    f.text(338,468,t['statistics'],18.5,color=f.c['muted'],maxw=w-365)
    f.text(24,514,t['prediction_note'],18,color=f.c['muted'],maxw=w-48)


def tile_group(f:Figure,cx:float,base:float,n:int,co:str)->None:
    cols=5; rows=math.ceil(n/cols); unit=11;gap=3
    for k in range(n):
        c=k%cols; r=k//cols
        f.rect(cx-34+c*(unit+gap),base-(rows-r)*(unit+gap),unit,unit,blend(co,'#FFFFFF',.33+(k%3)*.1),r=1.8)


def phase5(f:Figure,t:dict[str,Any],d:dict[str,Any],w:float,co:str)->None:
    phase_header(f,5,t,w,co)
    f.rect(24,115,w-48,42,f.c['teal_light'],blend(co,'#FFFFFF',.65),10,1.1)
    lock(f,47,137,25,co)
    f.text(75,142,t['recipe_lock'],21,True,co,maxw=w-100)
    f.text(26,186,t['training_amount'],19,color=f.c['muted'])
    for j,(cx,n) in enumerate(zip([132,356,580],d['phase5']['training_topologies'])):
        tile_group(f,cx,306,n,co)
        f.text(cx,339,str(n),27,True,co,anchor='middle')
        if j<2:f.arrow(cx+64,271,cx+153,271,blend(co,'#FFFFFF',.35),2,8)
    f.rect(787,204,290,119,'none',f.c['blue'],13,1.3,dash=True)
    sheet(f,808,224,41,51,f.c['blue'])
    f.text(865,228,t['full_size'],18,True,f.c['blue'])
    f.text(865,253,t['full_reuse'],17.7,color=f.c['muted'],leading=23)
    f.text(932,347,'69–70',25,True,f.c['blue'],anchor='middle')
    f.text(24,380,t['learning_repetitions'],20.5,True,co,maxw=w-48)
    # Learning-curve axes with no y-values: this is a DESIGN figure, not results.
    f.arrow(44,466,384,466,f.c['gray'],1.6,6)
    f.arrow(44,466,44,402,f.c['gray'],1.6,6)
    for x,label in [(83,'10'),(166,'20'),(267,'40'),(353,'full')]:
        f.line(x,466,x,472,f.c['gray'],1.2)
        f.text(x,491,label,16.2,color=f.c['muted'],anchor='middle')
        f.path(f'M {x} 454 L {x} 418',blend(co,'#FFFFFF',.75),1,dash=True)
    f.text(211,448,'?',36,True,co,anchor='middle')
    f.text(416,417,t['learning_title'],22,True,co)
    f.text(416,446,t['learning_metrics'],19,True)
    f.text(756,417,t['learning_interval'],18.2,color=f.c['muted'],leading=24)
    f.text(416,480,t['learning_scope'],19,True,co)
    f.text(24,514,t['learning_qualification'],18,color=f.c['muted'],maxw=w-48)


def footer(f:Figure,t:dict[str,Any])->None:
    f.rect(56,1680,2328,94,f.c['soft'],f.c['line'],16,1.1)
    for i,(x,title,lines) in enumerate(zip([82,642,1202,1772],t['footer_titles'],t['footer_lines'])):
        if i:f.line(x-25,1698,x-25,1757,f.c['line'],1)
        co=[f.c['blue'],f.c['gold'],f.c['purple'],f.c['teal']][i]
        if i==0:
            for j in range(3):sheet(f,x+j*9,1702,18,22,co)
        elif i==1:
            for j,ww in enumerate([32,24,17]):f.rect(x,1704+j*8,ww,5,co,r=1)
        elif i==2:
            for j in range(3):f.circle(x+7+j*12,1715,5,co)
        else:
            f.circle(x+17,1715,16,'#FFFFFF',co,1.5)
            f.text(x+17,1721,'i',18,True,co,anchor='middle')
        f.text(x+52,1720,title,18.5,True,co)
        if i==2:
            f.text(x,1744,lines,18,color=f.c['muted'],leading=20,maxw=540)
        else:
            f.text(x,1754,lines,19,color=f.c['muted'],maxw=560)
    f.text(56,1810,t['scope_footer'],19,True,maxw=1600)
    f.text(56,1833,t['schematic_footer'],15.5,color=f.c['muted'])
    f.text(2384,1810,'AMIGA · experimental design',18,color=f.c['muted'],anchor='end')


def validate_design(d:dict[str,Any])->None:
    expected={'LightGBM':27,'XGBoost':36,'CatBoost':36}
    counts={k:math.prod(len(v) for v in params.values()) for k,params in d['grid_domains'].items()}
    if counts!=expected:raise ValueError(f'Unexpected committed grid sizes: {counts}')
    if len(d['label_modes'])!=8 or d['phase5']['reselection'] or not d['phase5']['keep_exact_phase4_columns']:
        raise ValueError('The illustrated scientific contract is inconsistent')
    if d['phase5']['jobs']*len(d['final_seeds'])!=d['phase5']['fits']:
        raise ValueError('Learning-curve counts are inconsistent')


def export_panels(cfg:dict[str,Any],out:Path)->None:
    """Vector PDF zooms from the SAME composition (no independent redraw)."""
    dest=out/'phase_details';dest.mkdir(exist_ok=True)
    scale=cfg['canvas']['physical_width_mm']*72/25.4/cfg['canvas']['width']
    with fitz.open(out/(cfg['output_basename']+'.pdf')) as master:
        for name,(x,y,w,h) in cfg['panels'].items():
            clip=fitz.Rect(x*scale,y*scale,(x+w)*scale,(y+h)*scale)
            detail=fitz.open();page=detail.new_page(width=clip.width,height=clip.height)
            page.show_pdf_page(page.rect,master,0,clip=clip)
            detail.save(dest/(name+'.pdf'),no_new_id=True,deflate=True)
            page.get_pixmap(matrix=fitz.Matrix(2.4,2.4),alpha=False).save(dest/(name+'.png'))
            detail.close()


def build(cfg:dict[str,Any],text:dict[str,Any],design:dict[str,Any],out:Path,
          font_dir:Path|None,dpi:int)->dict[str,Any]:
    validate_design(design);out.mkdir(parents=True,exist_ok=True)
    fonts={b:find_font(cfg['fonts']['bold' if b else 'regular'],cfg['font_family'],b,font_dir) for b in [False,True]}
    f=Figure(cfg,out,fonts);t=text;c=f.c
    f.text(56,101,t['brand'],67,True,c['teal'])
    f.line(360,48,360,116,c['line'],2)
    f.text(399,85,t['headline'],44,True,maxw=1750)
    f.text(401,118,t['header_subtitle'],21,color=c['muted'])
    # Five connected phase dots form a small visual signature, not a result chart.
    for i in range(5):
        x=2168+i*46
        if i<4:f.line(x,86,x+46,86,c['line'],2)
        f.circle(x,86,13,[c['blue'],c['teal'],c['purple'],c['gold'],c['teal']][i])
        f.text(x,91,str(i+1),14,True,'#FFFFFF',anchor='middle')
    f.text(56,159,t['tagline'],25,color=c['muted'],maxw=2328)
    preparation(f,t,design)
    f.text(57,440,t['selection_heading'],22,True,c['teal'])
    f.text(454,440,t['selection_subheading'],21,color=c['muted'])
    lock(f,1925,432,30,c['gold'])
    f.text(1953,440,t['test_guard'],21,True,c['gold'],maxw=427)
    drawers=[phase1,phase2,phase3,phase4,phase5]
    for i,fn in enumerate(drawers,1):
        box=cfg['panels'][f'phase{i}'];ctx,co=panel(f,i,box,t)
        with ctx:fn(f,t,design,box[2],co)
    for i in (1,2):
        x,y,w,h=cfg['panels'][f'phase{i}'];n=cfg['panels'][f'phase{i+1}']
        f.arrow(x+w+11,y+233,n[0]-12,y+233,c['teal'],3,11)
    # Shared selection criterion below the three training-only phases.
    f.rect(56,970,1524,78,c['soft'],c['line'],12,1)
    f.text(78,1000,t['selection_rule_title'],21,True,maxw=1480)
    f.text(78,1031,t['selection_rule_detail'],20,color=c['muted'])
    f.arrow(2024,949,2024,963,c['purple'],2.5,7)
    f.rect(1664,967,720,64,c['purple_light'],blend(c['purple'],'#FFFFFF',.53),12,1.2)
    lock(f,1694,993,28,c['purple'])
    f.text(1724,992,t['frozen_title'],21,True,c['purple'],maxw=634)
    f.text(1724,1018,t['frozen_detail'],18.8,color=c['muted'],maxw=634)
    route(f,[(2024,1035),(2024,1080),(612,1080),(612,1111)],c['purple'],2.8,head=10)
    f.text(56,1105,t['evaluation_heading'],21,True,c['gold'])
    f.text(1269,1105,t['evaluation_subheading'],20,color=c['muted'])
    f.arrow(1181,1385,1258,1385,c['teal'],3,12)
    footer(f,t)
    checks=f.finish(dpi)
    checks.update(commit=cfg['commit'],scientific_scope=design['scope'],
                  font_files={('bold' if b else 'regular'):{'name':p.name,'sha256':sha256(p)} for b,p in fonts.items()},
                  sources_sha256={name:sha256(ROOT/name) for name in SOURCE_FILES})
    (out/'layout_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    if checks['text_box_overlaps'] or checks['text_outside_regions'] or checks['raster_images_in_pdf']:
        raise RuntimeError('Layout/vector check failed; inspect layout_checks.json')
    export_panels(cfg,out)
    (out/'build_environment.json').write_text(json.dumps(dict(
        python=platform.python_version(), platform=platform.platform(),
        packages={name:importlib.metadata.version(name) for name in
                  ('reportlab', 'fonttools', 'PyMuPDF', 'pillow', 'charset-normalizer')},
        font_files=checks['font_files'],
        note='Font binaries are not distributed. Byte identity requires matching dependencies and fonts.'
    ),indent=2)+'\n')
    artifacts=[out/(cfg['output_basename']+suffix) for suffix in
               ('.pdf', '.svg', '_outlined.svg', '.png', '_preview.png')]
    artifacts.extend(sorted((out/'phase_details').glob('*')))
    checks['artifacts']={p.relative_to(out).as_posix():sha256(p) for p in artifacts}
    (out/'layout_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    return checks


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=ROOT)
    parser.add_argument('--font-dir',type=Path)
    parser.add_argument('--dpi',type=int)
    args=parser.parse_args()
    load=lambda name:json.loads((ROOT/name).read_text(encoding='utf-8'))
    cfg=load('figure_config.json')
    dpi=args.dpi if args.dpi is not None else cfg.get('png_dpi',300)
    if not 72<=dpi<=1200:parser.error('--dpi must be between 72 and 1200')
    checks=build(cfg,load('figure_text.json'),load('design_snapshot.json'),args.out_dir.resolve(),args.font_dir,dpi)
    print(f"Generated figure: {checks['text_elements']} text elements, {checks['pdf_drawing_objects']} PDF vector drawings; no raster image.")
    return 0

if __name__=='__main__':raise SystemExit(main())
