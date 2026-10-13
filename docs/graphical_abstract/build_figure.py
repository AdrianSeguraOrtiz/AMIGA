#!/usr/bin/env python3
"""Build the AMIGA graphical abstract from code, never from a reference image.

Python >= 3.10. Run `python build_figure.py`. Rendering is entirely local.
Edit figure_text.json and figure_config.json for wording and styling.
The companion vector_engine.py writes the same primitives to SVG and PDF.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

from vector_engine import Figure, blend, find_font, sha256

ROOT = Path(__file__).resolve().parent


def checkmark(f: Figure, x: float, y: float, size: float, color: str) -> None:
    f.path(f'M {x-size*.35} {y} L {x-size*.08} {y+size*.27} L {x+size*.40} {y-size*.32}', color, max(1.5, size*.095))


def arrow_path(f: Figure, points: list[tuple[float, float]], color: str,
               width: float = 2, dashed: bool = False, head: float = 8) -> None:
    """Route an arrow through reserved corridors, finishing with a vector head."""
    if len(points) < 2:
        raise ValueError('An arrow requires two points')
    x1,y1 = points[-2]; x2,y2 = points[-1]
    a=math.atan2(y2-y1,x2-x1)
    bx=x2-head*math.cos(a); by=y2-head*math.sin(a)
    route=points[:-1]+[(bx,by)]
    f.path('M '+' L '.join(f'{x:.3f} {y:.3f}' for x,y in route),color,width,dash=dashed)
    f.path('M '+f'{x2:.3f} {y2:.3f} L {bx-head*.48*math.sin(a):.3f} {by+head*.48*math.cos(a):.3f} L {bx+head*.48*math.sin(a):.3f} {by-head*.48*math.cos(a):.3f} Z',fill=color)


def network(f: Figure, x: float, y: float, size: float, color: str,
            variant: int = 0, detailed: bool = False) -> None:
    """Original schematic directed graphs, NOT inferred or evaluated networks."""
    positions=[(-.38,-.03),(-.19,-.38),(.23,-.31),(.41,.03),(.17,.37),(-.22,.32)]
    if detailed:
        positions += [(-.08,-.09),(.15,.10),(-.36,.18),(.02,-.46)]
    edges=[(1,0),(1,2),(2,3),(0,5),(5,4),(3,4),(1,4)]
    if variant % 3 == 1: edges=[(1,0),(1,3),(2,0),(2,3),(3,5),(5,4),(4,2)]
    if variant % 3 == 2: edges=[(1,0),(1,2),(2,4),(0,5),(5,3),(1,4),(3,4)]
    if detailed: edges += [(1,6),(6,4),(6,7),(9,2),(9,6),(8,5),(8,6),(7,3),(7,4)]
    r=size*(.052 if detailed else .068)
    for j,(a,b) in enumerate(edges):
        ax,ay=positions[a]; bx,by=positions[b]
        dx,dy=bx-ax,by-ay; norm=math.hypot(dx,dy)
        start=(x+size*ax+dx/norm*r,y+size*ay+dy/norm*r)
        end=(x+size*bx-dx/norm*(r+1),y+size*by-dy/norm*(r+1))
        f.arrow(*start,*end,blend(color,'#FFFFFF',.22+(j%3)*.08),
                sw=max(1,size*.009),head=max(2.5,size*.031))
    for j,(px,py) in enumerate(positions):
        co = color if j in [1,2,9] else blend(color,'#FFFFFF',.30+(j%3)*.10)
        if j in [1,2,9]:
            f.rect(x+px*size-r,y+py*size-r,2*r,2*r,co,'#FFFFFF',r*.5,1.3)
        else:
            f.circle(x+px*size,y+py*size,r,co,'#FFFFFF',1.3)


def heatmap(f: Figure, x: float, y: float, w: float, h: float, color: str, seed: int = 0) -> None:
    cols,rows=9,7
    dx,dy=w/cols,h/rows
    for r in range(rows):
        for c in range(cols):
            t = .14+(((r*13+c*7+seed*11) % 19)/19)*.78
            f.rect(x+c*dx,y+r*dy,dx-1.2,dy-1.2,blend(color,'#FFFFFF',t),r=1)


def base_graphs(f: Figure, x: float, y: float, w: float, h: float, color: str) -> None:
    for i in [2,1,0]:
        xx=x+12*i; yy=y-7*i
        f.rect(xx,yy,w,h,'#FFFFFF',blend(color,'#FFFFFF',.61),9,1.4)
        network(f,xx+w/2,yy+h/2,h*.81,color,i)


def tree(f: Figure, x: float, y: float, size: float, color: str) -> None:
    pts=[(0,0),(-.24,.35),(.24,.35),(-.39,.75),(-.10,.75),(.10,.75),(.39,.75)]
    for a,b in [(0,1),(0,2),(1,3),(1,4),(2,5),(2,6)]:
        f.line(x+pts[a][0]*size,y+pts[a][1]*size,x+pts[b][0]*size,y+pts[b][1]*size,color,1.9)
    for i,(xx,yy) in enumerate(pts):
        if i<3: f.circle(x+xx*size,y+yy*size,3.7,'#FFFFFF',color,1.7)
        else: f.rect(x+xx*size-3.7,y+yy*size-3.4,7.4,6.8,color,r=1.3)


def lock(f: Figure, x: float, y: float, size: float, color: str) -> None:
    f.path(f'M {x-size*.22} {y} L {x-size*.22} {y-size*.18} C {x-size*.22} {y-size*.49} {x+size*.22} {y-size*.49} {x+size*.22} {y-size*.18} L {x+size*.22} {y}', color,2.2)
    f.rect(x-size*.34,y-size*.025,size*.68,size*.49,color,r=size*.07)
    f.circle(x,y+size*.19,size*.055,'#FFFFFF')


def mini_document(f: Figure,x: float,y: float,w: float,h: float,color: str) -> None:
    f.rect(x,y,w,h,'#FFFFFF',color,4,1.8)
    for k in range(3): f.line(x+w*.22,y+h*(.3+k*.2),x+w*.78,y+h*(.3+k*.2),color,1.5)


def feature_icon(f: Figure, x: float, y: float, which: int, co: str) -> None:
    if which==0:
        for i,pos in enumerate([-.18,.25,.05]):
            yy=y-14+i*13
            f.line(x-22,yy,x+22,yy,blend(co,'#FFFFFF',.55),2.5)
            f.circle(x+pos*60,yy,4.8,'#FFFFFF',co,2)
    elif which==1:
        f.line(x-21,y+15,x+24,y+15,co,1.5)
        for i,h in enumerate([15,32,22]):
            f.rect(x-18+i*14,y+15-h,9,h,blend(co,'#FFFFFF',.12*i),r=2)
    elif which==2:
        network(f,x,y,49,co,1)
    else:
        heatmap(f,x-23,y-17,48,36,co,2)


def front(f: Figure, x: float,y: float,w: float,h: float,t: dict[str,Any],
          colors: list[str],variant:int=0) -> None:
    # Two abstract maximizing objectives. Pareto candidates descend toward the
    # lower-right in screen coordinates; pale interior points are dominated.
    left=x+26; top=y+18; bottom=y+h-19; right=x+w-32
    f.arrow(left,bottom,right+10,bottom,f.c['gray'],1.4,5)
    f.arrow(left,bottom,left,top-3,f.c['gray'],1.4,5)
    f.text(left+8,top-3,t['objective_y'],12.8,color=f.c['muted'])
    f.text(right+4,bottom+19,t['objective_x'],12.8,color=f.c['muted'],anchor='end')
    pts=[(.10,.08),(.33,.15),(.55,.30),(.73,.56),(.90,.85)]
    if variant: pts=[(.11,.07),(.30,.17),(.54,.31),(.77,.55),(.90,.85)]
    xx=[left+15+u*(right-left-30) for u,v in pts]
    yy=[top+13+v*(bottom-top-30) for u,v in pts]
    f.path('M '+' L '.join(f'{a:g} {b:g}' for a,b in zip(xx,yy)),blend(f.c['blue'],'#FFFFFF',.55),1.8)
    for j,(u,v) in enumerate([(.09,.51),(.22,.65),(.40,.88),(.07,.85)]):
        f.circle(left+15+u*(right-left-30),top+13+v*(bottom-top-30),3.0,'#DBE5EE')
    for i,(a,b) in enumerate(zip(xx,yy)):
        f.circle(a,b,9.0,'#FFFFFF',colors[i],2.1)
        f.text(a,b+4.2,chr(65+i),10.6,True,colors[i],anchor='middle')


def card(f:Figure,ident:str,x:float,y:float,w:float,h:float):
    f.rect(x,y+3,w,h,'#EDF2F6',r=18)
    f.rect(x,y,w,h,'#FFFFFF',f.c['line'],18,1.2)
    return f.group(ident,x,y,region=(0,0,w,h))


def data_panel(f:Figure,t:dict[str,Any],training:bool,w:float,h:float) -> None:
    co=f.c['blue'] if training else f.c['teal']
    f.text(22,34,t['data_title_training' if training else 'data_title_application'],24,True,maxw=w-44)
    f.text(22,62,t['data_subtitle'],17.2,color=f.c['muted'])
    heatmap(f,25,100,133,92,co,0 if training else 3)
    base_graphs(f,194,112,90,73,co)
    f.text(91,220,t['expression_label'],16.2,color=f.c['muted'],anchor='middle')
    f.text(250,220,t['networks_label'],16.2,color=f.c['muted'],anchor='middle')
    f.line(24,243,w-24,243,f.c['line'],1.2)
    if training:
        f.rect(20,261,w-40,106,f.c['gold_light'],blend(f.c['gold'],'#FFFFFF',.60),12,1.2)
        network(f,75,304,72,f.c['gold'],2)
        f.text(133,286,t['reference_title'],18.5,True,f.c['gold'],leading=23)
        f.text(133,337,t['reference_subtitle'],15.8,color=f.c['muted'])
    else:
        f.rect(20,261,w-40,106,f.c['teal_light'],blend(co,'#FFFFFF',.67),12,1.2)
        f.circle(52,296,13,'#FFFFFF',co,1.4)
        checkmark(f,52,296,17,co)
        f.text(80,292,t['no_reference'],18.7,True,co,leading=25)
        f.text(80,345,t['no_reference_note'],14.8,color=f.c['muted'])


def candidates_panel(f:Figure,t:dict[str,Any],training:bool,w:float,h:float,
                     colors:list[str]) -> None:
    f.text(22,34,t['candidate_title'],23.5,True,maxw=w-44)
    f.text(22,62,t['candidate_subtitle'],16.4,color=f.c['muted'])
    front(f,24,94,240,127,t,colors,variant=int(not training))
    # Three candidate miniatures link the optimization points to actual GRNs.
    for j,(dy,var) in enumerate([(0,0),(39,1),(78,2)]):
        # Keep the candidate GRN miniatures closer to the Pareto plot and align
        # the stepped stack more cleanly.
        xx=281+j*8; yy=87+dy
        f.rect(xx,yy,85,62,'#FFFFFF',blend(colors[var],'#FFFFFF',.49),8,1.3)
        network(f,xx+47,yy+32,51,colors[var],var)
        f.text(xx+10,yy+17,chr(65+var),11.8,True,colors[var])
    f.text(24,255,t['front_explanation'],16.7,color=f.c['muted'],leading=24)
    if training:
        f.rect(20,307,w-40,60,f.c['gold_light'],blend(f.c['gold'],'#FFFFFF',.58),10,1.2)
        f.text(w/2,331,t['measure_quality'],17.4,True,f.c['gold'],anchor='middle')
        f.text(w/2,355,t['quality_label'],16.2,color=f.c['muted'],anchor='middle')
        f.arrow(w/2,284,w/2,302,f.c['gold'],1.7,5)
    else:
        f.rect(20,307,w-40,60,f.c['blue_light'],r=10)
        f.text(w/2,331,t['no_quality'],17.5,True,f.c['blue'],anchor='middle',leading=24)


def feature_panel(f:Figure,t:dict[str,Any],training:bool,w:float,h:float,
                  colors:list[str]) -> None:
    f.text(22,34,t['feature_title'],24,True,maxw=w-44)
    f.text(22,62,t['feature_subtitle'],17.0,color=f.c['muted'])
    blockcolors=[f.c['teal'],f.c['blue'],f.c['purple'],f.c['coral']]
    centers=[73,155,237,319]
    for k,cx in enumerate(centers):
        f.rect(cx-33,85,66,60,blend(blockcolors[k],'#FFFFFF',.93),r=10)
        feature_icon(f,cx,115,k,blockcolors[k])
        f.text(cx,166,t['feature_labels'][k],15.0,True,blockcolors[k],anchor='middle',leading=21)
    f.line(371,92,371,287,f.c['line'],1.1)
    if training:
        mini_document(f,394,94,36,47,f.c['gold'])
        f.text(412,166,t['label_header'],14.6,True,f.c['gold'],anchor='middle',leading=21)
    else:
        f.circle(412,117,21,f.c['soft'],f.c['gray'],1.3,dash=True)
        f.line(399,130,425,104,f.c['gray'],1.8)
        f.text(412,166,t['absent_label_header'],14.6,False,f.c['gray'],anchor='middle',leading=21)
    vals=[[[.7,.2,.1],[.2,.6,.8],[.5,.2,.7],[.4,.7,.2]],
          [[.1,.8,.1],[.3,.7,.6],[.2,.6,.5],[.4,.7,.2]],
          [[.2,.3,.5],[.4,.4,.7],[.8,.3,.4],[.4,.7,.2]],
          [[.5,.4,.1],[.8,.2,.4],[.4,.8,.2],[.4,.7,.2]],
          [[.3,.2,.5],[.9,.1,.3],[.7,.4,.2],[.4,.7,.2]]]
    qualities=[.42,.7,.91,.30,.54]
    f.rect(42,202,318,87,f.c['white'],f.c['line'],4,1)
    for row in range(5):
        yy=207+row*16
        f.text(29,yy+10.6,chr(65+row),11.7,True,colors[row],anchor='middle')
        for k in range(4):
            for col in range(3):
                val=vals[row][k][col]
                if not training and k!=3: val=(val+.27)%1
                elif not training and k==3: val=[.63,.42,.78][col]
                f.rect(47+k*79+col*24.8,yy,23.3,12.6,
                       blend(blockcolors[k],'#FFFFFF',.20+.70*(1-val)),r=.7)
        if training:
            f.rect(387,yy,52,12.6,blend(f.c['gold'],'#FFFFFF',.89),r=2)
            f.rect(387,yy,52*qualities[row],12.6,blend(f.c['gold'],'#FFFFFF',.20),r=2)
    if not training:
        f.rect(387,206,53,79,'none',f.c['gray'],4,1,dash=True)
        f.line(399,258,428,229,f.c['gray'],1.5)
    # Feature-table message is separated from graphics by a consistent inset.
    f.rect(20,307,w-40,60,f.c['soft'] if training else f.c['teal_light'],r=10)
    if training:
        f.text(w/2,331,t['feature_note_training'],16.7,color=f.c['muted'],anchor='middle',leading=24)
    else:
        lock(f,46,334,29,f.c['teal'])
        f.text(76,331,t['feature_note_application'],17,True,f.c['teal'],leading=24)


def learning_panel(f:Figure,t:dict[str,Any],w:float,h:float,colors:list[str]) -> None:
    f.text(22,34,t['learn_title'],23.0,True,maxw=w-44)
    f.text(22,62,t['learn_subtitle'],16.5,color=f.c['muted'])
    # Three candidate cards before/after a schematic tree-based ranker.
    for side,order in [(25,[0,2,1]),(347,[2,1,0])]:
        for j,idx in enumerate(order):
            yy=103+j*31
            f.rect(side,yy,121,25,blend(colors[idx],'#FFFFFF',.92),blend(colors[idx],'#FFFFFF',.58),6,1)
            f.circle(side+16,yy+12.5,7,colors[idx])
            f.text(side+32,yy+18,'Candidate '+chr(65+idx),12.7,True,colors[idx])
    # Center the tree-based ranker between the before/after candidate stacks.
    f.arrow(155,145,180,145,f.c['blue'],2,7)
    f.rect(188,94,117,99,f.c['blue_light'],blend(f.c['blue'],'#FFFFFF',.55),15,1.3)
    for i,(dx,dy) in enumerate([(0,0),(-24,10),(24,10)]):
        tree(f,246.5+dx,105+dy,52,f.c['blue'] if not i else blend(f.c['blue'],'#FFFFFF',.2))
    f.arrow(312,145,340,145,f.c['blue'],2,6)
    f.text(w/2,221,t['learn_order_note'],17.3,True,f.c['blue'],anchor='middle')
    # Reserve the left column for two lines; document icons begin at x=273.
    f.text(22,249,t['cv_note'],17.2,True,maxw=220,leading=23)
    for i in range(3):
        x=273+i*24
        f.rect(x,238,18,22,blend(f.c['blue'],'#FFFFFF',.86),f.c['blue'],3,1)
        for yy in [244,249,254]: f.line(x+4,yy,x+14,yy,f.c['blue'],1)
    f.line(354,239,354,265,f.c['gray'],1.1)
    f.rect(411,238,26,26,f.c['purple_light'],f.c['purple'],4,1.2)
    for yy in [245,251,257]: f.line(417,yy,431,yy,f.c['purple'],1.2)
    f.text(310,287,t['train_short'],12.7,color=f.c['muted'],anchor='middle')
    f.text(424,287,t['heldout_short'],12.7,color=f.c['muted'],anchor='middle')
    f.rect(20,307,w-40,60,f.c['blue_light'],blend(f.c['blue'],'#FFFFFF',.65),10,1.2)
    lock(f,46,333,30,f.c['blue'])
    f.text(76,332,t['model_saved'],17.6,True,f.c['blue'])
    f.text(76,354,t['model_refit'],15.6,color=f.c['muted'])


def ranking_panel(f:Figure,t:dict[str,Any],w:float,h:float,colors:list[str]) -> None:
    f.text(22,34,t['rank_title'],24,True,maxw=w-44)
    f.text(22,62,t['rank_subtitle'],17.0,color=f.c['muted'])
    f.text(121,97,t['predicted_order'],16.0,True,color=f.c['teal'],anchor='middle')
    order=[2,0,4,1,3]
    for rank,idx in enumerate(order):
        yy=113+rank*34
        co=f.c['teal'] if rank==0 else f.c['muted']
        f.rect(24,yy,208,28,f.c['teal_light'] if rank==0 else f.c['soft'],
               f.c['teal'] if rank==0 else f.c['line'],7,1.4 if rank==0 else 1)
        f.text(41,yy+19,str(rank+1),15.5,True,co,anchor='middle')
        f.circle(68,yy+14,8,colors[idx])
        f.text(88,yy+19,chr(65+idx),14,True,co)
        f.rect(114,yy+9,102,10,blend(f.c['teal'],'#FFFFFF',.91),r=2)
        f.rect(114,yy+9,[100,78,62,45,28][rank],10,
               f.c['teal'] if rank==0 else blend(f.c['teal'],'#FFFFFF',.46),r=2)
    f.rect(280,108,190,165,f.c['teal_light'],blend(f.c['teal'],'#FFFFFF',.61),16,1.3)
    f.rect(315,98,120,24,f.c['teal'],r=8)
    f.text(375,115,t['top_choice'],11.6,True,'#FFFFFF',anchor='middle')
    network(f,375,201,130,f.c['teal'],2,False)
    arrow_path(f,[(235,127),(259,127),(259,184),(277,184)],f.c['teal'],2.2,head=6)
    f.text(375,293,t['chosen_label'],16.9,True,f.c['teal'],anchor='middle')
    f.rect(20,315,w-40,39,f.c['teal_light'],r=10)
    checkmark(f,44,334,21,f.c['teal'])
    f.text(70,341,t['select_note'],17.7,True,f.c['teal'])
    f.text(w/2,375,t['export_note'],14.6,color=f.c['muted'],anchor='middle')


def build(cfg:dict[str,Any],text:dict[str,Any],out:Path,font_dir:Path|None,dpi:int) -> dict[str,Any]:
    out.mkdir(parents=True,exist_ok=True)
    fonts={False:find_font(cfg['fonts']['regular'],cfg['font_family'],False,font_dir),
           True:find_font(cfg['fonts']['bold'],cfg['font_family'],True,font_dir)}
    f=Figure(cfg,out,fonts); t=text
    co=f.c
    colors=[co['blue'],co['purple'],co['teal'],co['coral'],co['gold']]
    # Header hierarchy: brand, purpose, one-line explanation.
    f.text(64,99,t['brand'],72,True,co['teal'])
    f.line(390,40,390,115,co['line'],2)
    f.text(428,83,t['headline'],40,True)
    f.text(430,118,t['expansion'],18.5,color=co['muted'])
    # Miniature decision motif; no raster logo or external icon assets.
    for i in [2,1,0]:
        f.rect(1780+11*i,54-7*i,66,61,'#FFFFFF',blend(co['blue'],'#FFFFFF',.61),8,1.2)
        network(f,1813+11*i,84-7*i,48,colors[i],i)
    f.arrow(1881,82,1931,82,co['teal'],2.4,8)
    f.rect(1941,42,96,81,co['teal_light'],blend(co['teal'],'#FFFFFF',.60),14,1.2)
    network(f,1989,83,70,co['teal'],2)
    f.text(66,162,t['subtitle'],23,color=co['muted'])
    f.line(64,187,2056,187,co['line'],1.3)
    f.text(80,218,t['upstream_label'],15,True,co['muted'])
    f.text(1000,218,t['amiga_label'],15,True,co['teal'])

    # Distinct two-lane narrative; all cards share the same feature semantics.
    f.rect(48,236,2024,478,['#F1F6FD','#F8FBFE'],blend(co['blue'],'#FFFFFF',.75),24,1.3)
    f.rect(48,826,2024,478,['#EEF9F6','#F9FCFB'],blend(co['teal'],'#FFFFFF',.71),24,1.3)
    for yy,accent,num,title,subtitle in [(236,co['blue'],'1','training_title','training_subtitle'),
                                       (826,co['teal'],'2','application_title','application_subtitle')]:
        f.circle(88,272+yy-236,21,accent)
        f.text(88,280+yy-236,num,23,True,'#FFFFFF',anchor='middle')
        f.text(125,270+yy-236,t[title],28,True,accent)
        f.text(125,297+yy-236,t[subtitle],17.5,color=co['muted'])

    xs=[80,500,1000,1540]; widths=[345,430,470,500]; height=382
    # Scientific boundary between generation and AMIGA feature/ranking steps.
    f.path('M 965 240 L 965 705',co['line'],1.6,dash=True)
    f.path('M 965 830 L 965 1295',co['line'],1.6,dash=True)
    for training,y in [(True,315),(False,905)]:
        for i in range(3):
            f.arrow(xs[i]+widths[i]+9,y+170,xs[i+1]-10,y+170,
                    co['blue'] if training else co['teal'],2.4,9)
        with card(f,('training' if training else 'application')+'-data',xs[0],y,widths[0],height):
            data_panel(f,t,training,widths[0],height)
        with card(f,('training' if training else 'application')+'-candidates',xs[1],y,widths[1],height):
            candidates_panel(f,t,training,widths[1],height,colors)
        with card(f,('training' if training else 'application')+'-features',xs[2],y,widths[2],height):
            feature_panel(f,t,training,widths[2],height,colors)
        with card(f,('training' if training else 'application')+'-decision',xs[3],y,widths[3],height):
            if training: learning_panel(f,t,widths[3],height,colors)
            else: ranking_panel(f,t,widths[3],height,colors)
    # Gold-standard evidence terminates at candidate evaluation, not the optimizer.
    arrow_path(f,[(405,629),(455,629),(455,650),(518,650)],co['gold'],2.2,True,7)

    # Across-row transfer is a model/schema artifact, never a labelled table.
    f.arrow(1790,700,1790,744,co['blue'],2.6,9)
    f.rect(1513,747,554,64,'#FFFFFF',blend(co['blue'],'#FFFFFF',.43),15,1.4)
    lock(f,1545,775,32,co['blue'])
    f.text(1578,773,t['transfer_title'],18.6,True,co['blue'])
    f.text(1578,797,t['transfer_subtitle'],16,color=co['muted'])
    f.arrow(1790,814,1790,899,co['teal'],2.6,9)

    # A single explicit boundary note replaces technical clutter in the lanes.
    f.circle(96,771,15,co['gold_light'],blend(co['gold'],'#FFFFFF',.4),1.1)
    mini_document(f,89,761,14,20,co['gold'])
    f.text(125,778,t['quality_boundary'],21.6,True,co['gold'])
    f.line(64,1325,2056,1325,co['line'],1.3)
    f.text(64,1356,t['footer'],20.3,True)
    f.text(64,1384,t['schematic_note'],14.6,color=co['muted'])
    f.text(2056,1384,t['revision_note'],14.1,color=co['muted'],anchor='end')

    checks=f.finish(dpi)
    checks['scientific_scope']='Reusable AMIGA core, not the complete amiga-exp study.'
    checks['sources_revision']=cfg['commit']
    checks['fonts']={('bold' if b else 'regular'):{'filename':p.name,'sha256':sha256(p)} for b,p in fonts.items()}
    checks['sources_sha256']={name:sha256(ROOT/name) for name in
                              ('build_figure.py','vector_engine.py','figure_config.json','figure_text.json')}
    checks['artifacts']={p.name:sha256(p) for p in sorted(out.glob(cfg['output_basename']+'*')) if p.is_file()}
    (out/'layout_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    environment=dict(python=platform.python_version(),platform=platform.platform(),
                     dependencies={name:importlib.metadata.version(name) for name in
                                   ('reportlab','PyMuPDF','fonttools','pillow','charset-normalizer')},
                     note='Binary regeneration depends on the recorded dependency and font versions.')
    (out/'build_environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    if checks['text_box_overlaps'] or checks['text_outside_regions'] or checks['raster_images_in_pdf']:
        print(json.dumps(checks,indent=2),file=sys.stderr)
        raise RuntimeError('Figure failed layout/vector checks; inspect layout_checks.json')
    return checks


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',type=Path,default=ROOT)
    p.add_argument('--font-dir',type=Path)
    p.add_argument('--dpi',type=int,default=300)
    a=p.parse_args()
    if not 72<=a.dpi<=1200: p.error('--dpi must lie between 72 and 1200')
    cfg=json.loads((ROOT/'figure_config.json').read_text(encoding='utf-8'))
    text=json.loads((ROOT/'figure_text.json').read_text(encoding='utf-8'))
    checks=build(cfg,text,a.out_dir.resolve(),a.font_dir,a.dpi)
    print(f"Created {cfg['output_basename']}: {checks['text_elements']} text elements; "
          f"{checks['vector_primitives']} vector primitives; no raster images in PDF.")
    return 0

if __name__=='__main__':
    raise SystemExit(main())
