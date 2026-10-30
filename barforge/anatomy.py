"""Image-based shape refinement of anatomical predictions, never of whole scene boxes.

The specialized anatomical model supplies genital locations. GrabCut separates their foreground from
surrounding drawing; contours and large background holes are retained for clipping actual bar rectangles.
The broader model supplies optional anus predictions and scene boundaries for background sampling only.
"""
from __future__ import annotations

from dataclasses import replace
import threading

import cv2
import numpy as np

from .models import Rect

_MAX_REGIONS = 32
_SEGMENT_SIDE = 640
_grabcut_lock = threading.Lock()  # OpenCV's GMM initializer uses a process-global random generator


def _canonical(label):
    return {"pussy":"vagina","nipple_f":"nipple"}.get(label,label)


def _iou(first,second):
    a,b = first.rect,second.rect
    intersection = max(0,min(a.x+a.width,b.x+b.width)-max(a.x,b.x))*max(0,min(a.y+a.height,b.y+b.height)-max(a.y,b.y))
    return intersection/max(1,a.width*a.height+b.width*b.height-intersection)


def _foreground(pixels, rect, context=None):
    """Return the central anatomical foreground and substantial background holes, or no reliable shape."""
    height,width = pixels.shape[:2]
    margin = max(12,round(min(rect.width,rect.height)*.15))
    left,top = max(0,rect.x-margin),max(0,rect.y-margin)
    right,bottom = min(width,rect.x+rect.width+margin),min(height,rect.y+rect.height+margin)
    if context is not None:
        # Scene boundaries constrain background sampling, never the foreground or the bar placement.
        left,top = max(left,min(rect.x,context.x)),max(top,min(rect.y,context.y))
        right = min(right,max(rect.x+rect.width,context.x+context.width))
        bottom = min(bottom,max(rect.y+rect.height,context.y+context.height))
    crop = pixels[top:bottom,left:right]
    h,w = crop.shape[:2]
    if min(h,w) < 4:
        return None
    scale = min(1.,_SEGMENT_SIDE/max(h,w))
    size = (max(4,round(w*scale)),max(4,round(h*scale)))
    roi = cv2.resize(crop,size,interpolation=cv2.INTER_AREA) if scale < 1 else crop.copy()
    sx,sy = roi.shape[1]/w,roi.shape[0]/h
    x0,y0 = max(0,round((rect.x-left)*sx)),max(0,round((rect.y-top)*sy))
    x1,y1 = min(roi.shape[1],round((rect.x+rect.width-left)*sx)),min(roi.shape[0],round((rect.y+rect.height-top)*sy))
    if x1-x0 < 3 or y1-y0 < 3:
        return None
    mask = np.full(roi.shape[:2],cv2.GC_BGD,np.uint8)
    mask[y0:y1,x0:x1] = cv2.GC_PR_FGD
    cx,cy = (x0+x1)//2,(y0+y1)//2
    dx,dy = max(1,round((x1-x0)*.12)),max(1,round((y1-y0)*.12))
    mask[max(y0,cy-dy):min(y1,cy+dy),max(x0,cx-dx):min(x1,cx+dx)] = cv2.GC_FGD
    if np.all(mask != cv2.GC_BGD):
        return None
    with _grabcut_lock:
        cv2.setRNGSeed(0)
        cv2.grabCut(roi,mask,None,np.zeros((1,65),np.float64),np.zeros((1,65),np.float64),
                    4,cv2.GC_INIT_WITH_MASK)
    foreground = np.isin(mask,(cv2.GC_FGD,cv2.GC_PR_FGD)).astype(np.uint8)
    foreground = cv2.morphologyEx(foreground,cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    _,labels,stats,_ = cv2.connectedComponentsWithStats(foreground,connectivity=8)
    target = int(labels[cy,cx])
    if target == 0:
        return None
    box_area = (x1-x0)*(y1-y0)
    keep = [target] + [index for index in range(1,len(stats)) if index != target
                       and stats[index,cv2.CC_STAT_AREA] >= max(12,box_area*.02)]
    component = np.isin(labels,keep).astype(np.uint8)
    area = int(component.sum())
    if area < max(8,box_area*.12):
        return None
    contours,hierarchy = cv2.findContours(component,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return None
    hierarchy = hierarchy[0]
    outers = [i for i,item in enumerate(hierarchy) if item[3] == -1]
    outer = max(outers,key=lambda i:cv2.contourArea(contours[i]))

    def coordinates(contour):
        simplified = cv2.approxPolyDP(contour,.7,True).reshape(-1,2)
        return tuple((float(x/sx+left),float(y/sy+top)) for x,y in simplified)

    outline = coordinates(contours[outer])
    if len(outline) < 3:
        return None
    islands = tuple(coordinates(contours[index]) for index in outers if index != outer)
    holes = tuple(coordinates(contour) for index,contour in enumerate(contours)
                  if hierarchy[index][3] in outers and cv2.contourArea(contour) > max(16,box_area*.02)
                  and min(cv2.minAreaRect(contour)[1]) > min(x1-x0,y1-y0)*.25)
    return (outline,tuple(hole for hole in holes if len(hole) >= 3),
            tuple(island for island in islands if len(island) >= 3))


def _merge_foregrounds(first, second):
    """Unite repeated predictions only when their actual foregrounds describe the same object."""
    if _canonical(first.label) != _canonical(second.label) or _iou(first,second) < .15:
        return None
    a,b = first.rect,second.rect
    left,top = min(a.x,b.x),min(a.y,b.y)
    right,bottom = max(a.x+a.width,b.x+b.width),max(a.y+a.height,b.y+b.height)
    origin = np.asarray((left,top))
    masks = []
    for region in (first,second):
        mask = np.zeros((bottom-top,right-left),np.uint8)
        for contour in (region.outline,)+region.islands:
            cv2.fillPoly(mask,[np.rint(np.asarray(contour)-origin).astype(np.int32)],1)
        for hole in region.holes:
            cv2.fillPoly(mask,[np.rint(np.asarray(hole)-origin).astype(np.int32)],0)
        masks.append(mask)
    overlap = cv2.countNonZero(cv2.bitwise_and(*masks))
    if overlap < .8*min(cv2.countNonZero(mask) for mask in masks):
        return None
    merged = cv2.bitwise_or(*masks)
    contours,hierarchy = cv2.findContours(merged,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]
    outers = [i for i,item in enumerate(hierarchy) if item[3] == -1]
    outer = max(outers,key=lambda i:cv2.contourArea(contours[i]))

    def coordinates(index):
        return tuple((float(x+left),float(y+top)) for x,y in contours[index].reshape(-1,2))

    x,y,w,h = cv2.boundingRect(merged)
    return replace(first,rect=Rect(x+left,y+top,w,h),outline=coordinates(outer),
                   islands=tuple(coordinates(i) for i in outers if i != outer),
                   holes=tuple(coordinates(i) for i,item in enumerate(hierarchy) if item[3] != -1))


def refine_anatomy(pixels, by_head, options, done, progress=None, cancelled=None):
    """Return foreground-shaped anatomy; broad genital/scene boxes never become censor targets."""
    from . import autocensor as ac
    height,width = pixels.shape[:2]
    primary_rows = by_head['anime-censor-v1-s']
    primary = ac._detections(primary_rows[ac._dedupe(primary_rows)],width,height)
    secondary_rows = by_head['erax-anti-nsfw-v11']
    secondary = ac._detections(secondary_rows[ac._dedupe(secondary_rows)],width,height)
    scenes = [item for item in secondary if item.kind == 'act' and options.include_acts
              and item.confidence >= options.act_confidence]
    candidates = primary + ([item for item in secondary if item.kind == 'anus'] if options.include_anus else [])
    candidates.sort(key=lambda item:(-item.confidence,item.rect.y,item.rect.x))
    limited = len(candidates) > _MAX_REGIONS
    candidates = candidates[:_MAX_REGIONS]
    total = done+len(candidates)
    if progress is not None:
        progress(done,total)
    regions = []
    failures = 0
    for region in candidates:
        if cancelled is not None and cancelled():
            raise ac.CancelledError()
        r = region.rect
        cx,cy = r.x+r.width/2,r.y+r.height/2
        contexts = [scene.rect for scene in scenes if scene.rect.x <= cx <= scene.rect.x+scene.rect.width
                    and scene.rect.y <= cy <= scene.rect.y+scene.rect.height]
        context = min(contexts,key=lambda item:item.width*item.height) if contexts else None
        shape = _foreground(pixels,r,context)
        if shape is None:
            failures += 1
        else:
            outline,holes,islands = shape
            points = np.concatenate([np.asarray(contour) for contour in (outline,)+islands])
            lo,hi = np.floor(points.min(axis=0)).astype(int),np.ceil(points.max(axis=0)).astype(int)+1
            refined = replace(region,rect=Rect(int(lo[0]),int(lo[1]),int(hi[0]-lo[0]),int(hi[1]-lo[1])),
                              outline=outline,holes=holes,islands=islands)
            regions.append(refined)
        done += 1
        if progress is not None:
            progress(done,total)
    # Combine repeated foregrounds before computing their angle; nested boxes must not make two grids.
    selected = []
    for region in sorted(regions,key=lambda item:(-item.confidence,item.rect.y,item.rect.x)):
        for index,old in enumerate(selected):
            merged = _merge_foregrounds(old,region)
            if merged is not None:
                selected[index] = merged
                break
        else:
            selected.append(region)
    selected = [region for region in selected if ac._passes(region,options)]
    selected.sort(key=lambda item:(item.rect.y,item.rect.x,item.label))
    notes = ["Полосы ограничены выделенной формой анатомии. Сцены используются только для контекста; "
             "общие боксы тела не закрашиваются."]
    if failures:
        notes.append(f"Не удалось уверенно выделить форму областей: {failures}. Проверьте пропуски вручную.")
    if limited:
        notes.append(f"Уточнение ограничено {_MAX_REGIONS} областями; проверьте оставшиеся вручную.")
    return selected,notes
