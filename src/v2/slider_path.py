"""Evaluate native slider curves, including pixel-length trimming/extension.

Independent numerical implementation following osu!'s SliderPath semantics.
Curves are approximated to sub-pixel precision; output placement reserves an
additional two pixels. This is not an implementation of stacking or hit rules.
"""
from __future__ import annotations
import math
from dataclasses import replace
import numpy as np
from .events import EventKind


def _bezier(points, depth=0):
    if len(points) <= 2:
        return points
    # Bound flatness by second differences, including collinear backtracking.
    degree = len(points)-1
    curvature_bound = degree*(degree-1)*np.max(np.linalg.norm(np.diff(points, n=2, axis=0), axis=1))
    if curvature_bound/8 <= .05:
        return np.array([points[0], points[-1]])
    if depth >= 18:
        raise ValueError('slider curve exceeds approximation budget')
    levels = [points]
    while len(levels[-1]) > 1:
        levels.append((levels[-1][:-1] + levels[-1][1:]) / 2)
    left = np.array([p[0] for p in levels])
    right = np.array([p[-1] for p in levels[::-1]])
    return np.concatenate((_bezier(left, depth+1)[:-1], _bezier(right, depth+1)))


def _multi_bezier(points):
    parts = []
    start = 0
    for i in range(1, len(points)):
        if np.array_equal(points[i], points[i-1]):
            parts.append(_bezier(points[start:i]))
            start = i
    parts.append(_bezier(points[start:]))
    return np.concatenate(parts)


def _arc(points):
    a, b, c = points
    ab, ac = b-a, c-a
    cross = ab[0]*ac[1]-ab[1]*ac[0]
    if abs(cross) < 1e-7:
        return _multi_bezier(points)
    center = a + np.array([ac[1]*ab.dot(ab)-ab[1]*ac.dot(ac),
                            ab[0]*ac.dot(ac)-ac[0]*ab.dot(ab)]) / (2*cross)
    radius = np.linalg.norm(a-center)
    angles = np.arctan2(points[:,1]-center[1], points[:,0]-center[0])
    sweep = (angles[2]-angles[0]) % (2*math.pi)
    if (angles[1]-angles[0]) % (2*math.pi) > sweep:
        sweep -= 2*math.pi
    step = 2*math.acos(max(-1., 1-.05/max(radius, .05)))
    count = max(2, math.ceil(abs(sweep)/max(step, 1e-7))+1)
    if count > 1000:  # native implementation also falls back on enormous arcs
        return _multi_bezier(points)
    theta = np.linspace(angles[0], angles[0]+sweep, count)
    return center + radius*np.stack((np.cos(theta), np.sin(theta)), axis=1)


def _catmull(points):
    result = []
    # Native Catmull uses 50 steps per segment.
    for i in range(len(points)-1):
        p1 = points[max(0,i-1)]; p2=points[i]; p3=points[i+1]
        p4 = points[i+2] if i+2<len(points) else 2*p3-p2
        t=np.linspace(0,1,101)[:,None]
        result.append(.5*(2*p2+(-p1+p3)*t+(2*p1-5*p2+4*p3-p4)*t*t+
                           (-p1+3*p2-3*p3+p4)*t*t*t))
    return np.concatenate(result)


def slider_path(head, controls, curve_type, length):
    """Return the rendered centerline in serialized (integer) coordinates."""
    points=np.rint(np.array([head, *controls], dtype=np.float64))
    if len(points)<2 or not np.isfinite(points).all() or not math.isfinite(length) or length<=0:
        raise ValueError('invalid slider path')
    if len(points)>33 or np.abs(points).max()>1e6 or length>1e6:
        raise ValueError('slider exceeds path evaluation budget')
    kind=curve_type.upper()
    if kind=='L': path=points
    elif kind=='P' and len(points)==3: path=_arc(points)
    elif kind in ('B','P'): path=_multi_bezier(points)
    elif kind=='C': path=_catmull(points)
    else: raise ValueError('unsupported slider curve')
    keep=np.r_[True, np.linalg.norm(np.diff(path,axis=0),axis=1)>1e-9]
    path=path[keep]
    if len(path)<2:
        raise ValueError('degenerate slider has no travel direction')
    distances=np.r_[0., np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    if length < distances[-1]:
        i=int(np.searchsorted(distances,length,side='right'))
        end=path[i-1]+(path[i]-path[i-1])*(length-distances[i-1])/(distances[i]-distances[i-1])
        return np.concatenate((path[:i],end[None]))
    # Native repeated final controls suppress extension. Retain that behavior.
    if length>distances[-1] and not np.array_equal(points[-1],points[-2]):
        direction=(path[-1]-path[-2])/(distances[-1]-distances[-2])
        path=np.concatenate((path,(path[-1]+direction*(length-distances[-1]))[None]))
    return path


def circle_radius(cs):
    return 54.4-4.48*float(cs)


def event_path(event):
    if event.kind != EventKind.SLIDER:
        return np.rint([[event.x,event.y]])
    return slider_path((event.x,event.y),event.control_points,event.curve_type,event.slider_length)


def fit_event_to_playfield(event, cs):
    """Translate entire curves without changing timing, length, repeats or shape.

    An oversized shape is rejected instead of clipping its anchors or quietly
    replacing it with a straight slider. Integer translations survive export.
    """
    if event.kind==EventKind.SPINNER:
        return event
    path=event_path(event)
    margin=circle_radius(cs)+2.
    low=path.min(axis=0); high=path.max(axis=0)
    minimum=np.ceil(margin-low); maximum=np.floor(np.array([512.,384.])-margin-high)
    if (minimum>maximum).any():
        raise ValueError('rendered slider cannot fit inside playfield without changing its shape')
    shift=np.clip(np.zeros(2),minimum,maximum)
    return replace(event,x=round(event.x)+float(shift[0]),y=round(event.y)+float(shift[1]),
                   control_points=tuple((round(x)+float(shift[0]),round(y)+float(shift[1])) for x,y in event.control_points))
