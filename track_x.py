#!/usr/bin/env python3
import gi, sys
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)

CAMERA = "/dev/video0"
HEF = "/usr/local/hailo/resources/models/hailo8l/yolov8s.hef"
SO  = "/usr/local/hailo/resources/so/libyolo_hailortpp_postprocess.so"

PERSON = "person"
ALPHA = 0.25
ema_cx = None

# add this helper near the top of your file
def _get(obj, *names):
    """Return attribute or zero-arg method value for the first name that exists."""
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            return v() if callable(v) else v
    raise AttributeError(f"none of {names} on {type(obj)}")

def _val(obj, name):
    v = getattr(obj, name)
    return v() if callable(v) else v

def _first(obj, names, default=None):
    for n in names:
        if hasattr(obj, n):
            try:
                return float(_val(obj, n))
            except Exception:
                pass
    if default is not None:
        return default
    raise AttributeError(f"None of {names} on {type(obj)}")

def _bbox_xywh(b):
    # Try direct x/y/w/h
    x = _first(b, ["x", "get_x", "xmin", "get_xmin", "x_min", "get_x_min", "left", "get_left"])
    y = _first(b, ["y", "get_y", "ymin", "get_ymin", "y_min", "get_y_min", "top", "get_top"])
    # Prefer width/height if present; else derive from right/bottom
    w = _first(b, ["width", "get_width", "w", "get_w"], default=None)
    h = _first(b, ["height", "get_height", "h", "get_h"], default=None)
    if w is None or h is None:
        rx = _first(b, ["right", "get_right", "xmax", "get_xmax", "x_max", "get_x_max"])
        by = _first(b, ["bottom", "get_bottom", "ymax", "get_ymax", "y_max", "get_y_max"])
        w = rx - x if w is None else w
        h = by - y if h is None else h
    return x, y, w, h

def on_probe(pad, info):
    import hailo
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK

    # Frame size from caps
    caps = pad.get_current_caps() or pad.get_allowed_caps()
    try:
        s = caps.get_structure(0)
        fw = int(s.get_value("width")); fh = int(s.get_value("height"))
    except Exception:
        return Gst.PadProbeReturn.OK

    try:
        roi = hailo.get_roi_from_buffer(buf)
    except Exception:
        return Gst.PadProbeReturn.OK

    objs = list(roi.get_objects_typed(hailo.HAILO_DETECTION))

    if not objs:
        return Gst.PadProbeReturn.OK

    # largest bbox (don’t depend on labels until we confirm them)
    best = None; best_area = -1.0
    for det in objs:
        try:
            b = det.get_bbox()
            print("[det]", det, flush=True)
            print("[b]", b, flush=True)
            x, y, w, h = _bbox_xywh(b)
            area = w * h
            if area > best_area:
                best_area, best = area, (x, y, w, h)
        except Exception:
            continue

    if best:
        global ema_cx
        x, y, w, h = best
        cx = (x + 0.5 * w) / float(fw)
        cx = 0.0 if cx < 0 else (1.0 if cx > 1.0 else cx)
        ALPHA = 0.25
        ema_cx = cx if ema_cx is None else (ALPHA * cx + (1 - ALPHA) * ema_cx)
        print(f"{ema_cx:.4f}", flush=True)

    return Gst.PadProbeReturn.OK



#def on_probe(pad, info):
#    import hailo
#    buf = info.get_buffer()
#    if not buf:
#        return Gst.PadProbeReturn.OK
#
#    # Get frame size from caps
#    caps = pad.get_current_caps() or pad.get_allowed_caps()
#    try:
#        s = caps.get_structure(0)
#        fw = int(s.get_value('width')); fh = int(s.get_value('height'))
#    except Exception:
#        return Gst.PadProbeReturn.OK
#
#    try:
#        roi = hailo.get_roi_from_buffer(buf)
#    except Exception:
#        return Gst.PadProbeReturn.OK
#
#    objs = list(roi.get_objects_typed(hailo.HAILO_DETECTION))
#    # periodic debug
#    if not hasattr(on_probe, "_f"): on_probe._f = 0
#    on_probe._f += 1
#    if on_probe._f % 30 == 0:
#        labels_seen = {}
#        for d in objs[:10]:
#            lbl = d.get_label() or "<none>"
#            labels_seen[lbl] = labels_seen.get(lbl, 0) + 1
#        print(f"[dbg] dets={len(objs)} labels={labels_seen}", flush=True)
#
#    if not objs:
#        return Gst.PadProbeReturn.OK
#
#    # choose largest bbox
#    best = None; best_area = -1.0
#    for det in objs:
#        b = det.get_bbox()
#        area = float(b.width) * float(b.height)
#        if area > best_area:
#            best_area = area; best = b
#
#    if best:
#        cx = (best.x + 0.5 * best.width) / float(fw)
#        cx = 0.0 if cx < 0 else (1.0 if cx > 1.0 else cx)
#        # EMA smoothing
#        global ema_cx
#        ema_cx = cx if ema_cx is None else (ALPHA * cx + (1 - ALPHA) * ema_cx)
#        print(f"{ema_cx:.4f}", flush=True)
#
#    return Gst.PadProbeReturn.OK


#def on_probe(pad, info):
#    import hailo
#    buf = info.get_buffer()
#    if not buf:
#        return Gst.PadProbeReturn.OK
#
#    # Get frame size from negotiated caps on this pad
#    caps = pad.get_current_caps() or pad.get_allowed_caps()
#    try:
#        s = caps.get_structure(0)
#        fw = s.get_value('width')
#        fh = s.get_value('height')
#        fw = int(fw) if fw is not None else None
#        fh = int(fh) if fh is not None else None
#    except Exception:
#        fw = fh = None
#
#    try:
#        roi = hailo.get_roi_from_buffer(buf)
#    except Exception:
#        return Gst.PadProbeReturn.OK
#
#    # Fall back if caps didn't give us dims (shouldn’t happen)
#    if not fw or not fh:
#        try:
#            # some builds expose roi.get_stream_info().get_width()/get_height()
#            si = roi.get_stream_info()
#            fw = fw or int(getattr(si, 'get_width')())
#            fh = fh or int(getattr(si, 'get_height')())
#        except Exception:
#            return Gst.PadProbeReturn.OK
#
#    # Pick largest "person" and emit normalized cx
#    global ema_cx
#    best = None; best_area = -1.0
#    for det in roi.get_objects_typed(hailo.HAILO_DETECTION):
#        if det.get_label() != "person":
#            continue
#        b = det.get_bbox()  # has x,y,width,height in pixels
#        area = float(b.width) * float(b.height)
#        if area > best_area:
#            best_area, best = area, b
#
#    if best:
#        cx = (best.x + 0.5*best.width) / float(fw)
#        cx = 0.0 if cx < 0 else (1.0 if cx > 1.0 else cx)
#        ema_cx = cx if ema_cx is None else (ALPHA*cx + (1-ALPHA)*ema_cx)
#        print(f"{ema_cx:.4f}", flush=True)
#
#    return Gst.PadProbeReturn.OK


def mk(name):
    e = Gst.ElementFactory.make(name)
    if not e:
        print(f"Failed to create element: {name}", file=sys.stderr); sys.exit(1)
    return e

def link_chain(elems):
    for a, b in zip(elems, elems[1:]):
        if not a.link(b):
            print(f"Link failed: {a.name} → {b.name}", file=sys.stderr)
            sys.exit(1)

def main():
    # elements matching the pipeline that linked for you
    src = mk("v4l2src"); src.set_property("device", CAMERA); src.set_property("io-mode", 2); src.set_property("do-timestamp", True)
    caps_mjpg = mk("capsfilter"); caps_mjpg.set_property("caps", Gst.Caps.from_string("image/jpeg,width=640,height=480,framerate=30/1"))
    jpegdec = mk("jpegdec")
    vconv = mk("videoconvert")
    vscale = mk("videoscale")
    caps_rgb_sq = mk("capsfilter"); caps_rgb_sq.set_property("caps", Gst.Caps.from_string("video/x-raw,format=RGB,width=640,height=640"))
    q = mk("queue"); q.set_property("max-size-buffers", 3); q.set_property("leaky", 2)
    hailo_net = mk("hailonet"); hailo_net.set_property("hef-path", HEF)
    hailo_filt = mk("hailofilter")
    hailo_filt.set_property("function-name", "yolov8s")
    hailo_filt.set_property("so-path", "/usr/local/hailo/resources/so/libyolo_hailortpp_postprocess.so")
    hailo_filt.set_property("config-path", "/home/pi/yolo_person.json")

    sink = mk("fakesink"); sink.set_property("sync", False)

    pipe = Gst.Pipeline.new("pipe")
    for e in (src, caps_mjpg, jpegdec, vconv, vscale, caps_rgb_sq, q, hailo_net, hailo_filt, sink):
        pipe.add(e)

    link_chain([src, caps_mjpg, jpegdec, vconv, vscale, caps_rgb_sq, q, hailo_net, hailo_filt, sink])

    # tap detections after postproc
    hailo_filt.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, on_probe)

    pipe.set_state(Gst.State.PLAYING)
    bus = pipe.get_bus()
    try:
        while True:
            msg = bus.timed_pop_filtered(500 * Gst.MSECOND, Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if not msg:
                continue
            if msg.type == Gst.MessageType.ERROR:
                err, dbg = msg.parse_error()
                print(f"GStreamer ERROR: {err}; {dbg}", file=sys.stderr)
                break
            if msg.type == Gst.MessageType.EOS:
                break
    except KeyboardInterrupt:
        pass
    finally:
        pipe.set_state(Gst.State.NULL)

if __name__ == "__main__":
    main()
