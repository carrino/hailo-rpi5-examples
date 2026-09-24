#!/usr/bin/env python3

import argparse, gi, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
gi.require_version('Gst', '1.0')

import cv2
import numpy as np

from gi.repository import Gst
Gst.init(None)

from rpi_hardware_pwm import HardwarePWM

pwm = HardwarePWM(pwm_channel=2, hz=10000, chip=0)

CAMERA = "/dev/video0"
HEF = "/usr/local/hailo/resources/models/hailo8l/yolov8s.hef"
SO  = "/usr/local/hailo/resources/so/libyolo_hailortpp_postprocess.so"

ALPHA = 0.25
MIN_CONFIDENCE = 0.35
ema_cx = None

# Maps where the person is in the camera image to where the eyes should point.
# Each pair is (cx, duty): cx is the person's center in the image (0.0 = left edge,
# 1.0 = right edge), duty is the PWM duty cycle (0-100) that aims the eyes at them.
# Between pairs the duty is linearly interpolated; outside them it is extrapolated
# from the nearest pair. Use --calibrate to find these. The default is the original
# duty = cx * 100 behavior. Swap the duties to flip direction.
CALIBRATION = [(0.0, 0.0), (1.0, 100.0)]
# Hard limits so a bad calibration can't drive the eyes past their mechanical range.
DUTY_MIN = 0.0
DUTY_MAX = 100.0

DEBUG_PORT = 8080   # live view at http://<pi>:8080/ ; --debug-port 0 turns it off

current_duty = 50.0
calibrating = False
log_frames = True
save_enabled = False
debug_last_request = 0.0   # when a viewer last asked for a frame

# Latest frame + detections, shared from the GStreamer probe to the debug view/saver threads.
debug_lock = threading.Lock()
debug_seq = 0
debug_snap = None   # (frame_rgb, dets, best, ema_cx, duty)
debug_jpeg = (-1, None)   # (seq, jpeg bytes) cache of the last rendered snap

def cx_to_duty(cx):
    pts = sorted(CALIBRATION)
    if len(pts) == 1:
        return pts[0][1]
    i = 1
    while i < len(pts) - 1 and cx > pts[i][0]:
        i += 1
    (x0, d0), (x1, d1) = pts[i - 1], pts[i]
    return d0 + (cx - x0) * (d1 - d0) / (x1 - x0)

def set_duty(duty):
    global current_duty
    current_duty = DUTY_MIN if duty < DUTY_MIN else (DUTY_MAX if duty > DUTY_MAX else duty)
    pwm.change_duty_cycle(float(current_duty))

def set_pos01(pin, pos):
    pos = 0.0 if pos < 0 else (1.0 if pos > 1.0 else pos)
    pi.hardware_PWM(pin, 1000, int(pos * 1_000_000))

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
    x = _first(b, ["xmin", "x", "get_x", "xmin", "get_xmin", "x_min", "get_x_min", "left", "get_left"])
    y = _first(b, ["ymin", "y", "get_y", "ymin", "get_ymin", "y_min", "get_y_min", "top", "get_top"])
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

    frame = grab_frame(buf, s, fw, fh) if debug_wanted() else None
    dets = []   # every person seen, (x, y, w, h, confidence), for the debug view

    if not objs:
        publish_debug(frame, dets, None)
        return Gst.PadProbeReturn.OK

    # largest bbox (don’t depend on labels until we confirm them)
    best = None; best_area = -1.0
    for det in objs:
        try:
            label = getattr(det, "get_label", lambda: "")()
            if label != "person":
                continue
            b = det.get_bbox()
            x, y, w, h = _bbox_xywh(b)
            c = getattr(det, "get_confidence", lambda: None)()
            if log_frames:
                print(f"[x] {x} {y} {h} {w} {c}")
            dets.append((x, y, w, h, c))
            if c < MIN_CONFIDENCE:
                continue
            area = w * h
            if area > best_area:
                best_area, best = area, (x, y, w, h)
        except Exception:
            continue

    if best:
        global ema_cx
        x, y, w, h = best
        cx = (x + 0.5 * w)
        ema_cx = cx if ema_cx is None else (ALPHA * cx + (1 - ALPHA) * ema_cx)
        if not calibrating:
            set_duty(cx_to_duty(ema_cx))
        if log_frames:
            print(f"{ema_cx:.4f} duty={current_duty:.1f}", flush=True)
    elif log_frames:
        print("-1.0", flush=True)

    publish_debug(frame, dets, best)
    return Gst.PadProbeReturn.OK


def debug_wanted():
    """Only copy frames while someone is watching or we're saving, so idle cost is nil."""
    return save_enabled or time.time() - debug_last_request < 5.0

def grab_frame(buf, s, fw, fh):
    """Copy the RGB frame out of the GStreamer buffer (the 640x640 image the model saw)."""
    if s.get_value("format") != "RGB":
        return None
    ok, mi = buf.map(Gst.MapFlags.READ)
    if not ok:
        return None
    try:
        return np.frombuffer(mi.data, dtype=np.uint8, count=fw * fh * 3).reshape(fh, fw, 3).copy()
    finally:
        buf.unmap(mi)

def publish_debug(frame, dets, best):
    global debug_seq, debug_snap
    if frame is None:
        return
    with debug_lock:
        debug_seq += 1
        debug_snap = (frame, dets, best, ema_cx, current_duty)

def render(snap):
    frame, dets, best, ema, duty = snap
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (640, 480))   # undo the 640x480 -> 640x640 squash so people look normal
    H, W = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # cx grid, so you can read off where a person is for CALIBRATION
    for i in range(1, 10):
        X = int(i / 10 * W)
        cv2.line(img, (X, 0), (X, H), (90, 90, 90), 1)
        cv2.putText(img, f"{i / 10:.1f}", (X + 2, H - 6), font, 0.4, (220, 220, 220), 1)

    # boxes: green = the one being tracked, yellow = candidate, red = below MIN_CONFIDENCE
    for (x, y, w, h, c) in dets:
        if best and (x, y, w, h) == best:
            color = (0, 255, 0)
        elif c is not None and c >= MIN_CONFIDENCE:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)
        p1 = (int(x * W), int(y * H)); p2 = (int((x + w) * W), int((y + h) * H))
        cv2.rectangle(img, p1, p2, color, 2)
        cv2.putText(img, f"{c:.2f}" if c is not None else "?", (p1[0] + 2, p1[1] + 16), font, 0.5, color, 1)

    if best:
        X = int((best[0] + 0.5 * best[2]) * W)
        cv2.line(img, (X, 0), (X, H), (0, 255, 0), 1)
    if ema is not None:
        X = int(ema * W)
        cv2.line(img, (X, 0), (X, H), (255, 0, 255), 3)

    ema_txt = f"{ema:.3f}" if ema is not None else "-"
    mode = "CALIBRATE " if calibrating else ""
    text = f"{mode}eyes cx={ema_txt} duty={duty:.1f} people={len(dets)}"
    cv2.putText(img, text, (8, 22), font, 0.6, (0, 0, 0), 4)
    cv2.putText(img, text, (8, 22), font, 0.6, (255, 255, 255), 1)
    return img

def latest_jpeg():
    """Return (seq, jpeg bytes) of the newest frame, rendering it at most once."""
    global debug_jpeg, debug_last_request
    debug_last_request = time.time()
    with debug_lock:
        seq, snap = debug_seq, debug_snap
        if snap is None or debug_jpeg[0] == seq:
            return debug_jpeg
    ok, enc = cv2.imencode(".jpg", render(snap), [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return debug_jpeg
    with debug_lock:
        if seq > debug_jpeg[0]:
            debug_jpeg = (seq, enc.tobytes())
        return debug_jpeg


class DebugHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            body = b"<html><body style='margin:0;background:#000'><img src='/stream' style='width:100%'></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/snapshot.jpg":
            _, jpg = latest_jpeg()
            if jpg is None:
                self.send_error(503, "no frame yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.end_headers()
            self.wfile.write(jpg)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last = -1
            try:
                while True:
                    seq, jpg = latest_jpeg()
                    if jpg is not None and seq != last:
                        last = seq
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                        self.wfile.write(jpg + b"\r\n")
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass

def save_loop(save_dir, every):
    os.makedirs(save_dir, exist_ok=True)
    last = -1
    while True:
        time.sleep(every)
        seq, jpg = latest_jpeg()
        if jpg is None or seq == last:
            continue
        last = seq
        name = time.strftime("%Y%m%d-%H%M%S") + f"-{seq:07d}.jpg"
        with open(os.path.join(save_dir, name), "wb") as f:
            f.write(jpg)

def calibrate_loop():
    print("calibrate: type a duty cycle (0-100) + enter to move the eyes. Stand somewhere, read your cx", flush=True)
    print("off the debug view, adjust duty until the eyes look at you, and write down (cx, duty).", flush=True)
    for line in sys.stdin:
        try:
            d = float(line)
        except ValueError:
            print("not a number", flush=True)
            continue
        set_duty(d)
        ema_txt = f"{ema_cx:.3f}" if ema_cx is not None else "-"
        print(f"duty={current_duty:.1f}  person cx={ema_txt}  -> ({ema_txt}, {current_duty:.1f})", flush=True)


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
    global calibrating, log_frames, save_enabled

    ap = argparse.ArgumentParser(description="Halloween eyes: follow people with the eyes")
    ap.add_argument("--debug-port", type=int, default=DEBUG_PORT,
                    help=f"serve a live annotated view at http://<pi>:PORT/ (default {DEBUG_PORT}, 0 = off)")
    ap.add_argument("--save-dir", default=None, help="save annotated frames to this directory")
    ap.add_argument("--save-every", type=float, default=1.0, help="seconds between saved frames (default 1)")
    ap.add_argument("--calibrate", action="store_true",
                    help="don't track; set the duty cycle by typing numbers so you can build CALIBRATION")
    args = ap.parse_args()

    calibrating = args.calibrate
    log_frames = not args.calibrate
    save_enabled = args.save_dir is not None

    pwm.start(current_duty)

    if args.debug_port:
        srv = ThreadingHTTPServer(("0.0.0.0", args.debug_port), DebugHandler)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"debug view on http://0.0.0.0:{args.debug_port}/", flush=True)
    if args.save_dir is not None:
        threading.Thread(target=save_loop, args=(args.save_dir, args.save_every), daemon=True).start()
    if args.calibrate:
        threading.Thread(target=calibrate_loop, daemon=True).start()

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
    hailo_filt.set_property("config-path", "/home/pi/hailo-rpi5-examples/yolo_person.json")

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
        pwm.stop()

if __name__ == "__main__":
    main()
