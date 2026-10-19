"""
ทำนายกิจกรรมแบบเรียลไทม์จาก phyphox แล้วแสดงผลบนหน้าเว็บ

วิธีใช้งาน
1. เปิดการทดลองใน phyphox (accelerometer + gyroscope) แล้วเปิด Allow remote access
2. รัน   python realtime_web.py
   (หรือใส่ URL ไปเลย: python realtime_web.py http://10.20.122.173:8080)
3. เบราว์เซอร์จะเปิดหน้า http://localhost:8000 ให้เอง ใส่ URL ของ phyphox แล้วกด "เริ่ม"
4. หยุดโปรแกรมด้วย Ctrl+C ในเทอร์มินัล

ขั้นตอนการทำนายเหมือนใน realtime_activity.ipynb ทุกอย่าง ต่างกันแค่ส่งผลไปแสดงบนเว็บแทนการ print
ใช้แค่ไลบรารีมาตรฐานของ Python + numpy
"""
import json
import sys
import threading
import time
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

# ===== ค่าที่ต้องตั้ง =====
DEFAULT_URL = "http://10.20.122.173:8080"   # URL ที่ phyphox แสดง (แก้ในหน้าเว็บได้)
PORT        = 8000                          # พอร์ตของหน้าเว็บ → http://localhost:8000
STD_DDOF    = 1                             # ส่วนเบี่ยงเบนมาตรฐาน: หารด้วย n − 1
# =========================

HERE       = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model2.npz"
HTML_PATH  = HERE / "web" / "index.html"

FS         = 50     # อัตราการเก็บค่า (ค่าต่อวินาที)
WIN        = 128    # ความยาวช่วงที่ใช้ทำนาย 1 ครั้ง (128 ค่า = 2.56 วินาที)
PRED_EVERY = 0.5    # ทำนายทุก 0.5 วินาที
VOTE_SEC   = 5.0    # สรุปผลจากช่วง 5 วินาทีล่าสุด
SHOW_EVERY = 1.0    # อัปเดตผลทุก 1 วินาที
SIGNAL_SEC = 5.0    # ความยาวกราฟสัญญาณบนหน้าเว็บ
TIMELINE_N = 60     # จำนวนผลย้อนหลังที่แสดงในแถบไทม์ไลน์

ACC = ["accX", "accY", "accZ"]
GYR = ["gyrX", "gyrY", "gyrZ"]

# ---------- 1. โหลดโมเดล ----------
m = np.load(MODEL_PATH, allow_pickle=False)
FEATURE_COLS = [str(c) for c in m["feature_cols"]]
MU, W, CENTROIDS = m["mu"], m["W"], m["centroids"]
CLASSES = np.array([str(c) for c in m["classes"]])
SCALE_MIN_ARR = m["scale_min"].astype(float)
SCALE_MAX_ARR = m["scale_max"].astype(float)
if SCALE_MIN_ARR.shape != (24,) or SCALE_MAX_ARR.shape != (24,):
    sys.exit("model2.npz ต้องมี scale_min และ scale_max อย่างละ 24 ค่า")

# ---------- 2. แปลงสัญญาณและทำนาย ----------
CHANNEL_INDEX = {"ax": 0, "ay": 1, "az": 2, "gx": 3, "gy": 4, "gz": 5}
STATS = {"mean": lambda v: v.mean(), "std": lambda v: v.std(ddof=STD_DDOF),
         "min": lambda v: v.min(), "max": lambda v: v.max()}


def window_to_features(win):
    """win ขนาด (128, 6) เรียง ax, ay, az, gx, gy, gz → ฟีเจอร์ 24 ค่า เรียงตาม FEATURE_COLS"""
    feats = []
    for col in FEATURE_COLS:
        ch, st = col.split("_")
        feats.append(STATS[st](win[:, CHANNEL_INDEX[ch]]))
    return np.array(feats)


def scale(f):
    return 2 * (f - SCALE_MIN_ARR) / (SCALE_MAX_ARR - SCALE_MIN_ARR) - 1


def predict_window(win):
    z = (scale(window_to_features(win)) - MU) @ W
    d = np.linalg.norm(CENTROIDS - z, axis=1)
    return str(CLASSES[np.argmin(d)])


# ---------- 3. รับข้อมูลจาก phyphox ----------
def get(url, path):
    with urllib.request.urlopen(url + path, timeout=3) as r:
        return json.loads(r.read())


def fetch_new(url, tname, names, since):
    """ค่าที่มีเวลามากกว่า since → (เวลา, ค่า 3 แกน)"""
    q = "&".join([f"{tname}={since:.6f}"] + [f"{n}={since:.6f}%7C{tname}" for n in names])
    buf = get(url, "/get?" + q)["buffer"]
    cols = [buf[tname]["buffer"]] + [buf[n]["buffer"] for n in names]
    n = min(len(c) for c in cols)
    arr = np.array([[np.nan if x is None else x for x in c[:n]] for c in cols], dtype=float).reshape(4, n)
    t, v = arr[0], arr[1:].T
    ok = np.isfinite(t) & np.isfinite(v).all(axis=1)
    return t[ok], v[ok]


def window_at(end, tA, A, tG, G):
    """สัญญาณ 2.56 วินาทีที่สิ้นสุด ณ เวลา end → (128, 6)"""
    grid = end - (WIN - 1) / FS + np.arange(WIN) / FS
    acc = np.column_stack([np.interp(grid, tA, A[:, i]) for i in range(3)])
    gyr = np.column_stack([np.interp(grid, tG, G[:, i]) for i in range(3)])
    return np.hstack([acc, gyr])


def signal_tail(now, tA, A, tG, G):
    """ขนาดของเวกเตอร์ |acc| และ |gyr| ช่วง 5 วินาทีล่าสุด (25 จุดต่อวินาที) สำหรับวาดกราฟ"""
    grid = np.arange(now - SIGNAL_SEC, now, 1 / 25)
    acc = np.column_stack([np.interp(grid, tA, A[:, i]) for i in range(3)])
    gyr = np.column_stack([np.interp(grid, tG, G[:, i]) for i in range(3)])
    return (np.round(np.linalg.norm(acc, axis=1), 3).tolist(),
            np.round(np.linalg.norm(gyr, axis=1), 3).tolist())


# ---------- 4. ทำนายแบบเรียลไทม์ (ทำงานเบื้องหลัง) ----------
class Session:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop_evt = threading.Event()
        self.thread = None
        self.state = self._blank("idle", "กด เริ่ม เพื่อเชื่อมต่อกับ phyphox", DEFAULT_URL)

    @staticmethod
    def _blank(status, message, url):
        return {"status": status, "message": message, "url": url, "t": 0.0,
                "current": None, "counts": {c: 0 for c in CLASSES}, "total": 0,
                "timeline": [], "acc": [], "gyr": []}

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.state))

    def update(self, **kw):
        with self.lock:
            self.state.update(kw)

    def start(self, url):
        self.stop()
        url = url.strip().rstrip("/") or DEFAULT_URL
        if not url.startswith("http"):
            url = "http://" + url
        self.stop_evt = threading.Event()
        with self.lock:
            self.state = self._blank("connecting", "กำลังเชื่อมต่อ phyphox …", url)
        self.thread = threading.Thread(target=self._run, args=(url, self.stop_evt), daemon=True)
        self.thread.start()

    def stop(self):
        if self.thread and self.thread.is_alive():
            self.stop_evt.set()
            self.thread.join(timeout=5)

    def _run(self, url, stop_evt):
        try:
            get(url, "/control?cmd=clear")
            get(url, "/control?cmd=start")
        except Exception as e:
            self.update(status="error", message=f"เชื่อมต่อ {url} ไม่ได้ — ตรวจว่าเปิด Allow remote access และอยู่ Wi-Fi เดียวกัน ({e})")
            return

        tA, A = np.empty(0), np.empty((0, 3))
        tG, G = np.empty(0), np.empty((0, 3))
        preds, timeline, next_pred, next_show = [], [], None, None
        self.update(status="collecting", message="กำลังรวบรวมข้อมูล")
        fails = 0
        while not stop_evt.is_set():
            try:
                t, v = fetch_new(url, "acc_time", ACC, tA[-1] if len(tA) else -1)
                tA, A = np.append(tA, t), np.vstack([A, v])
                t, v = fetch_new(url, "gyr_time", GYR, tG[-1] if len(tG) else -1)
                tG, G = np.append(tG, t), np.vstack([G, v])
                fails = 0
            except Exception as e:
                fails += 1
                if fails >= 10:
                    self.update(status="error", message=f"ขาดการเชื่อมต่อกับ phyphox ({e})")
                    return
                time.sleep(0.3)
                continue

            if len(tA) >= 2 and len(tG) >= 2:
                start = max(tA[0], tG[0]) + (WIN - 1) / FS       # เวลาแรกที่มีข้อมูลครบ 1 ช่วง
                now = min(tA[-1], tG[-1])
                if next_pred is None:
                    next_pred, next_show = start, start + VOTE_SEC
                while next_pred <= now:                           # ทำนายทุก 0.5 วินาที
                    preds.append((next_pred, predict_window(window_at(next_pred, tA, A, tG, G))))
                    next_pred += PRED_EVERY

                acc_sig, gyr_sig = signal_tail(now, tA, A, tG, G)
                if now >= next_show:                              # อัปเดตผลทุก 1 วินาที
                    recent = [p for tp, p in preds if tp > now - VOTE_SEC]
                    counts = {c: recent.count(c) for c in CLASSES}
                    top = max(counts, key=counts.get)
                    timeline = (timeline + [[round(float(now), 1), top]])[-TIMELINE_N:]
                    self.update(status="running", message="", t=float(now), current=top,
                                counts=counts, total=len(recent), timeline=timeline,
                                acc=acc_sig, gyr=gyr_sig)
                    while next_show <= now:
                        next_show += SHOW_EVERY
                else:
                    with self.lock:
                        self.state.update(t=float(now), acc=acc_sig, gyr=gyr_sig)
                        if self.state["status"] != "running":
                            left = max(0.0, next_show - now)
                            self.state["message"] = f"กำลังรวบรวมข้อมูล (อีก {left:.0f} วินาที)"

                keep_from = now - (VOTE_SEC + WIN / FS + 2)        # เก็บข้อมูลย้อนหลังไว้แค่พอใช้
                tA, A = tA[tA >= keep_from], A[tA >= keep_from]
                tG, G = tG[tG >= keep_from], G[tG >= keep_from]
                preds = [p for p in preds if p[0] > now - VOTE_SEC - 1]
            stop_evt.wait(0.2)

        try:
            get(url, "/control?cmd=stop")
        except Exception:
            pass
        self.update(status="stopped", message="หยุดการทำงานแล้ว")


SESSION = Session()


# ---------- 5. เว็บเซิร์ฟเวอร์ ----------
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False), "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
        elif self.path.startswith("/web/"):                      # ไฟล์ประกอบหน้าเว็บ เช่น cats.js
            f = (HERE / self.path.lstrip("/")).resolve()
            if f.is_file() and HTML_PATH.parent in f.parents:
                ctype = {".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml",
                         ".png": "image/png"}.get(f.suffix, "application/octet-stream")
                self._send(200, f.read_bytes(), ctype + "; charset=utf-8" if f.suffix in (".js", ".css") else ctype)
            else:
                self._send(404, "not found", "text/plain")
        elif self.path == "/api/state":
            self._json(SESSION.snapshot())
        elif self.path == "/api/info":
            self._json({"classes": CLASSES.tolist(), "k": int(W.shape[1]),
                        "vote_sec": VOTE_SEC, "pred_every": PRED_EVERY})
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/api/start":
            SESSION.start(body.get("url", ""))
            self._json({"ok": True})
        elif self.path == "/api/stop":
            SESSION.stop()
            self._json({"ok": True})
        else:
            self._send(404, "not found", "text/plain")

    def log_message(self, *args):   # ไม่ต้องพิมพ์ log ทุก request
        pass


def main():
    global DEFAULT_URL
    if len(sys.argv) > 1:
        DEFAULT_URL = sys.argv[1].rstrip("/")
        SESSION.state["url"] = DEFAULT_URL
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    page = f"http://localhost:{PORT}"
    print(f"เปิดหน้าเว็บที่ {page}   (กด Ctrl+C เพื่อหยุด)")
    threading.Timer(0.8, lambda: webbrowser.open(page)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        SESSION.stop()
        server.server_close()
        print("\nหยุดการทำงานแล้ว")


if __name__ == "__main__":
    main()
