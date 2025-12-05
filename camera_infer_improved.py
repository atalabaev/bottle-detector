#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
import csv
import sys

import cv2
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--save_path", type=str, default=None, help="save video file (optional)")
    p.add_argument("--snapshot_dir", type=str, default="snapshots", help="where to save snapshots when pressing 's'")
    p.add_argument("--log_csv", type=str, default="detection_log.csv", help="CSV log file for detections")
    p.add_argument("--auto_save_every", type=float, default=0.0, help="auto-save a snapshot every N seconds (0 = disabled)")
    p.add_argument("--device", type=str, default=None, help="device for inference e.g. cpu or 0 or cuda:0")
    return p.parse_args()


class DetectorApp:
    def __init__(self, args):
        self.args = args
        self.model = YOLO(args.weights)
        self.cam_idx = args.camera
        self.conf = args.conf
        self.imgsz = args.imgsz
        self.save_path = Path(args.save_path) if args.save_path else None
        self.snapshot_dir = Path(args.snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.log_csv = Path(args.log_csv)
        self.auto_save_every = args.auto_save_every
        self.device = args.device

        self.cap = None
        self.writer = None
        self.last_snapshot = 0.0
        self.frame_id = 0
        self.start_time = time.time()
        self.fps = 0.0
        self.dt_smooth = None
        self.running = True

        # detection analytics
        self.total_frames = 0
        self.detections_total = 0
        self.count_window = []  # sliding window of last N detection counts
        self.window_size = 30

        # ensure CSV header if not exists
        if not self.log_csv.exists():
            with open(self.log_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'frame_id', 'class', 'conf', 'x1', 'y1', 'x2', 'y2'])

    def open_camera(self):
        if self.cap and self.cap.isOpened():
            return True
        self.cap = cv2.VideoCapture(self.cam_idx)
        time.sleep(0.2)
        return self.cap.isOpened()

    def open_writer(self, w, h, fps=20.0):
        if not self.save_path:
            return
        # choose codec that's widely supported
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.writer = cv2.VideoWriter(str(self.save_path), fourcc, fps, (w, h))

    def close(self):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        try:
            if self.writer:
                self.writer.release()
        except Exception:
            pass
        cv2.destroyAllWindows()

    def log_detections(self, timestamp, frame_id, results):
        rows = []
        for r in results:
            # results is ultralytics.Results; may contain .boxes
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                rows.append([timestamp, frame_id, cls, conf, x1, y1, x2, y2])
        if rows:
            with open(self.log_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)

    def draw_overlay(self, frame, results):
        # draw boxes and analytics
        h, w = frame.shape[:2]
        total_this_frame = 0
        for r in results:
            for box in r.boxes:
                total_this_frame += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{cls}:{conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(frame, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

        # update sliding window
        self.count_window.append(total_this_frame)
        if len(self.count_window) > self.window_size:
            self.count_window.pop(0)

        avg_count = sum(self.count_window) / len(self.count_window) if self.count_window else 0.0

        # overlay stats
        elapsed = time.time() - self.start_time
        stats = [f"FPS: {self.fps:.1f}", f"Frame: {self.frame_id}", f"Avg detections: {avg_count:.2f}", f"Total detections: {self.detections_total}"]
        for i, s in enumerate(stats):
            cv2.putText(frame, s, (10, 20 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def run(self):
        # open camera
        if not self.open_camera():
            print("ERROR: cannot open camera index", self.cam_idx)
            return

        # get properties
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        cam_fps = self.cap.get(cv2.CAP_PROP_FPS) or 20.0
        self.open_writer(w, h, fps=max(5.0, cam_fps))

        last_time = time.time()
        last_auto_save = time.time()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Warning: empty frame, retrying camera...")
                time.sleep(0.3)
                if not self.open_camera():
                    time.sleep(1.0)
                continue

            self.frame_id += 1
            self.total_frames += 1

            t0 = time.time()
            # perform inference (single-frame)
            try:
                results = self.model.predict(source=frame, conf=self.conf, imgsz=self.imgsz)
            except Exception as e:
                print("Inference error:", e)
                results = []

            # count detections
            frame_dets = 0
            for r in results:
                frame_dets += len(r.boxes)
            self.detections_total += frame_dets

            # log to csv
            timestamp = time.time()
            self.log_detections(timestamp, self.frame_id, results)

            # draw boxes and overlay analytics
            frame = self.draw_overlay(frame, results)

            # write to file if needed
            if self.writer:
                try:
                    self.writer.write(frame)
                except Exception:
                    pass

            # compute fps
            t1 = time.time()
            dt = t1 - last_time
            last_time = t1
            self.fps = 1.0 / dt if dt > 0 else 0.0

            # show
            cv2.imshow("Detector", frame)

            # auto-save snapshots
            if self.auto_save_every > 0 and (time.time() - last_auto_save) >= self.auto_save_every:
                last_auto_save = time.time()
                fname = self.snapshot_dir / f"auto_{int(time.time())}.jpg"
                cv2.imwrite(str(fname), frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                self.running = False
                break
            elif key == ord('s'):
                fname = self.snapshot_dir / f"snap_{int(time.time())}.jpg"
                cv2.imwrite(str(fname), frame)
                print("Saved snapshot:", fname)

        self.close()


if __name__ == '__main__':
    args = parse_args()
    app = DetectorApp(args)
    try:
        app.run()
    except KeyboardInterrupt:
        app.close()
        sys.exit(0)

