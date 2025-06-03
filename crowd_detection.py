import cv2
import csv
import os
import argparse
import numpy as np
from collections import deque

class CrowdDetector:
    def __init__(self, video_path, output_csv,
                 detection_confidence=0.0,
                 crowd_size=3, persistence_frames=10,
                 proximity_thresh=100, visualize=False):
        # Video and output paths
        self.video_path = video_path
        self.output_csv = output_csv

        # Initialize HOG person detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Parameters
        self.conf_thresh = detection_confidence  # allow all detections by default
        self.crowd_size = crowd_size
        self.persistence_frames = persistence_frames
        self.proximity_thresh = proximity_thresh
        self.visualize = visualize

        # Persistence buffer: track presence of crowd per frame
        self.presence_buffer = deque(maxlen=persistence_frames)
        self.logged_frames = set()
        self.logs = []

    def detect_people(self, frame):
        # returns centroids and bounding boxes
        rects, weights = self.hog.detectMultiScale(
            frame,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.05
        )
        centroids = []
        boxes = []
        for (x, y, w, h), weight in zip(rects, weights):
            if weight < self.conf_thresh:
                continue
            centroids.append((x + w // 2, y + h // 2))
            boxes.append((x, y, w, h))
        return centroids, boxes

    def group_crowds(self, centroids):
        groups = []
        used = set()
        for i, c1 in enumerate(centroids):
            if i in used:
                continue
            group = [c1]
            used.add(i)
            for j, c2 in enumerate(centroids[i+1:], start=i+1):
                if j not in used and np.linalg.norm(np.array(c1)-np.array(c2)) < self.proximity_thresh:
                    group.append(c2)
                    used.add(j)
            if len(group) >= self.crowd_size:
                groups.append(group)
        return groups

    def run(self):
        if not os.path.isfile(self.video_path):
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            centroids, boxes = self.detect_people(frame)
            groups = self.group_crowds(centroids)

            # Debug
            print(f"Frame {frame_idx}: {len(centroids)} detected, {len(groups)} groups")

            has_crowd = len(groups) > 0
            self.presence_buffer.append((frame_idx, has_crowd, groups))

            # If buffer full and all recent frames have crowd, log once per start frame
            if len(self.presence_buffer) == self.persistence_frames:
                start_frame, present, _ = self.presence_buffer[0]
                if present and all(p for (_, p, _) in self.presence_buffer):
                    if start_frame not in self.logged_frames:
                        max_count = max(len(g) for (_, _, gs) in self.presence_buffer for g in gs)
                        self.logs.append((start_frame, max_count))
                        self.logged_frames.add(start_frame)
                        print(f"--> Logged crowd from frame {start_frame} size {max_count}")

            if self.visualize:
                for (x, y, w, h) in boxes:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                for group in groups:
                    for cx, cy in group:
                        cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)
                cv2.imshow('Crowd Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if self.visualize:
            cv2.destroyAllWindows()

        if not self.logs:
            print("No persistent crowds detected. Try adjusting --crowd, --persist, or --prox parameters.")
        self._save_logs()

    def _save_logs(self):
        with open(self.output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame Number', 'Person Count in Crowd'])
            for row in sorted(self.logs):
                writer.writerow(row)
        print(f"Results saved to {self.output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--output', default='crowd_events.csv', help='Output CSV file')
    parser.add_argument('--conf', type=float, default=0.0, help='Detection weight threshold')
    parser.add_argument('--crowd', type=int, default=3, help='Min people for crowd')
    parser.add_argument('--persist', type=int, default=10, help='Consecutive frames for persistence')
    parser.add_argument('--prox', type=float, default=100, help='Proximity threshold (pixels)')
    parser.add_argument('--viz', action='store_true', help='Visualize detections')
    args = parser.parse_args()

    detector = CrowdDetector(
        video_path=args.video,
        output_csv=args.output,
        detection_confidence=args.conf,
        crowd_size=args.crowd,
        persistence_frames=args.persist,
        proximity_thresh=args.prox,
        visualize=args.viz
    )
    detector.run()