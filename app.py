from flask import Flask, render_template, Response
import cv2
import numpy as np
import time
import json
import os

app = Flask(__name__)

# --- CONFIGURATION ---
CAMERA_INDEX = 0
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", str(CAMERA_INDEX))
BACKEND = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY

# Global storage for real-time stats
# This allows the video loop to "talk" to the data stream
global_stats = {
    "health": 0,
    "damage": 0,
    "coverage": 0,
    "status": "Waiting for Plant..."
}

def create_leaf_mask(frame):
    h, w = frame.shape[:2]
    norm_frame = frame.astype(np.float32) / 255.0
    b, g, r = norm_frame[:,:,0], norm_frame[:,:,1], norm_frame[:,:,2]
    exg = 2 * g - r - b
    tau = 0.1
    leaf_mask = (exg > tau).astype(np.uint8)
    
    # Cleaning noise
    kernel = np.ones((3,3), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    
    # Filter small objects
    total_pixels = h * w
    min_leaf_pixels = int(0.05 * total_pixels)
    
    if np.sum(leaf_mask) < min_leaf_pixels:
        return np.zeros((h, w), np.uint8), 0, None
        
    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((h, w), np.uint8), 0, None
        
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < min_leaf_pixels:
        return np.zeros((h, w), np.uint8), 0, None
        
    x, y, w, h = cv2.boundingRect(largest_contour)
    return leaf_mask, np.sum(leaf_mask), (x, y, w, h)

def draw_symmetric_tensor_mesh(frame, bbox, leaf_mask, damage_ratio):
    x, y, w_bbox, h_bbox = bbox
    density = max(10, min(w_bbox, h_bbox) // 15)
    cols = np.arange(x, x + w_bbox, density)
    rows = np.arange(y, y + h_bbox, density)
    points = []
    
    for row in rows:
        for col in cols:
            if (0 <= col < frame.shape[1] and 0 <= row < frame.shape[0] and 
                leaf_mask[int(row), int(col)] > 0):
                points.append((int(col), int(row)))
    
    if len(points) < 3: return
    
    points_np = np.array(points, dtype=np.float32)
    rect = (0, 0, frame.shape[1], frame.shape[0])
    dt = cv2.Subdiv2D(rect)
    for p in points_np: dt.insert(p)
    
    triangle_list = dt.getTriangleList()
    
    # Mesh color shifts from Green (Healthy) to Red (Damaged)
    r = int(100 + 155 * damage_ratio)
    g = int(200 - 150 * damage_ratio)
    b = int(100 + 50 * damage_ratio)
    mesh_color = (b, g, r)
    
    for tri in triangle_list:
        pt1, pt2, pt3 = (int(tri[0]), int(tri[1])), (int(tri[2]), int(tri[3])), (int(tri[4]), int(tri[5]))
        
        # Draw mesh only inside bounding box to avoid artifacts
        if (x <= pt1[0] <= x+w_bbox and y <= pt1[1] <= y+h_bbox):
            cv2.line(frame, pt1, pt2, mesh_color, 1)
            cv2.line(frame, pt2, pt3, mesh_color, 1)
            cv2.line(frame, pt3, pt1, mesh_color, 1)
            
    for pt in points:
        cv2.circle(frame, pt, 2, (255, 255, 255), -1)

def generate_frames():
    global global_stats
    use_camera = os.getenv("USE_CAMERA", "true").lower() in ("1", "true", "yes")

    camera = None
    if use_camera:
        source = os.getenv("CAMERA_SOURCE", CAMERA_SOURCE)
        print(f"DEBUG: Opening camera source {source}...")

        if source.isdigit():
            camera = cv2.VideoCapture(int(source), BACKEND)
        else:
            camera = cv2.VideoCapture(source)

        time.sleep(1)

        if not camera.isOpened():
            print("WARNING: Camera source could not open. Falling back to placeholder mode.")
            camera.release()
            camera = None

    if camera is None:
        # Render and local demo mode when no camera exists (e.g., cloud host)
        global_stats["status"] = "NO_CAMERA"
        global_stats["health"] = 0
        global_stats["damage"] = 0
        global_stats["coverage"] = 0

        while True:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            text = "No Camera: Render cloud mode"
            cv2.putText(frame, text, (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)
        return

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        h, w = frame.shape[:2]

        # --- ANALYSIS LOGIC ---
        leaf_mask, num_leaf_pixels, bbox = create_leaf_mask(frame)

        if num_leaf_pixels == 0 or bbox is None:
            global_stats["status"] = "No Plant Detected"
            global_stats["health"] = 0
            global_stats["damage"] = 0
            global_stats["coverage"] = 0
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_healthy = np.array([40, 50, 50])
            upper_healthy = np.array([80, 255, 255])
            healthy_mask = cv2.inRange(hsv, lower_healthy, upper_healthy) / 255

            leaf_healthy = np.logical_and(leaf_mask, healthy_mask).astype(np.float32)
            num_healthy = np.sum(leaf_healthy)

            damage_percent = ((num_leaf_pixels - num_healthy) / num_leaf_pixels) * 100
            health_percent = 100 - damage_percent
            coverage_percent = (num_leaf_pixels / (h * w)) * 100

            damage_ratio = min(damage_percent / 100, 1.0)

            global_stats["status"] = "HEALTHY" if health_percent > 70 else ("MODERATE" if health_percent > 30 else "DAMAGED")
            global_stats["health"] = round(health_percent, 1)
            global_stats["damage"] = round(damage_percent, 1)
            global_stats["coverage"] = round(coverage_percent, 1)

            x, y, w_bbox, h_bbox = bbox
            box_color = (int(255 * damage_ratio), int(255 * (1 - damage_ratio)), 0)
            cv2.rectangle(frame, (x, y), (x + w_bbox, y + h_bbox), box_color, 2)
            draw_symmetric_tensor_mesh(frame, bbox, leaf_mask, damage_ratio)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_data():
    """Stream JSON data to the client"""
    while True:
        time.sleep(0.1) # Update every 100ms
        # Format as Server-Sent Event (SSE)
        json_data = json.dumps(global_stats)
        yield f"data: {json_data}\n\n"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats_feed')
def stats_feed():
    return Response(generate_data(), mimetype='text/event-stream')

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)