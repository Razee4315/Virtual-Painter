import cv2
import mediapipe as mp
import numpy as np
import time
import math

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Toolbar settings
toolbar_height = 120
toolbar_color = (40, 40, 40)  # Dark gray background
button_color = (60, 60, 60)   # Slightly lighter gray for buttons
text_color = (255, 255, 255)  # White text
highlight_color = (0, 255, 0)  # Green highlight

# Colors in BGR format
colors = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
]

# Tools available
tools = ["Freehand", "Rectangle", "Circle", "Eraser"]

# Initial settings
current_color_index = 0
current_tool_index = 0
brush_thickness = 15
eraser_thickness = 30
drawing_mode = False
smoothing_factor = 0.5  # For drawing smoothing (0 to 1)

# Points for shape drawing
start_x, start_y = 0, 0
preview_shape = None
shape_started = False

# Canvas to draw on
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

# Previous positions for smooth drawing
prev_points = []
MAX_POINTS = 5  # Number of points to use for smoothing

# Gesture states
gesture_start_time = 0
GESTURE_DELAY = 0.2  # Seconds to hold gesture before activating

def smooth_points(points, factor):
    """Apply smoothing to a list of points"""
    if len(points) < 2:
        return points[-1] if points else (0, 0)
    
    smoothed = np.array(points[-1])
    for i in range(len(points)-2, -1, -1):
        weight = factor ** (len(points) - i - 1)
        smoothed = smoothed * (1 - weight) + np.array(points[i]) * weight
    
    return tuple(map(int, smoothed))

def create_toolbar():
    """Create the toolbar with all options"""
    toolbar = np.ones((toolbar_height, 1280, 3), dtype=np.uint8) * toolbar_color
    
    # Color selection boxes
    box_width = 60
    box_height = 60
    margin = 20
    start_x = margin
    start_y = (toolbar_height - box_height) // 2
    
    # Draw color selections
    for i, color in enumerate(colors):
        end_x = start_x + box_width
        # Draw color box with rounded corners
        cv2.rectangle(toolbar, (start_x, start_y), (end_x, start_y + box_height), color, -1)
        
        # Highlight selected color
        if i == current_color_index and current_tool_index != tools.index("Eraser"):
            cv2.rectangle(toolbar, (start_x-2, start_y-2), (end_x+2, start_y + box_height+2), highlight_color, 2)
        
        start_x = end_x + margin
    
    # Draw tool selections
    tool_box_width = 120
    tool_start_x = start_x + 50  # Extra gap between colors and tools
    
    for i, tool in enumerate(tools):
        tool_end_x = tool_start_x + tool_box_width
        # Draw tool box with rounded corners
        cv2.rectangle(toolbar, (tool_start_x, start_y), (tool_end_x, start_y + box_height), button_color, -1)
        
        # Add tool name
        text_size = cv2.getTextSize(tool, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = tool_start_x + (tool_box_width - text_size[0]) // 2
        text_y = start_y + (box_height + text_size[1]) // 2
        cv2.putText(toolbar, tool, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # Highlight selected tool
        if i == current_tool_index:
            cv2.rectangle(toolbar, (tool_start_x-2, start_y-2), (tool_end_x+2, start_y + box_height+2), highlight_color, 2)
        
        tool_start_x = tool_end_x + margin
    
    return toolbar

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def detect_toolbar_selection(x, y):
    """Detect if user is selecting an item in the toolbar"""
    global current_color_index, current_tool_index
    
    # Only detect selections if in toolbar area
    if y >= toolbar_height:
        return False
        
    box_width = 60
    box_height = 60
    margin = 20
    start_x = margin
    start_y = (toolbar_height - box_height) // 2
    
    # Check color selection
    for i in range(len(colors)):
        end_x = start_x + box_width
        if start_x <= x <= end_x and start_y <= y <= start_y + box_height:
            if current_tool_index == tools.index("Eraser"):  # Switch back to previous tool if eraser was selected
                current_tool_index = 0
            current_color_index = i
            return True
        start_x = end_x + margin
    
    # Check tool selection
    tool_box_width = 120
    tool_start_x = start_x + 50
    
    for i in range(len(tools)):
        tool_end_x = tool_start_x + tool_box_width
        if tool_start_x <= x <= tool_end_x and start_y <= y <= start_y + box_height:
            current_tool_index = i
            return True
        tool_start_x = tool_end_x + margin
    
    return False

def get_gesture_state(hand_landmarks):
    """Detect drawing gesture state based on hand landmarks"""
    if not hand_landmarks:
        return False, None
    
    # Get relevant finger landmarks
    index_tip = hand_landmarks.landmark[8]
    index_pip = hand_landmarks.landmark[6]  # First joint of index finger
    middle_tip = hand_landmarks.landmark[12]
    
    # Convert to screen coordinates
    index_tip_y = int(index_tip.y * 720)
    index_pip_y = int(index_pip.y * 720)
    middle_tip_y = int(middle_tip.y * 720)
    
    # Detect drawing gesture: index finger up, middle finger down
    drawing_gesture = (index_tip_y < index_pip_y) and (middle_tip_y > index_pip_y)
    
    return drawing_gesture, (int(index_tip.x * 1280), int(index_tip.y * 720))

def main():
    global drawing_mode, prev_points, canvas, start_x, start_y, preview_shape, gesture_start_time, shape_started
    
    while True:
        # Read frame from webcam
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break
            
        # Flip image horizontally for selfie-view
        img = cv2.flip(img, 1)
        
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process hand landmarks
        results = hands.process(img_rgb)
        
        # Create working image
        img_with_overlay = img.copy()
        
        # Create and overlay toolbar
        toolbar = create_toolbar()
        img_with_overlay[0:toolbar_height, 0:1280] = toolbar
        
        # Initialize gesture state
        gesture_detected = False
        current_point = None
        
        # Process hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_draw.draw_landmarks(img_with_overlay, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get gesture state and finger position
                gesture_detected, current_point = get_gesture_state(hand_landmarks)
                
                if gesture_detected:
                    # Draw pointer circle
                    cv2.circle(img_with_overlay, current_point, 10, (0, 255, 0), -1)
                    
                    # Handle toolbar selection
                    if detect_toolbar_selection(current_point[0], current_point[1]):
                        gesture_start_time = 0  # Reset gesture timer when selecting from toolbar
                        drawing_mode = False
                        shape_started = False
                        prev_points = []
                        continue
                    
                    # Gesture timing logic
                    if not drawing_mode:
                        if gesture_start_time == 0:
                            gesture_start_time = time.time()
                        elif time.time() - gesture_start_time >= GESTURE_DELAY:
                            drawing_mode = True
                            if current_tool_index in [1, 2] and not shape_started:  # Rectangle or Circle
                                start_x, start_y = current_point
                                shape_started = True
                            prev_points = [current_point]
                    
                    # Drawing logic
                    if drawing_mode and current_point[1] > toolbar_height:
                        if current_tool_index == tools.index("Eraser"):
                            # Draw white circle for eraser
                            cv2.circle(canvas, current_point, eraser_thickness, (0, 0, 0), -1)
                        elif current_tool_index == 0:  # Freehand
                            prev_points.append(current_point)
                            if len(prev_points) > MAX_POINTS:
                                prev_points.pop(0)
                            
                            if len(prev_points) >= 2:
                                smooth_point = smooth_points(prev_points, smoothing_factor)
                                cv2.line(canvas, prev_points[-2], smooth_point, colors[current_color_index], brush_thickness)
                        
                        elif current_tool_index in [1, 2] and shape_started:  # Rectangle or Circle
                            # Create preview shape
                            preview_shape = canvas.copy()
                            if current_tool_index == 1:  # Rectangle
                                cv2.rectangle(preview_shape, (start_x, start_y), current_point, colors[current_color_index], brush_thickness)
                            else:  # Circle
                                radius = int(calculate_distance((start_x, start_y), current_point))
                                cv2.circle(preview_shape, (start_x, start_y), radius, colors[current_color_index], brush_thickness)
                else:
                    # Handle gesture release
                    if drawing_mode:
                        if current_tool_index in [1, 2] and preview_shape is not None and shape_started:
                            canvas = preview_shape.copy()
                            preview_shape = None
                            shape_started = False
                        drawing_mode = False
                        gesture_start_time = 0
                        prev_points = []
        
        # Combine canvas with camera feed
        display_canvas = preview_shape if preview_shape is not None else canvas
        mask = cv2.cvtColor(display_canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        img_with_overlay = cv2.bitwise_and(img_with_overlay, img_with_overlay, mask=cv2.bitwise_not(mask))
        img_with_overlay = cv2.add(img_with_overlay, display_canvas)
        
        # Show the result
        cv2.imshow("Virtual Painter", img_with_overlay)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
