# utils/mitotic_counter.py
import os
import cv2
import numpy as np
from ultralytics import YOLO
from django.conf import settings
from django.core.files import File
import shutil
from mitotic_app.models import DetectedFigure


def process_video(video_path, model_path, analysis_id):
    """Process video to count mitotic and non-mitotic figures"""
    
    # Create output directories
    base_dir = os.path.join(settings.MEDIA_ROOT, f'analysis_{analysis_id}')
    output_dir_mitotic = os.path.join(base_dir, 'output_mitotic')
    output_dir_non_mitotic = os.path.join(base_dir, 'output_non_mitotic')
    output_debug_mitotic = os.path.join(base_dir, 'output_debug', 'output_mitotic')
    output_debug_non_mitotic = os.path.join(base_dir, 'output_debug', 'output_non_mitotic')
    processed_video_path = os.path.join(base_dir, 'processed_video.mp4')

    # Create directories if they don't exist
    for directory in [output_dir_mitotic, output_dir_non_mitotic, output_debug_mitotic, output_debug_non_mitotic]:
        os.makedirs(directory, exist_ok=True)

    # Load the YOLO model
    model = YOLO(model_path)

    # Set the confidence threshold
    conf_threshold = 0.7

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video dimensions and properties
    ret, first_frame = cap.read()
    if not ret:
        print("Error reading video file")
        return None
        
    height, width = first_frame.shape[:2]
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the vertical line position (in the middle of the frame)
    line_x = width // 2
    # Define the gap (5% from top and bottom)
    gap_top = int(height * 0.05)
    gap_bottom = int(height * 0.05)
    line_start = (line_x, gap_top)
    line_end = (line_x, height - gap_bottom)

    # Counters for objects crossing the line
    mitotic_count = 0
    non_mitotic_count = 0
    frame_count = 0

    # Reset video capture to start
    cap.release()
    cap = cv2.VideoCapture(video_path)

    # Set up output video writer for processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))

    # Dictionary to track objects and their paths
    objects_track = {}  
    next_id = 0
    max_disappeared = 15  # Frames before we forget an object
    iou_threshold = 0.3  # Threshold for matching boxes between frames

    # Colors for visualization
    COLOR_MITOTIC = (0, 255, 0)       # Green for mitotic
    COLOR_NON_MITOTIC = (255, 165, 0)  # Orange for non-mitotic
    COLOR_CROSSED = (0, 0, 255)       # Red for crossed line

    figures_data = []

    def calculate_iou(box1, box2):
        """Calculate IoU between two boxes [x1, y1, x2, y2]"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0

    # For each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Make a copy of the frame for visualization
        vis_frame = frame.copy()
        
        # Create a copy for the debug output (with line)
        debug_frame = frame.copy()
        
        # Draw the vertical line for visualization (only on debug frame)
        cv2.line(debug_frame, line_start, line_end, COLOR_CROSSED, 2)
        
        # Perform inference
        results = model(frame)
        
        # Mark all objects as not found in this frame
        for obj_id in objects_track:
            objects_track[obj_id]['found'] = False
        
        # Check for detections
        for box in results[0].boxes:
            confidence = box.conf.item()
            class_id = int(box.cls.item())
            
            if confidence >= conf_threshold:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                current_box = [x1, y1, x2, y2]
                
                # Determine if this is mitotic or non-mitotic
                is_mitotic = class_id == 0  # Change this if your class mappings are different   ##change
                
                # Set color based on class
                color = COLOR_MITOTIC if is_mitotic else COLOR_NON_MITOTIC
                
                # Draw bounding box on both frames
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add label only to debug frame
                label = f'{"Mitotic" if is_mitotic else "Non-Mitotic"} {confidence:.2f}'
                cv2.putText(debug_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Calculate center of the box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Skip objects that are significantly outside the valid vertical region
                if y2 < gap_top or y1 > (height - gap_bottom):
                    continue
                    
                # Try to match with existing objects
                matched = False
                best_match_id = None
                best_iou = 0
                
                for obj_id, obj_data in objects_track.items():
                    if obj_data['found'] or obj_data['class'] != class_id:
                        # Skip already matched objects or objects of different class
                        continue
                        
                    if len(obj_data['positions']) > 0:
                        last_pos = obj_data['positions'][-1]
                        curr_iou = calculate_iou(last_pos, current_box)
                        
                        if curr_iou > iou_threshold and curr_iou > best_iou:
                            best_match_id = obj_id
                            best_iou = curr_iou
                
                if best_match_id is not None:
                    # Update the existing object
                    objects_track[best_match_id]['positions'].append(current_box)
                    objects_track[best_match_id]['found'] = True
                    objects_track[best_match_id]['disappeared'] = 0
                    
                    # Check if it has crossed the line from right to left
                    if not objects_track[best_match_id]['crossed']:
                        # Get previous and current positions
                        positions = objects_track[best_match_id]['positions']
                        
                        if len(positions) >= 2:
                            prev_box = positions[-2]
                            curr_box = positions[-1]
                            prev_center_x = (prev_box[0] + prev_box[2]) // 2
                            curr_center_x = (curr_box[0] + curr_box[2]) // 2
                            
                            # Check if it crossed the line from right to left
                            if prev_center_x > line_x and curr_center_x <= line_x:
                                # Increment appropriate counter
                                if objects_track[best_match_id]['class'] == 1:  # Mitotic
                                    mitotic_count += 1
                                    counter = mitotic_count
                                    output_dir = output_dir_mitotic
                                    debug_dir = output_debug_mitotic
                                    count_type = "Mitotic"
                                else:  # Non-mitotic
                                    non_mitotic_count += 1
                                    counter = non_mitotic_count
                                    output_dir = output_dir_non_mitotic
                                    debug_dir = output_debug_non_mitotic
                                    count_type = "Non-Mitotic"
                                    
                                objects_track[best_match_id]['crossed'] = True
                                
                                # Draw crossing indicators (only on debug frame)
                                cv2.circle(debug_frame, (center_x, center_y), 8, COLOR_CROSSED, -1)
                                cv2.putText(debug_frame, f"CROSS", (center_x, center_y - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_CROSSED, 2)
                                
                                # Save the clean frame with just the bounding box
                                clean_filename = os.path.join(output_dir, f'{count_type.lower()}crossing{counter:04d}frame{frame_count:04d}.jpg')
                                cv2.imwrite(clean_filename, vis_frame)
                                
                                # Save the debug frame with all visualization
                                debug_filename = os.path.join(debug_dir, f'{count_type.lower()}crossing{counter:04d}frame{frame_count:04d}_debug.jpg')
                                
                                # Add count information to the debug frame
                                info_text = f"Mitotic: {mitotic_count} | Non-Mitotic: {non_mitotic_count}"
                                cv2.putText(debug_frame, info_text, (10, 30), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                                
                                cv2.imwrite(debug_filename, debug_frame)
                                print(f'{count_type} figure crossed the line! Count: {counter}. Frame: {frame_count}')
                                
                                # Store figure data for database
                                rel_path = os.path.relpath(clean_filename, settings.MEDIA_ROOT)
                                figures_data.append({
                                    'image_path': rel_path,
                                    'category': 'mitotic' if count_type == "Mitotic" else 'non_mitotic',
                                    'confidence': confidence,
                                    'frame_number': frame_count
                                })
                else:
                    # Create a new object
                    objects_track[next_id] = {
                        'positions': [current_box],
                        'crossed': False,
                        'found': True,
                        'disappeared': 0,
                        'class': class_id  # Store the class ID
                    }
                    # For new objects starting on the left side of the line
                    center_x = (x1 + x2) // 2
                    if center_x <= line_x:
                        # Already crossed when first detected, mark as crossed
                        objects_track[next_id]['crossed'] = True
                    next_id += 1
        
        # Update tracking - increment disappeared counter for objects not found
        object_ids_to_delete = []
        
        for obj_id, obj_data in objects_track.items():
            if not obj_data['found']:
                obj_data['disappeared'] += 1
                
                # If object has been missing for too long, mark it for deletion
                if obj_data['disappeared'] > max_disappeared:
                    object_ids_to_delete.append(obj_id)
        
        # Remove objects that have disappeared for too long
        for obj_id in object_ids_to_delete:
            del objects_track[obj_id]
        
        # Draw tracking information (only on debug frame)
        for obj_id, obj_data in objects_track.items():
            if len(obj_data['positions']) > 0 and obj_data['disappeared'] < 3:
                last_box = obj_data['positions'][-1]
                x1, y1, x2, y2 = last_box
                
                # Color based on class and crossed status
                if obj_data['crossed']:
                    color = COLOR_CROSSED
                else:
                    color = COLOR_MITOTIC if obj_data['class'] == 1 else COLOR_NON_MITOTIC
                    
                # Draw tracking box and ID on debug frame only
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 1)
                class_label = "M" if obj_data['class'] == 1 else "N"
                cv2.putText(debug_frame, f"{class_label}:{obj_id}", (x1, y1-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add count to the frame
        info_text = f"Mitotic: {mitotic_count} | Non-Mitotic: {non_mitotic_count}"
        cv2.putText(debug_frame, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Write the processed frame to output video
        out.write(debug_frame)
        
        frame_count += 1

    print(f"Total counts:")
    print(f"- Mitotic figures: {mitotic_count}")
    print(f"- Non-mitotic figures: {non_mitotic_count}")
    print(f"- Total figures: {mitotic_count + non_mitotic_count}")
    
    cap.release()
    out.release()
    
    results = {
        'mitotic_count': mitotic_count,
        'non_mitotic_count': non_mitotic_count,
        'total_count': mitotic_count + non_mitotic_count,
        'figures_data': figures_data,
        'processed_video': os.path.relpath(processed_video_path, settings.MEDIA_ROOT)
    }
    
    return results

def process_video_with_boxes(input_video_path, model_path, output_path):
    """Process video and add bounding boxes using YOLO model"""
    # Load your YOLO model
    model = YOLO(model_path)
    
    # Load video
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Set up output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO prediction
        results = model(frame)
        
        # Plot results on the frame
        annotated_frame = results[0].plot()  # draws bounding boxes, etc.
        
        # Write annotated frame to output video
        out.write(annotated_frame)
    
    # Clean up
    cap.release()
    out.release()
    
    return output_path

def move_figure(figure_id, new_category):
    try:
        figure = DetectedFigure.objects.get(id=figure_id)
        old_path = figure.image_file.path
        old_category = figure.category

        if not os.path.exists(old_path):
            print(f"Original file does not exist: {old_path}")
            return False

        # Construct new path
        filename = os.path.basename(old_path)
        new_dir = os.path.join(settings.MEDIA_ROOT, 'figures', new_category)
        os.makedirs(new_dir, exist_ok=True)

        new_path = os.path.join(new_dir, filename)

        # Prevent overwrite
        if os.path.exists(new_path):
            base, ext = os.path.splitext(filename)
            count = 1
            while os.path.exists(new_path):
                new_filename = f"{base}_{count}{ext}"
                new_path = os.path.join(new_dir, new_filename)
                count += 1
            filename = os.path.basename(new_path)

        # Move file
        shutil.move(old_path, new_path)

        # Update database path
        relative_path = os.path.relpath(new_path, settings.MEDIA_ROOT)
        figure.image_file.name = relative_path.replace("\\", "/")
        figure.category = new_category
        figure.save()

        return True

    except DetectedFigure.DoesNotExist:
        print("Figure not found in database.")
        return False
    except Exception as e:
        print(f"Error during move: {e}")
        return False