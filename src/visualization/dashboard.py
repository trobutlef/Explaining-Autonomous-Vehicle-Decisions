import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def render_episode_overview(frames, actions, saliencies, feature_names=None, output_path="outputs/explanation_video.mp4", fps=5):
    """
    Visualize frames with overlays and action timeline/feature importance.
    
    Args:
        frames: List of numpy arrays (H, W, C) - RGB images
        actions: List of integers (actions taken)
        saliencies: List of feature importance arrays/dicts corresponding to each frame.
                    If SHAP: list of arrays [obs_dim]
                    If LIME: list of lists of (feature_idx, weight)
        feature_names: List of strings (names of features)
        output_path: Path to save the video
        fps: Frames per second
    """
    if not frames:
        print("No frames to render.")
        return

    height, width, _ = frames[0].shape
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Output size will be larger to accommodate the plot
    # Let's say we double the height to put the plot below
    out_height = height + 400
    out_width = max(width, 600) # Ensure enough width for the plot
    
    video = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    fig, ax = plt.subplots(figsize=(out_width/100, 4), dpi=100)
    
    for i, (frame, action, saliency) in enumerate(zip(frames, actions, saliencies)):
        # 1. Prepare the frame image
        # Resize frame to fit width if needed, or center it
        img_canvas = np.zeros((out_height, out_width, 3), dtype=np.uint8)
        
        # Place the environment frame at the top
        h_offset = 0
        w_offset = (out_width - width) // 2
        img_canvas[h_offset:h_offset+height, w_offset:w_offset+width] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 2. Generate the Feature Importance Plot
        ax.clear()
        
        # Parse saliency
        vals = []
        names = []
        colors = []
        
        if isinstance(saliency, (list, np.ndarray)):
            # Assume SHAP values (array)
            vals = saliency
            if feature_names:
                names = feature_names
            else:
                names = [f"Feat {j}" for j in range(len(vals))]
        elif hasattr(saliency, 'as_list'):
            # LIME explanation object
            lime_list = saliency.as_list()
            # lime_list is [(feature_name, value), ...]
            # We need to sort or just take top N
            lime_list.sort(key=lambda x: abs(x[1]), reverse=True)
            vals = [x[1] for x in lime_list[:10]] # Top 10
            names = [x[0] for x in lime_list[:10]]
        
        # Color bars: Red for positive (pushes towards action), Blue for negative
        colors = ['red' if v > 0 else 'blue' for v in vals]
        
        y_pos = np.arange(len(names))
        ax.barh(y_pos, vals, align='center', color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Frame {i} | Action: {action}')
        
        # Draw plot to canvas
        canvas = FigureCanvas(fig)
        canvas.draw()
        plot_img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        plot_img = plot_img.reshape(canvas.get_width_height()[::-1] + (3,))
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
        
        # Resize plot to match out_width
        plot_h, plot_w, _ = plot_img.shape
        scale = out_width / plot_w
        new_plot_h = int(plot_h * scale)
        plot_img_resized = cv2.resize(plot_img, (out_width, new_plot_h))
        
        # Place plot at the bottom
        # Ensure we don't go out of bounds
        remaining_h = out_height - height
        if new_plot_h > remaining_h:
             plot_img_resized = plot_img_resized[:remaining_h, :]
             
        img_canvas[height:height+plot_img_resized.shape[0], 0:out_width] = plot_img_resized
        
        video.write(img_canvas)
        
    video.release()
    plt.close(fig)
    print(f"Saved explanation video to {output_path}")
