import math
import cv2
import numpy as np

def get_attention_heatmap(attention_store, token_indices, img):
    """
    Generates pure JET attention heatmaps for multiple tokens using GLOBAL normalization,
    and returns a single combined image row: [Original | Token 1 Map | Token 2 Map | ...]
    """
    h, w = img.shape[:2]
    img_size = (w, h) 
    
    attention_maps = attention_store.get_average_attention()
    attn_list = attention_maps["up_cross"]
    
    # Start the row with a copy of the original image
    images_to_concat = [img.copy()]
    
    # --- PASS 1: Collect raw heatmaps and find GLOBAL min/max ---
    raw_heatmaps = []
    global_min = float('inf')
    global_max = float('-inf')
    
    for token_idx in token_indices:
        heatmaps = []
        for attn_tensor in attn_list:
            attn = attn_tensor.mean(dim=0)
            attn = attn[:, token_idx]
            spatial = int(math.sqrt(attn.shape[0]))
            attn = attn.reshape(spatial, spatial).cpu().numpy()
            attn = cv2.resize(attn, img_size)
            heatmaps.append(attn)
            
        # Average all resized maps for this specific token
        attn_avg = np.mean(heatmaps, axis=0)
        raw_heatmaps.append(attn_avg)
        
        # Track global extremes
        if attn_avg.min() < global_min:
            global_min = attn_avg.min()
        if attn_avg.max() > global_max:
            global_max = attn_avg.max()

    # --- PASS 2: Normalize globally, smooth, and colorize ---
    for attn in raw_heatmaps:
        # Normalize using the GLOBAL max and min
        attn = (attn - global_min) / (global_max - global_min + 1e-8)
        
        # Convert to uint8 
        attn_uint8 = (attn * 255).astype(np.uint8)
        
        # Apply Gaussian Smoothing
        attn_uint8 = cv2.GaussianBlur(attn_uint8, (15, 15), 0)
        
        # Apply standard JET colormap (Returns BGR format)
        heatmap_color = cv2.applyColorMap(attn_uint8, cv2.COLORMAP_JET)
        
        # --- THE FIX: Convert OpenCV's BGR output to standard RGB ---
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # Add the pure heatmap to our row
        images_to_concat.append(heatmap_color)
        
    # Horizontally concatenate all images in the list
    final_row = cv2.hconcat(images_to_concat)
    
    return final_row