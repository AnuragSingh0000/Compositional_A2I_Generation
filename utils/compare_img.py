import os
from PIL import Image, ImageDraw, ImageFont

# Folders for each model
model_dirs = [
    "output_macs/VGG/imgs",
    "output_vgg/imgs/VGG_AT_org",
    "output_vgg/imgs/VGG_AT",
    "output_vgg/imgs/VGG_mixed",
    "output_vgg/imgs/VGG_mixed_claps",
]

# Extract labels from folder names (e.g., "VGGn_AT", "VGGn_Ane")
model_labels = [os.path.basename(os.path.normpath(d)) for d in model_dirs]

output_dir = "output_vgg/cmp"
os.makedirs(output_dir, exist_ok=True)

valid_ext = (".png", ".jpg", ".jpeg")

model_maps = []

for d in model_dirs:
    file_map = {}
    if not os.path.exists(d):
        print(f"Warning: Directory {d} does not exist.")
        continue
    for f in os.listdir(d):
        if f.lower().endswith(valid_ext):
            base = os.path.splitext(f)[0]
            file_map[base] = os.path.join(d, f)
    model_maps.append(file_map)

# Find common basenames
if not model_maps:
    print("No valid directories found.")
    exit()

common_names = set(model_maps[0].keys())
for m in model_maps[1:]:
    common_names &= set(m.keys())

common_names = sorted(common_names)

# Configuration for the text header
header_height = 50
try:
    # Try to load a standard truetype font for better visibility (Arial)
    # You might need to adjust the font path depending on your OS (e.g., 'DejaVuSans.ttf' on Linux)
    font = ImageFont.truetype("arial.ttf", 24)
except IOError:
    # Fallback to default if arial.ttf is not found
    font = ImageFont.load_default()

for name in common_names:
    images = [Image.open(m[name]) for m in model_maps]

    # Get size from first image
    w, h = images[0].size
    total_width = w * len(images)
    total_height = h + header_height

    # Create canvas with a white background for the header
    combined = Image.new("RGB", (total_width, total_height), color="white")
    draw = ImageDraw.Draw(combined)

    # Paste side by side and add labels
    for i, (img, label) in enumerate(zip(images, model_labels)):
        x_offset = i * w
        
        # Paste the image below the header
        combined.paste(img, (x_offset, header_height))
        
        # Calculate text bounding box to center it
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        text_x = x_offset + (w - text_w) // 2
        text_y = (header_height - text_h) // 2
        
        # Draw the label
        draw.text((text_x, text_y), label, fill="black", font=font)

    # Save
    combined.save(os.path.join(output_dir, f"{name}.png"))

print(f"Done! Processed {len(common_names)} images.")
