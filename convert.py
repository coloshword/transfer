import cairosvg
from PIL import Image

# Define the square icon size (must be square for Windows)
icon_size = 256  # Windows prefers square sizes like 256x256

# Convert SVG to PNG
cairosvg.svg2png(url="scb.svg", write_to="temp.png")

# Open the original image
img = Image.open("temp.png")

# Get original dimensions
orig_width, orig_height = img.size

# Determine the new square canvas size (based on the tallest side)
max_size = max(orig_width, orig_height)

# Create a new square image with transparency
new_img = Image.new("RGBA", (max_size, max_size), (0, 0, 0, 0))

# Center the tall image inside the square canvas
offset_x = (max_size - orig_width) // 2
offset_y = 0  # Keep it at the top, or adjust if needed
new_img.paste(img, (offset_x, offset_y), img)

# Save as ICO (Windows requires square images, but transparency will preserve the tall effect)
new_img.save("scb.ico", format="ICO", sizes=[(16,16), (32,32), (48,48), (64,64), (128,128), (256,256)])
