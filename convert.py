import cairosvg
from PIL import Image

# Convert SVG to PNG
cairosvg.svg2png(url="SCB.svg", write_to="temp.png")
icon_width = 128
icon_height = 256

# Convert PNG to ICO
img = Image.open("temp.png")
img.save("scb.ico", format="ICO", sizes=[(icon_width, icon_height)])
