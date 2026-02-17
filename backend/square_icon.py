from PIL import Image

def make_square(image_path, output_path, fill_color=(0, 0, 0, 0)):
    img = Image.open(image_path)
    img = img.convert("RGBA")
    
    width, height = img.size
    new_size = max(width, height)
    
    new_img = Image.new("RGBA", (new_size, new_size), fill_color)
    
    # Center the original image
    offset_x = (new_size - width) // 2
    offset_y = (new_size - height) // 2
    new_img.paste(img, (offset_x, offset_y), img)
    
    new_img.save(output_path)
    print(f"Created square icon: {output_path} ({new_size}x{new_size})")

make_square("../frontend/public/logo.png", "../frontend/app-icon.png")
