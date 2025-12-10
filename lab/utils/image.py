import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed


def check_image(img_path):
    try:
        with Image.open(img_path) as img:
            img.verify()
    except Exception as e:
        return f"Corrupt image: {img_path} ({e})"
    return None


def check_images(root_dir, num_workers=8):
    img_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                img_paths.append(os.path.join(dirpath, filename))
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(check_image, path) for path in img_paths]
        for future in as_completed(futures):
            result = future.result()
            if result:
                print(result)


if __name__ == "__main__":
    check_images("/home/jovyan/nas/yrc/dataset/imagenet-1k/train", num_workers=8)