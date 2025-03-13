import os
import cv2
import numpy as np
import random
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image, ImageStat, UnidentifiedImageError

# 设置数据集路径
image_dir = "dataset/images"
annotation_dir = "dataset/annotations"

# 过滤后数据存储路径
filtered_image_dir = "dataset_filtered/images"
filtered_annotation_dir = "dataset_filtered/annotations"

# 最终 20K 采样数据存储路径
selected_image_dir = "dataset_selected/images"
selected_annotation_dir = "dataset_selected/annotations"

# 创建文件夹
os.makedirs(filtered_image_dir, exist_ok=True)
os.makedirs(filtered_annotation_dir, exist_ok=True)
os.makedirs(selected_image_dir, exist_ok=True)
os.makedirs(selected_annotation_dir, exist_ok=True)

# 📌 1. 过滤损坏的图片（无法打开的直接删除）
def is_valid_image(img_path):
    try:
        img = Image.open(img_path)
        img.verify()  # 验证是否为损坏图片
        return True
    except (UnidentifiedImageError, IOError):
        return False

for img_file in tqdm(os.listdir(image_dir), desc="Checking images"):
    img_path = os.path.join(image_dir, img_file)
    
    if not img_file.endswith((".jpg", ".png")) or not is_valid_image(img_path):
        print(f"⚠️ 删除损坏图片: {img_file}")
        os.remove(img_path)

print("✅ 已删除所有损坏的图片")


# 📌 2. 过滤低质量图片（模糊、亮度低、对比度低）
def brightness(img_path):
    img = Image.open(img_path).convert('L')
    stat = ImageStat.Stat(img)
    return stat.mean[0]

def contrast(img_path):
    img = Image.open(img_path).convert('L')
    return np.std(np.array(img))

def sharpness(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return cv2.Laplacian(img, cv2.CV_64F).var()

BRIGHTNESS_THRESHOLD = 50  
CONTRAST_THRESHOLD = 10  
SHARPNESS_THRESHOLD = 100  

filtered_images = []
for img_file in tqdm(os.listdir(image_dir), desc="Filtering images"):
    img_path = os.path.join(image_dir, img_file)

    b = brightness(img_path)
    c = contrast(img_path)
    s = sharpness(img_path)

    if b > BRIGHTNESS_THRESHOLD and c > CONTRAST_THRESHOLD and s > SHARPNESS_THRESHOLD:
        filtered_images.append(img_file)
        shutil.copy(img_path, os.path.join(filtered_image_dir, img_file))

print(f"✅ 已过滤低质量图片，剩余 {len(filtered_images)} 张")


# 📌 3. 删除无效 XML 标注文件（为空或损坏）
def is_valid_xml(xml_path):
    try:
        # 读取原始字节数据，防止编码错误
        with open(xml_path, 'rb') as f:
            raw = f.read()

        # 如果文件为空，直接返回 False
        if not raw.strip():
            return False

        # 以 UTF-8 解码（忽略非法字符）
        content = raw.decode('utf-8', errors='ignore').strip()

        # 确保 XML 以 "<?xml" 开头（防止非 XML 文件）
        if not content.startswith("<?xml"):
            return False

        # 解析 XML
        ET.fromstring(content)  # 不用 ET.parse()，避免报错
        return True
    except (ET.ParseError, UnicodeDecodeError, FileNotFoundError):
        return False  # XML 解析失败

for xml_file in os.listdir(annotation_dir):
    xml_path = os.path.join(annotation_dir, xml_file)

    if not is_valid_xml(xml_path):  # 检测损坏 XML
        print(f"⚠️ 删除损坏的 XML 文件: {xml_file}")
        os.remove(xml_path)

print("✅ 已删除所有损坏或空 XML 文件")


# 📌 4. 删除过小 LEGO 颗粒（<50×50 px）
for xml_file in os.listdir(filtered_annotation_dir):
    xml_path = os.path.join(filtered_annotation_dir, xml_file)
    try:
        tree = ET.parse(xml_path)  # 解析 XML
    except ET.ParseError:
        print(f"⚠️ 跳过无法解析的 XML 文件: {xml_file}")
        continue  # 直接跳过该文件
    
    root = tree.getroot()

    keep = False
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        area = (xmax - xmin) * (ymax - ymin)
        if area > 2500:  # 50x50 pixels
            keep = True
            break

    if not keep:
        os.remove(xml_path)
        img_file = xml_file.replace(".xml", ".jpg")
        if os.path.exists(os.path.join(filtered_image_dir, img_file)):
            os.remove(os.path.join(filtered_image_dir, img_file))

print("✅ 已删除过小 LEGO 颗粒的图片和标注")


# 📌 5. 选取 20K 张高质量图片及其标注
image_files = [f for f in os.listdir(filtered_image_dir) if f.endswith(".jpg")]
random.shuffle(image_files)
sample_files = image_files[:20000]

for file in tqdm(sample_files, desc="Selecting 20K images"):
    shutil.copy(os.path.join(filtered_image_dir, file), os.path.join(selected_image_dir, file))

    annotation_file = file.replace(".jpg", ".xml")
    if os.path.exists(os.path.join(filtered_annotation_dir, annotation_file)):
        shutil.copy(os.path.join(filtered_annotation_dir, annotation_file), os.path.join(selected_annotation_dir, annotation_file))

print("✅ 已成功采样 20,000 张图片及其标注！")


# 📌 6. 确保图片和标注文件一一对应
image_files = set(f.replace(".jpg", "") for f in os.listdir(selected_image_dir) if f.endswith(".jpg"))
annotation_files = set(f.replace(".xml", "") for f in os.listdir(selected_annotation_dir) if f.endswith(".xml"))

missing_annotations = image_files - annotation_files
missing_images = annotation_files - image_files

for img in missing_annotations:
    img_path = os.path.join(selected_image_dir, img + ".jpg")
    os.remove(img_path)

for ann in missing_images:
    ann_path = os.path.join(selected_annotation_dir, ann + ".xml")
    os.remove(ann_path)

print(f"✅ 已删除 {len(missing_annotations)} 张无标注图片，{len(missing_images)} 份无图片标注。")

print("\n🎉 数据预处理完成，最终 20K 张高质量 LEGO 颗粒图片已准备好！")
