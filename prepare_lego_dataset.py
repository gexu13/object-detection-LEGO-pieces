import os
import random
import shutil
import xml.etree.ElementTree as ET

# ========================== 1️⃣ 设定数据路径 ==========================
# 原始数据
IMAGE_DIR = "dataset_selected/images"
ANNOTATION_DIR = "dataset_selected/annotations"

# 清理后数据
CLEAN_IMAGE_DIR = "dataset_clean/images"
CLEAN_ANNOTATION_DIR = "dataset_clean/annotations"

# 最终数据集
FINAL_DATASET_DIR = "dataset_final"

# 创建文件夹
os.makedirs(CLEAN_IMAGE_DIR, exist_ok=True)
os.makedirs(CLEAN_ANNOTATION_DIR, exist_ok=True)
os.makedirs(FINAL_DATASET_DIR, exist_ok=True)

for split in ["train", "val", "test"]:
    os.makedirs(f"{FINAL_DATASET_DIR}/{split}/images", exist_ok=True)
    os.makedirs(f"{FINAL_DATASET_DIR}/{split}/annotations", exist_ok=True)

# ========================== 2️⃣ 数据清理（移除无效数据） ==========================
def parse_voc_annotation(xml_file):
    """解析 PASCAL VOC XML 标注"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        objects = root.findall("object")
        boxes = []
        for obj in objects:
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
        return boxes if boxes else None
    except:
        return None

# 遍历数据，移除无效标注
valid_files = []
for filename in os.listdir(ANNOTATION_DIR):
    xml_path = os.path.join(ANNOTATION_DIR, filename)
    img_path = os.path.join(IMAGE_DIR, filename.replace(".xml", ".jpg"))

    # 确保图片和标注文件存在
    if not os.path.exists(img_path):
        continue

    # 解析 XML，如果没有有效标注，则跳过
    boxes = parse_voc_annotation(xml_path)
    if boxes:
        shutil.copy(img_path, CLEAN_IMAGE_DIR)
        shutil.copy(xml_path, CLEAN_ANNOTATION_DIR)
        valid_files.append(filename.replace(".xml", ""))

print(f"✅ 数据清理完成，剩余有效图片数量: {len(valid_files)}")

# ========================== 3️⃣ 划分数据集（70% 训练，15% 验证，15% 测试） ==========================
random.seed(42)
random.shuffle(valid_files)

total = len(valid_files)
train_split = int(total * 0.7)
val_split = int(total * 0.85)

train_files = valid_files[:train_split]
val_files = valid_files[train_split:val_split]
test_files = valid_files[val_split:]

# 复制文件到对应数据集
def copy_files(file_list, split):
    for file in file_list:
        img_src = os.path.join(CLEAN_IMAGE_DIR, file + ".jpg")
        xml_src = os.path.join(CLEAN_ANNOTATION_DIR, file + ".xml")
        img_dst = f"{FINAL_DATASET_DIR}/{split}/images/{file}.jpg"
        xml_dst = f"{FINAL_DATASET_DIR}/{split}/annotations/{file}.xml"
        shutil.copy(img_src, img_dst)
        shutil.copy(xml_src, xml_dst)

copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

print(f"✅ 数据集划分完成:")
print(f"  训练集: {len(train_files)} 张")
print(f"  验证集: {len(val_files)} 张")
print(f"  测试集: {len(test_files)} 张")

# ========================== 4️⃣ 统一类别标签（600 类 -> "lego"） ==========================
def update_annotations(annotation_dir):
    for filename in os.listdir(annotation_dir):
        xml_path = os.path.join(annotation_dir, filename)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 统一所有类别为 "lego"
        for obj in root.findall("object"):
            obj.find("name").text = "lego"

        tree.write(xml_path)

for split in ["train", "val", "test"]:
    update_annotations(f"{FINAL_DATASET_DIR}/{split}/annotations")

print("✅ 统一标签完成 (所有类别 -> lego)")

print("🚀 数据处理完成，可以用于训练 SSD 目标检测模型！")
