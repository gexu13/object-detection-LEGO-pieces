import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Lambda
from tensorflow.keras.applications import MobileNetV2

# ========================== 1️⃣ 设定数据路径 ==========================
DATASET_DIR = "dataset_final"
TEST_DIR = os.path.join(DATASET_DIR, "test")  # 测试集路径

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
MAX_BOXES = 10  # 最多检测 10 个目标

# ========================== 2️⃣ 解析 PASCAL VOC 数据 ==========================
def parse_voc_annotation(xml_file):
    """解析 PASCAL VOC XML 标注"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text) / IMAGE_WIDTH
        ymin = int(bbox.find("ymin").text) / IMAGE_HEIGHT
        xmax = int(bbox.find("xmax").text) / IMAGE_WIDTH
        ymax = int(bbox.find("ymax").text) / IMAGE_HEIGHT
        boxes.append([xmin, ymin, xmax, ymax])

    while len(boxes) < MAX_BOXES:
        boxes.append([0, 0, 0, 0])  # 用 0 填充
    return np.array(boxes[:MAX_BOXES], dtype=np.float32)

def load_image_and_annotation(image_path, annotation_path):
    """加载单张图片及其标注"""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    image = image.astype(np.float32) / 255.0  # 归一化

    boxes = parse_voc_annotation(annotation_path)
    return image, boxes

def get_image_label_paths(image_dir, annotation_dir):
    """获取所有测试图片和标注路径"""
    image_paths, annotation_paths = [], []
    for filename in os.listdir(annotation_dir):
        xml_path = os.path.join(annotation_dir, filename)
        img_path = os.path.join(image_dir, filename.replace(".xml", ".jpg"))
        if os.path.exists(img_path):
            image_paths.append(img_path)
            annotation_paths.append(xml_path)
    return image_paths, annotation_paths

# 获取测试集路径
test_image_paths, test_annotation_paths = get_image_label_paths(
    os.path.join(TEST_DIR, "images"), os.path.join(TEST_DIR, "annotations")
)

# ========================== 3️⃣ 加载已训练模型 ==========================
def build_ssd():
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False)
    x = base_model.output
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Flatten()(x)
    x = Dense(10 * 4, activation='sigmoid')(x)  # 10 个 bounding box，每个 4 个坐标
    x = Lambda(lambda y: tf.reshape(y, (-1, 10, 4)), output_shape=(10, 4))(x)  # ✅ 明确指定输出形状
    return Model(inputs=base_model.input, outputs=x)

# ✅ 重新构建模型
model = build_ssd()
print("✅ 成功重建模型！")

# ✅ 只加载训练好的权重，而不加载架构
model.load_weights("ssd_lego.h5")
print("✅ 成功加载权重！")

# ========================== 4️⃣ 计算 mAP@0.5 ==========================
def iou(boxA, boxB):
    """计算 IoU (Intersection over Union)"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    denominator = float(boxAArea + boxBArea - interArea)

    # ✅ 避免除以 0
    if denominator == 0:
        return 0.0

    return interArea / denominator

def mean_average_precision(true_boxes, pred_boxes, iou_threshold=0.5):
    """计算 mAP@0.5"""
    total_iou = 0
    num_samples = len(true_boxes)

    for true, pred in zip(true_boxes, pred_boxes):
        avg_iou = np.mean([iou(true_box, pred_box) for true_box, pred_box in zip(true, pred)])
        total_iou += avg_iou

    return total_iou / num_samples if num_samples > 0 else 0

def evaluate_model(model, image_paths, annotation_paths):
    """在测试集上评估模型并计算 mAP@0.5"""
    all_true_boxes = []
    all_pred_boxes = []

    for img_path, ann_path in zip(image_paths, annotation_paths):
        image, true_boxes = load_image_and_annotation(img_path, ann_path)
        image = np.expand_dims(image, axis=0)  # 增加 batch 维度
        pred_boxes = model.predict(image)[0]  # 预测的 bounding boxes

        all_true_boxes.append(true_boxes)
        all_pred_boxes.append(pred_boxes)
        print("🔍 真实框:", true_boxes)
        print("🔍 预测框:", pred_boxes)

    # 计算 mAP@0.5
    map_score = mean_average_precision(all_true_boxes, all_pred_boxes, iou_threshold=0.5)
    print(f"📊 测试集 mAP@0.5: {map_score:.4f}")


# if __name__ == "__main__":
#     # 评估模型
#     print("📝 评估模型...")
#     evaluate_model(model, test_image_paths, test_annotation_paths)
#     print("✅ 评估完成！")




import matplotlib.pyplot as plt

def visualize_predictions(image_path, pred_boxes):
    """可视化模型预测的 bounding boxes"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB
    h, w, _ = image.shape  # 获取原始图像大小

    plt.figure(figsize=(8, 8))
    plt.imshow(image)

    # 画出预测框
    for box in pred_boxes:
        if sum(box) == 0:  # 忽略全 0 的框
            continue
        xmin, ymin, xmax, ymax = box
        xmin, xmax = int(xmin * w), int(xmax * w)  # 反归一化
        ymin, ymax = int(ymin * h), int(ymax * h)
        plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                          edgecolor='r', linewidth=2, fill=False))

    plt.title("Predicted Bounding Boxes")
    plt.show()

# ========================== 运行可视化 ==========================
# 选一张测试图片
sample_image_path = test_image_paths[0]  # 取测试集中的第一张图片
sample_image = cv2.imread(sample_image_path)
sample_image = cv2.resize(sample_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
sample_image = sample_image.astype(np.float32) / 255.0  # 归一化
sample_image = np.expand_dims(sample_image, axis=0)  # 增加 batch 维度

# 让模型预测
sample_pred_boxes = model.predict(sample_image)[0]  # 取第一个样本的预测结果
print("Predicted Bounding Boxes:", sample_pred_boxes)

# 可视化预测框
visualize_predictions(sample_image_path, sample_pred_boxes)