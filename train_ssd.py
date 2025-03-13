import os
import xml.etree.ElementTree as ET
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Lambda
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tqdm import tqdm

# ========================== 设定数据路径 ==========================
DATASET_DIR = "dataset_final"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

# 解析 PASCAL VOC 数据
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
MAX_BOXES = 10  # 设定最多检测 10 个目标

def parse_voc_annotation(xml_file):
    """解析 PASCAL VOC 标注"""
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
    if isinstance(image_path, tf.Tensor):
        image_path = image_path.numpy().decode("utf-8")
    if isinstance(annotation_path, tf.Tensor):
        annotation_path = annotation_path.numpy().decode("utf-8")

    print(f"loading image: {image_path}")
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))  # SSD 需要 224x224 输入
    image = image.astype(np.float32) / 255.0  # 归一化

    boxes = parse_voc_annotation(annotation_path)
    return image, boxes

def get_image_label_paths(image_dir, annotation_dir):
    """获取所有图片和标注路径"""
    image_paths, annotation_paths = [], []
    for filename in os.listdir(annotation_dir):
        xml_path = os.path.join(annotation_dir, filename)
        img_path = os.path.join(image_dir, filename.replace(".xml", ".jpg"))
        if os.path.exists(img_path):
            image_paths.append(img_path)
            annotation_paths.append(xml_path)
    return image_paths, annotation_paths

# 获取数据路径
train_image_paths, train_annotation_paths = get_image_label_paths(
    os.path.join(TRAIN_DIR, "images"), os.path.join(TRAIN_DIR, "annotations")
)
val_image_paths, val_annotation_paths = get_image_label_paths(
    os.path.join(VAL_DIR, "images"), os.path.join(VAL_DIR, "annotations")
)

# ========================== 3️⃣ 生成数据集 ==========================
def data_generator(image_paths, annotation_paths, batch_size=4):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, annotation_paths))

    def load_data(image_path, annotation_path):
        image, boxes = tf.py_function(load_image_and_annotation, [image_path, annotation_path], [tf.float32, tf.float32])
        image.set_shape((IMAGE_WIDTH, IMAGE_HEIGHT, 3))
        boxes.set_shape((MAX_BOXES, 4))  # 固定 shape
        return image, boxes

    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = data_generator(train_image_paths, train_annotation_paths, batch_size=4)
val_dataset = data_generator(val_image_paths, val_annotation_paths, batch_size=4)

# ========================== 4️⃣ SSD 目标检测模型 ==========================
def build_ssd():
    base_model = MobileNetV2(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3), include_top=False)
    x = base_model.output
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Flatten()(x)
    x = Dense(MAX_BOXES * 4, activation='sigmoid')(x)  # 预测 10 个框，每个框 4 个坐标
    x = Lambda(lambda y: tf.reshape(y, (-1, MAX_BOXES, 4)))(x)
    return Model(inputs=base_model.input, outputs=x)

model = build_ssd()
model.compile(optimizer="adam", loss=Huber(delta=1.0))

# ========================== 5️⃣ 进度条 & 训练 ==========================
class TQDMProgressBar(Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params["epochs"]
        self.progress_bar = tqdm(total=self.epochs, desc="Training Progress")

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss", 0)
        val_loss = logs.get("val_loss", 0)
        self.progress_bar.update(1)
        print(f"Epoch {epoch+1}/{self.epochs} - Loss: {loss:.4f} - Val Loss: {val_loss:.4f}")

    def on_train_end(self, logs=None):
        self.progress_bar.close()

# ========================== 6️⃣ 训练 ==========================
EPOCHS = 20

# 训练模型
print("🚀 开始训练 SSD 模型...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[ModelCheckpoint("ssd_lego.h5", save_best_only=True), TQDMProgressBar()],
    verbose=2  # 关闭默认 Keras 进度条
)

print("✅ 训练完成，模型已保存为 ssd_lego.h5")

# ========================== 7️⃣ 绘制 Loss 曲线 ==========================
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Loss Curve")
plt.savefig("loss_curve.png")
print("📉 Loss 曲线已保存为 loss_curve.png")