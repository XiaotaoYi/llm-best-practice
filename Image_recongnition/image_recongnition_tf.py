import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt

# 加载 CIFAR-10 数据集
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# 归一化像素值到 [0, 1] 范围
train_images = train_images / 255.0
test_images = test_images / 255.0

# 类别名称
class_names = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

model = models.Sequential([
    # 第一层：卷积层 + 池化层
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    # 第二层：卷积层 + 池化层
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # 第三层：卷积层
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # 展平层：将多维特征图转换为一维向量
    layers.Flatten(),
    
    # 全连接层：提取高级特征
    layers.Dense(64, activation='relu'),
    
    # 输出层：10 个类别（softmax 激活）
    layers.Dense(10, activation='softmax')
])

# 打印模型结构
model.summary()


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(train_images, train_labels, 
                    epochs=10, 
                    batch_size=64,
                    validation_split=0.1)  # 使用 10% 的训练数据作为验证集

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\n测试集准确率: {test_acc:.4f}")

# 绘制准确率曲线
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('训练和验证准确率')
plt.legend(loc='lower right')
plt.show()

# 绘制损失曲线
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练和验证损失')
plt.legend(loc='upper right')
plt.show()

# 预测测试集中的前 10 张图像
predictions = model.predict(test_images[:10])

# 可视化预测结果
plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    image = test_images[i]
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i][0]
    color = 'green' if predicted_label == true_label else 'red'
    plt.imshow(image)
    plt.xlabel(f"{class_names[predicted_label]} ({class_names[true_label]})", color=color)
plt.show()