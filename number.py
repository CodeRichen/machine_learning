import keras
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.enable_eager_execution()


def load_my_digits(folder_path):
    X = []
    y = []
    for fname in os.listdir(folder_path):
        if fname.endswith('.png'):
            label = int(fname.split('_')[1].split('.')[0])  # 例如 digit_5.png ⇒ 5
            img = Image.open(os.path.join(folder_path, fname)).convert('L').resize((28, 28))
            img_arr = 255 - np.array(img)
            img_arr = img_arr / 255.0
            X.append(img_arr.reshape(28, 28, 1))
            y.append(label)
    return np.array(X), np.array(y)

# 定義模型檔案名稱
model_path = "output/mnist_model.h5"

# 設定圖片大小
img_rows, img_cols = 28, 28

# 如果已有模型就載入
if os.path.exists(model_path):
    print("找到模型，直接載入")
    model = load_model(model_path)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # 重新編譯模型

else:
    print("未找到模型，開始訓練")
    # 先切割出驗證集


    # 載入資料集
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # reshape 成 CNN 格式：28x28x1
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    # 正規化
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
     # One-hot 編碼
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    n_out=len(y_train[0])
    
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  # 抑制過擬合
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # 再加一層 Dropout
    model.add(Dense(n_out, activation='softmax'))
    
    model.summary()
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    X_small = X_train[:60000 ] 
    y_small = y_train[:60000 ] # 用前x張訓練圖片
    X_train_small, X_val, y_train_small, y_val = train_test_split(
        X_small, y_small, test_size=0.1, random_state=42)

    datagen = ImageDataGenerator(
    rotation_range=10,         # 隨機旋轉 ±10度
    zoom_range=0.1,            # 隨機縮放 ±10%
    width_shift_range=0.1,     # 水平平移 ±10%
    height_shift_range=0.1     # 垂直平移 ±10%
    )
    datagen.fit(X_small)

    model.fit(
    datagen.flow(X_train_small, y_train_small, batch_size=64),
    epochs=15,
    validation_data=(X_val, y_val),
    verbose=2
    ) # batch_size越小epochs越大正確率越高
    #進度調顯示的數字就是X_train/batch_size
    model.save(model_path) # 儲存模型
# 之後用 keras.models.load_model 載入

# ====== 讀取自己的手寫圖像來進行微調 ======

folder = "output/digit"  # 放你自製圖的資料夾，例如 digit_0.png, digit_1.png, ...
if os.path.exists(folder):
    X_personal, y_personal = load_my_digits(folder)
    print(f"載入 {len(X_personal)} 張自己的圖像")

    y_personal_cat = to_categorical(y_personal, num_classes=10)

    # 微調模型（訓練幾輪）
    datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
    )
    datagen.fit(X_personal)

    model.fit(datagen.flow(X_personal, y_personal_cat, batch_size=8),
            epochs=10, verbose=2)

    # 儲存微調後的模型（可選）
    model.save("output/mnist_model_finetuned.h5")
    print("✅ 微調完成，模型已儲存為 mnist_model_finetuned.h5")
else:
    print("⚠️ 找不到個人圖像資料夾，跳過微調階段")

# 載入自己的手寫圖像

img = Image.open('output/digit.png').convert('L')  # 轉灰階
img = img.resize((28, 28))                    # MNIST 大小
img_arr = np.array(img)
img_arr = 255 - img_arr                       # 白底黑字轉成黑底白字（MNIST格式）
img_arr = img_arr / 255.0                     # 正規化
img_arr = img_arr.reshape(1, 28, 28, 1)
plt.imshow(img_arr[0].reshape(28, 28), cmap='gray')
plt.title("your digit")
plt.axis('off')
plt.show()
# 預測
prediction = model.predict(img_arr)
predicted_digit = np.argmax(prediction)
print("模型預測數字：", predicted_digit)

# 隨機挑選 10 張訓練圖片
# indices = np.random.choice(len(X_train), 10, replace=False)
# sample_images = X_train[indices]
# sample_labels = y_train[indices]
# true_classes = np.argmax(sample_labels, axis=1)

# # 預測
# predictions = model.predict(sample_images)
# predicted_classes = np.argmax(predictions, axis=1)

# # 顯示結果
# for i in range(10):
#     plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
#     plt.title(f"guess: {predicted_classes[i]} real: {true_classes[i]} ")
#     plt.axis('off')
#     plt.show()
    
# 顯示測試集評估結果
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1).astype('float32') / 255
y_test = to_categorical(y_test)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
