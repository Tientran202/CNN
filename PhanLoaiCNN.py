import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import csv


# Load file X_train, Y_train, X_test
X_train = np.load("X_train.npy")
Y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")

print(X_train[1])
# Chuyển đổi về hình dạng 28x28
X_train = X_train.reshape(-1, 28, 28)
X_test = X_test.reshape(-1, 28, 28)
print(X_train[1])


# chuyển dữ liệu X_train, X_test về khoảng 0 và 1
from tensorflow.keras.utils import to_categorical

X_train, X_test = X_train / 255.0, X_test / 255.0
# chuyển dữ liệu y_train từ label sang encode
Y_train = to_categorical(Y_train)


num_classes = 10
input_shape = (28, 28, 1)

# định nghĩa hàm loss
loss_fn = tf.keras.losses.CategoricalCrossentropy()
# định nghĩa thuật toán tối ưu
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.01)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(
            16, kernel_size=(3, 3), activation="relu", padding="same", strides=(1, 1)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(
            32, kernel_size=(3, 3), activation="relu", padding="same", strides=(1, 1)
        ),
        tf.keras.layers.Conv2D(
            64, kernel_size=(3, 3), activation="relu", padding="same", strides=(1, 1)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)
model.summary()

# khỏi tạo mô hình với loss, hàm tối ưu, và thông số đo hiệu suất của mô hình
model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

# nơi lưu trữ file mô hình CNN
weights_filepath = "./weights/"
# Tạo một callbback để lưu mô hình theo một cách mong muốn
callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=weights_filepath, monitor="val_loss", verbose=1, save_best_only=True
)

# bắt đầu training
his = model.fit(
    X_train,
    Y_train,
    epochs=50,
    batch_size=128,
    validation_split=0.2,
    callbacks=callback,
)

model = tf.keras.models.load_model("./weights/")
# kiểm tra mô hình đã load chính xác
model.summary()

import matplotlib.pyplot as plt

# chọn một bức hình và vẽ nó lên
input_image = X_test[666]
plt.imshow(input_image, cmap=plt.get_cmap("gray"))
input_image = input_image / 255.0

# chiều của bức ảnh test
print("shape của 1 bức ảnh", input_image.shape)
# shape của 1 bức ảnh (28, 28)
input_image = np.expand_dims(input_image, axis=0)
# tăng thêm 1 chiều, định nghĩa cho số lượng mẫu
input_image = np.expand_dims(input_image, axis=3)
# tăng thêm 1 chiều, định nghĩa cho số kênh màu ảnh
print("shape phù hợp với mô hình là 4 chiều", input_image.shape)
# shape phù hợp với mô hình là 4 chiều (1, 28, 28, 1)
output = model.predict(input_image)
print(output)
print("số dự đoán là :", output.argmax())


# vẽ đường loss trên tập train và tập validation
plt.plot(his.history["val_loss"], c="coral", label="validation loss line")
plt.plot(his.history["loss"], c="blue", label="train loss line")
legend = plt.legend(loc="upper center")
plt.show()

# vẽ đường accuracy trên tập train và tập validation
plt.plot(his.history["val_accuracy"], c="coral", label="validation accuracy line")
plt.plot(his.history["accuracy"], c="blue", label="train accuracy line")
legend = plt.legend(loc="lower center")
plt.show()


# Load file mô hình đã huấn luyện
model = tf.keras.models.load_model("./weights/")

# Đánh giá mô hình trên tập test
loss, acc = model.evaluate(X_test, verbose=0)
print("loss tập test = ", loss, "| accuracy tập test = ", acc)

data = [["index", "label"]]


def append_dataPT(x, y):
    data.append([x, y])


# lấy 1 hình ảnh bất kỳ ở tập test và dự đoán
for i in range(len(X_test)):
    print(f" \n {i} ")
    input_image = X_test[i]
    # plt.imshow(input_image, cmap=plt.get_cmap("gray"))
    # print("shape của 1 bức ảnh", input_image.shape)
    input_image = np.expand_dims(input_image, axis=0)
    # print("shape phù hợp với mô hình là 3 chiều", input_image.shape)
    output = model.predict(input_image)
    # print("số dự đoán là :", output.argmax())
    # plt.show()
    append_dataPT(i, output.argmax())

with open("123TGMT2002_9H53_CNN_1.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)
