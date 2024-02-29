Bước 1: Load file X_train, Y_train, X_test

Bước 2: 
    - Chuyển ảnh về dạng 28*28 (Chuyển mảng dữ liệu của ảnh về dạng 28 * 28)
    - Chuyển dữ liệu X_train,X_test về khoảng 0 và 1
    - Chuyển dữ liệu Y_train từ label sang encode (Từ số nguyên về dạng mã nhị phân)

Bước 3:
    - Định nghĩa hàm loss: loss_fn = tf.keras.losses.CategoricalCrossentropy()
    - Định nghĩa thuật toán tối ưu: optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.01)

Bước 4: Xây dựng mô hình CNN
    - Layer 1: Dùng Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same", strides=(1, 1) ),: Lớp Conv2D đầu tiên với 16 bộ lọc,  kích thước kernel (3, 3), sử dụng hàm kích hoạt ReLU giúp mô hình học được các đặc trưng phi tuyến tính và giảm hiện tượng vanishing gradient.
    - Layer 2: Lớp MaxPooling2D sau Conv2D đầu tiên giảm kích thước đầu ra từ (None, 28, 28, 16) xuống còn (None, 14, 14, 16).
    - Layer 3: Dùng Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same", strides=(1, 1) ),: Lớp có 32 bộ lọc,  kích thước kernel (3, 3), sử dụng hàm kích hoạt ReLU giúp mô hình học được các đặc trưng phi tuyến tính và giảm hiện tượng vanishing gradient.
    - Layer 4: Dùng Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same", strides=(1, 1) ),: Lớp có 64 bộ lọc,  kích thước kernel (3, 3), sử dụng hàm kích hoạt ReLU giúp mô hình học được các đặc trưng phi tuyến tính và giảm hiện tượng vanishing gradient.
    - Layer 5: MaxPooling2D (pool_size=(2, 2)): Lớp MaxPooling2D  giảm kích thước đầu ra từ (None, 14, 14, 64) xuống còn (None, 7, 7, 64).
    - Layer 6: Lớp Flatten chuyển đổi tensor từ đa chiều thành mảng một chiều với kích thước (None, 3136).
    - Layer 7: Lớp Dense với 64 đơn vị (units) và sử dụng hàm kích hoạt ReLU (Rectified Linear Unit).
    - Layer 8: Lớp Dense cuối cùng có 10 đơn vị, tương ứng với số lượng lớp đầu ra.

Bước 5: Train (dữ liệu sau khi train được bỏ vào file csv)


