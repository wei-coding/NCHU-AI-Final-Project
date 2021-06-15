# 貪食蛇AI

## 進度表

### 目前已完成部份：

- [x] 遊戲本體
- [x] 狀態輸出
- [x] AI接入操作點


### 未完成：

- [x] Deep Q Learning


### 各model解釋:

* model1: 11狀態，一次讀3個frame，隱藏層為64, 128, 64
* model2: 11狀態，一次讀3個frame，隱藏層為256
* model3: 11狀態，一次讀3個frame，隱藏層為512
* model4: 11狀態，一次讀3個frame，隱藏層為512，activation改為softmax
* model5: 14狀態(原本11 + 上左右的身體距離倒數)，一次個frame，隱藏層[128, 128]