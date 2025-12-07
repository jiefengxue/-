import time  
import mindspore as ms
from mindspore import dataset, nn, ops, Tensor, save_checkpoint, load_checkpoint, load_param_into_net
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
from PIL import Image, ImageDraw
import cv2
from puzzle_core import *


ms.set_context(mode=ms.PYNATIVE_MODE)
CHECKPOINT_PATH = "./mnist_net.ckpt" #checkpoint模型参数保存路径

#视觉模型定义
class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(28*28, 64)
        self.fc2 = nn.Dense(64, 64)
        self.fc3 = nn.Dense(64, 64)
        self.fc4 = nn.Dense(64, 10)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(axis=1)

    def construct(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.log_softmax(self.fc4(x))
        return x

#数据集
def load_mnist_npz(is_train):
    npz_path = f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz"
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"未找到mnist.npz文件：{npz_path}")
    with np.load(npz_path) as f:
        if is_train:
            images, labels = f["x_train"], f["y_train"]      #训练集
        else:
            images, labels = f["x_test"], f["y_test"]         #测试集
    images = images.astype(np.float32) / 255.0
    images = np.expand_dims(images, axis=1)
    labels = labels.astype(np.int32)
    return images, labels

def get_data_loader(is_train):
    images, labels = load_mnist_npz(is_train)
    mnist_dataset = dataset.NumpySlicesDataset(
        data={"image": images, "label": labels},
        shuffle=True
    )
    mnist_dataset = mnist_dataset.batch(batch_size=15)
    return mnist_dataset

def evaluate(test_data, net):#模型评估函数
    n_correct = 0
    n_total = 0
    for data in test_data.create_dict_iterator():
        x = data["image"]
        y = data["label"]
        outputs = net.construct(x.reshape(-1, 28*28))
        for i in range(outputs.shape[0]):
            output_single = outputs[i:i+1]
            pred = ops.argmax(output_single, dim=1).asnumpy()[0]
            label = y[i].asnumpy() if isinstance(y[i], ms.Tensor) else y[i]
            if pred == label:
                n_correct += 1
            n_total += 1
    test_data.reset()
    return n_correct / n_total

#判断单/两位数  #长宽差大于20 两位数（切分识别） #长宽差小于20 单数字（直接识别）
def auto_judge_and_preprocess(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"图片不存在：{img_path}")
    
    # 1. 读取图片，获取尺寸
    img = Image.open(img_path).convert("L")
    w, h = img.size
    size_diff = abs(w - h)
    print(f"\n图片尺寸：宽={w}，高={h}，长宽差={size_diff}")
    
    # 2. 判断类型
    if size_diff > 20:
        print("判定为：两位数（长宽差＞20）")
        # 两位数处理：等比例缩放，填充，切分
        target_w, target_h = 56, 28
        original_ratio = w / h
        target_ratio = target_w / target_h
        
        if original_ratio > target_ratio:
            new_w = target_w
            new_h = int(new_w / original_ratio)
        else:
            new_h = target_h
            new_w = int(new_h * original_ratio)
        
        img_scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        img_padded = Image.new("L", (target_w, target_h), 0)
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        img_padded.paste(img_scaled, (offset_x, offset_y))
        img_np = np.array(img_padded, dtype=np.float32) / 255.0
        
        # 黑底白字校验
        if np.mean(img_np) > 0.5:
            img_np = 1.0 - img_np
            print("图片为白底黑字，已反转为黑底白字")
        
        # 切分十位/个位
        ten_digit_np = img_np[:, :28]
        one_digit_np = img_np[:, 28:]
        
        return {
            "type": "double",
            "raw_img": img,
            "ten_digit_tensor": ms.Tensor(ten_digit_np.reshape(1, 28*28), dtype=ms.float32),
            "one_digit_tensor": ms.Tensor(one_digit_np.reshape(1, 28*28), dtype=ms.float32),
            "ten_digit_np": ten_digit_np,
            "one_digit_np": one_digit_np,
            "padded_img": img_padded,
            "final_num": None,  # 预留最终识别结果
            "prob": (0, 0)      # 预留概率
        }
    else:
        print("判定为：单数字（长宽差小于20）")
        # 单数字处理：等比例缩放→28×28
        img_scaled = img.resize((28, 28), Image.Resampling.LANCZOS)
        img_np = np.array(img_scaled, dtype=np.float32) / 255.0
        
        # 黑底白字校验
        if np.mean(img_np) > 0.5:
            img_np = 1.0 - img_np
            print("图片为白底黑字，已反转为黑底白字")
        
        return {
            "type": "single",
            "raw_img": img,
            "img_tensor": ms.Tensor(img_np.reshape(1, 28*28), dtype=ms.float32),
            "img_np": img_np,
            "final_num": None,  # 预留最终识别结果
            "prob": 0           # 预留概率
        }

#统一识别函数
def infer_auto(net, img_path):
    try:
        preprocess_result = auto_judge_and_preprocess(img_path)
        
        net.set_train(False)
        if preprocess_result["type"] == "double":
            # 两位数识别 - CPU计时
            start_ten = time.time()
            ten_output = net.construct(preprocess_result["ten_digit_tensor"])
            end_ten = time.time()
            ten_time = (end_ten - start_ten) * 1000  # 转换为毫秒
            
            start_one = time.time()
            one_output = net.construct(preprocess_result["one_digit_tensor"])
            end_one = time.time()
            one_time = (end_one - start_one) * 1000  # 转换为毫秒
            
            ten_digit = ops.argmax(ten_output, dim=1).asnumpy()[0]
            ten_prob = np.exp(ten_output.asnumpy()[0])[ten_digit]
            
            one_digit = ops.argmax(one_output, dim=1).asnumpy()[0]
            one_prob = np.exp(one_output.asnumpy()[0])[one_digit]
            
            final_num = ten_digit * 10 + one_digit
            
            # 打印CPU单次两位数耗时
            print(f"【CPU(CheckPoint)】识别结果：{final_num}（两位数）| 十位耗时：{ten_time:.2f}ms | 个位耗时：{one_time:.2f}ms | 总耗时：{ten_time+one_time:.2f}ms")
            
            # 保存结果到字典
            preprocess_result["final_num"] = final_num
            preprocess_result["prob"] = (ten_prob, one_prob)
            return final_num
        else:
            # 单数字识别 - CPU计时
            start_single = time.time()
            output = net.construct(preprocess_result["img_tensor"])
            end_single = time.time()
            single_time = (end_single - start_single) * 1000  # 转换为毫秒
            
            final_num = ops.argmax(output, dim=1).asnumpy()[0]
            prob = np.exp(output.asnumpy()[0])[final_num]
            
            # 打印CPU单次单数字耗时
            print(f"【CPU(CheckPoint)】识别结果：{final_num}（单数字）| 推理耗时：{single_time:.2f}ms")
            
            # 保存结果到字典
            preprocess_result["final_num"] = final_num
            preprocess_result["prob"] = prob
            return final_num
    except Exception as e:
        print(f"识别失败：{str(e)}")
        return None

#抠黑底块+坐标排序
def extract_black_bg_blocks(img_path, save_dir="./15_puzzle_blocks"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 读取图片
    img = Image.open(img_path).convert("L")
    img_cv = np.array(img)
    img_rgb = Image.open(img_path).convert("RGB")  # 仅用于可视化轮廓
    
    # 二值化检测黑底
    _, binary = cv2.threshold(img_cv, 50, 255, cv2.THRESH_BINARY_INV)
    
    # 查找外轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 过滤小轮廓
    min_area = 100
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    if len(valid_contours) != 16:
        print(f"检测到{len(valid_contours)}个黑底块（预期16个），继续处理")
    
    # 抠出每个黑底块并记录信息
    blocks_info = []
    for idx, cnt in enumerate(valid_contours):
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 完整抠出黑底块
        block = img.crop((x, y, x+w, y+h))
        block_path = f"{save_dir}/block_{idx}.png"
        block.save(block_path)
        
        # 计算中心坐标
        center_x = x + w/2
        center_y = y + h/2
        
        blocks_info.append({
            "path": block_path,
            "center_x": center_x,
            "center_y": center_y,
            "original_size": (w, h),
            "block": block,
            "pred": 0  # 预留识别结果
        })
        
        # 可视化轮廓
        draw = ImageDraw.Draw(img_rgb)
        draw.rectangle((x, y, x+w, y+h), outline="red", width=2)
        draw.text((center_x-10, center_y-10), str(idx), fill="red")
    
    # 保存轮廓可视化图
    img_rgb.save(f"{save_dir}/contours.png")
    print(f"黑底块轮廓图已保存：{save_dir}/contours.png")
    
    return blocks_info

def sort_blocks_to_4x4(blocks_info):#4x4排序
    # 先按y坐标分组（行）
    blocks_info.sort(key=lambda x: x["center_y"])
    rows = []
    row_size = len(blocks_info) // 4
    
    for i in range(4):
        start = i * row_size
        end = start + row_size
        row_blocks = blocks_info[start:end]
        
        # 每行内按x坐标排序（列）
        row_blocks.sort(key=lambda x: x["center_x"])
        rows.append(row_blocks)
    
    return rows

def parse_puzzle_from_image(img_path):#图片识别入口函数
    try:
        # 加载训练好的模型
        if os.path.exists(CHECKPOINT_PATH):
            net = Net()
            param_dict = load_checkpoint(CHECKPOINT_PATH)
            load_param_into_net(net, param_dict)
        else:
            return None, "模型文件不存在，请先训练模型！"
        
        # 抠出黑底块
        blocks_info = extract_black_bg_blocks(img_path)
        if len(blocks_info) == 0:
            return None, "未检测到任何黑底块！"
        
        # CPU批量计时 - 开始
        batch_start = time.time()
        
        # 逐块识别
        for idx, block_info in enumerate(blocks_info):
            pred = infer_auto(net, block_info["path"])
            blocks_info[idx]["pred"] = pred if pred is not None else 0
        
        # CPU批量计时 - 结束
        batch_end = time.time()
        batch_total_time = (batch_end - batch_start) * 1000  # 总耗时（毫秒）
        batch_avg_time = batch_total_time / len(blocks_info)  # 平均耗时（毫秒/个）
        
        # 打印CPU批量耗时汇总
        print("\n" + "="*60)
        print(f"【CPU(CheckPoint)】批量识别耗时汇总")
        print(f"识别数字总数：{len(blocks_info)} 个")
        print(f"批量总耗时：{batch_total_time:.2f} ms")
        print(f"单数字平均耗时：{batch_avg_time:.2f} ms")
        print("="*60 + "\n")
        
        # 排序生成4×4矩阵
        sorted_rows = sort_blocks_to_4x4(blocks_info)
        puzzle_matrix = []
        for row in sorted_rows:
            row_pred = [block["pred"] for block in row]
            puzzle_matrix.append(row_pred)
        
        puzzle_matrix = np.array(puzzle_matrix)
        
        # 仅绘制最终汇总可视化图
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        fig.suptitle("15_puzzle_result", fontsize=16)
        for row_idx, row in enumerate(sorted_rows):
            for col_idx, block_info in enumerate(row):
                block = Image.open(block_info["path"])
                axes[row_idx, col_idx].imshow(block, cmap="gray")
                axes[row_idx, col_idx].set_title(f"{block_info['pred']}", fontsize=12)
                axes[row_idx, col_idx].axis("off")
        plt.tight_layout()
        plt.show()
        
        # 返回信息包含CPU耗时
        return puzzle_matrix, f"识别成功！\n识别结果矩阵：\n{puzzle_matrix}\n\n【CPU(CheckPoint)】批量总耗时：{batch_total_time:.2f}ms | 平均耗时：{batch_avg_time:.2f}ms/个"
    
    except Exception as e:
        return None, f"识别出错：{str(e)}"

def train_model():#模型训练入口函数
    try:
        train_data = get_data_loader(is_train=True)
        test_data = get_data_loader(is_train=False)
        net = Net()
        print("initial accuracy:", evaluate(test_data, net))
        optimizer = nn.Adam(params=net.trainable_params(), learning_rate=0.001)
        
        for epoch in range(5):
            for data in train_data.create_dict_iterator():
                x = data["image"]
                y = data["label"]
                def forward_fn():
                    output = net.construct(x.reshape(-1, 28*28))
                    loss = nn.NLLLoss()(output, y)
                    return loss
                grad_fn = ops.value_and_grad(forward_fn, None, net.trainable_params())
                loss, grads = grad_fn()
                optimizer(grads)
            print(f"epoch {epoch} accuracy:", evaluate(test_data, net))
            train_data.reset()
        
        save_checkpoint(net, CHECKPOINT_PATH)
        print(f"\n模型保存至：{CHECKPOINT_PATH}")
        
        # 可视化测试集
        n = 0
        for data in test_data.create_dict_iterator():
            if n > 3:
                break
            x = data["image"]
            x_single = x[0:1]
            x_flat = x_single.reshape(1, 28*28)
            output = net.construct(x_flat)
            predict = ops.argmax(output, dim=1).asnumpy()[0]
            
            plt.figure(n)
            plt.imshow(x_single.reshape(28, 28).asnumpy(), cmap="gray")
            plt.title(f"prediction: {predict}")
            n += 1
        plt.show()
        return True
    
    except Exception as e:
        print(f"训练出错：{str(e)}")
        raise e