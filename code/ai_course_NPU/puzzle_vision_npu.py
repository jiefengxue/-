import time  
import mindspore as ms
from mindspore import dataset, nn, ops, Tensor, save_checkpoint, load_checkpoint, load_param_into_net
from mindspore import context
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
from PIL import Image, ImageDraw
import cv2
from puzzle_core import *

try:
    context.set_context(
        mode=context.GRAPH_MODE,  
        device_target="Ascend",
        device_id=0
    )
    DEVICE_TARGET = "Ascend"
    print("成功加载昇腾310B NPU上下文")
except Exception as e:
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    DEVICE_TARGET = "CPU"
    print(f"NPU加载失败，使用CPU：{str(e)}")

CHECKPOINT_PATH = "./mnist_net.ckpt"
ms.set_context(mode=ms.PYNATIVE_MODE)


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.fc1_weight = ms.Parameter(ms.Tensor(np.random.randn(64, 28*28), ms.float16), name="fc1.weight")
        self.fc1_bias = ms.Parameter(ms.Tensor(np.random.randn(64), ms.float16), name="fc1.bias")
        self.fc2_weight = ms.Parameter(ms.Tensor(np.random.randn(64, 64), ms.float16), name="fc2.weight")
        self.fc2_bias = ms.Parameter(ms.Tensor(np.random.randn(64), ms.float16), name="fc2.bias")
        self.fc3_weight = ms.Parameter(ms.Tensor(np.random.randn(64, 64), ms.float16), name="fc3.weight")
        self.fc3_bias = ms.Parameter(ms.Tensor(np.random.randn(64), ms.float16), name="fc3.bias")
        self.fc4_weight = ms.Parameter(ms.Tensor(np.random.randn(10, 64), ms.float16), name="fc4.weight")
        self.fc4_bias = ms.Parameter(ms.Tensor(np.random.randn(10), ms.float16), name="fc4.bias")
        
        self.matmul = ops.MatMul(transpose_b=True)
        self.add = ops.Add()
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(axis=1)
        self.cast=ops.Cast()

    def construct(self, x):
        x=self.cast(x,ms.float16)
        x = self.relu(self.add(self.matmul(x, self.fc1_weight), self.fc1_bias))
        x = self.relu(self.add(self.matmul(x, self.fc2_weight), self.fc2_bias))
        x = self.relu(self.add(self.matmul(x, self.fc3_weight), self.fc3_bias))
        x = self.log_softmax(self.add(self.matmul(x, self.fc4_weight), self.fc4_bias))
        return self.cast(x,ms.float32)

def init_npu_infer_model():
    if os.path.exists(CHECKPOINT_PATH):
        net = Net()
        param_dict = load_checkpoint(CHECKPOINT_PATH)
        load_param_into_net(net, param_dict)
        net.set_train(False)
        print("加载CheckPoint模型（Ascend/CPU兼容模式）")
        return net
    else:
        raise FileNotFoundError("模型文件不存在，请先训练模型！")

def load_mnist_npz(is_train):
    npz_path = f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz"
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"未找到mnist.npz文件：{npz_path}")
    with np.load(npz_path) as f:
        if is_train:
            images, labels = f["x_train"], f["y_train"]      
        else:
            images, labels = f["x_test"], f["y_test"]         
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

def evaluate(test_data, net):
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

def auto_judge_and_preprocess(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"图片不存在：{img_path}")
    
    img = Image.open(img_path).convert("L")
    w, h = img.size
    size_diff = abs(w - h)
    print(f"\n图片尺寸：宽={w}，高={h}，长宽差={size_diff}")
    
    if size_diff > 20:
        print("判定为：两位数（长宽差＞20）")
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
        
        if np.mean(img_np) > 0.5:
            img_np = 1.0 - img_np
            print("图片为白底黑字，已反转为黑底白字")
        
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
            "final_num": None,
            "prob": (0, 0)
        }
    else:
        print("判定为：单数字（长宽差小于20）")
        img_scaled = img.resize((28, 28), Image.Resampling.LANCZOS)
        img_np = np.array(img_scaled, dtype=np.float32) / 255.0
        
        if np.mean(img_np) > 0.5:
            img_np = 1.0 - img_np
            print("图片为白底黑字，已反转为黑底白字")
        
        return {
            "type": "single",
            "raw_img": img,
            "img_tensor": ms.Tensor(img_np.reshape(1, 28*28), dtype=ms.float32),
            "img_np": img_np,
            "final_num": None,
            "prob": 0
        }

def infer_auto(net, img_path):
    try:
        preprocess_result = auto_judge_and_preprocess(img_path)
        
        net.set_train(False)
        if preprocess_result["type"] == "double":
            start_ten = time.time()
            ten_output = net.construct(preprocess_result["ten_digit_tensor"])
            end_ten = time.time()
            ten_time = (end_ten - start_ten) * 1000
            
            start_one = time.time()
            one_output = net.construct(preprocess_result["one_digit_tensor"])
            end_one = time.time()
            one_time = (end_one - start_one) * 1000
            
            ten_digit = ops.argmax(ten_output, dim=1).asnumpy()[0]
            ten_prob = np.exp(ten_output.asnumpy()[0][ten_digit])
            one_digit = ops.argmax(one_output, dim=1).asnumpy()[0]
            one_prob = np.exp(one_output.asnumpy()[0][one_digit])
            final_num = ten_digit * 10 + one_digit
            
            print(f"【{DEVICE_TARGET}】识别结果：{final_num}（两位数）| 十位耗时：{ten_time:.2f}ms | 个位耗时：{one_time:.2f}ms | 总耗时：{ten_time+one_time:.2f}ms")
            
            preprocess_result["final_num"] = final_num
            preprocess_result["prob"] = (ten_prob, one_prob)
            return final_num
        else:
            start_single = time.time()
            output = net.construct(preprocess_result["img_tensor"])
            end_single = time.time()
            single_time = (end_single - start_single) * 1000
            
            final_num = ops.argmax(output, dim=1).asnumpy()[0]
            prob = np.exp(output.asnumpy()[0])[final_num]
            
            print(f"【{DEVICE_TARGET}】识别结果：{final_num}（单数字）| 推理耗时：{single_time:.2f}ms")
            
            preprocess_result["final_num"] = final_num
            preprocess_result["prob"] = prob
            return final_num
    except Exception as e:
        print(f"识别失败：{str(e)}")
        return None

def extract_black_bg_blocks(img_path, save_dir="./15_puzzle_blocks"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    img = Image.open(img_path).convert("L")
    img_cv = np.array(img)
    img_rgb = Image.open(img_path).convert("RGB")
    
    _, binary = cv2.threshold(img_cv, 50, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = 100
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    if len(valid_contours) != 16:
        print(f"检测到{len(valid_contours)}个黑底块（预期16个），继续处理")
    
    blocks_info = []
    for idx, cnt in enumerate(valid_contours):
        x, y, w, h = cv2.boundingRect(cnt)
        
        block = img.crop((x, y, x+w, y+h))
        block_path = f"{save_dir}/block_{idx}.png"
        block.save(block_path)
        
        center_x = x + w/2
        center_y = y + h/2
        
        blocks_info.append({
            "path": block_path,
            "center_x": center_x,
            "center_y": center_y,
            "original_size": (w, h),
            "block": block,
            "pred": 0
        })
        
        draw = ImageDraw.Draw(img_rgb)
        draw.rectangle((x, y, x+w, y+h), outline="red", width=2)
        draw.text((center_x-10, center_y-10), str(idx), fill="red")
    
    img_rgb.save(f"{save_dir}/contours.png")
    print(f"黑底块轮廓图已保存：{save_dir}/contours.png")
    
    return blocks_info

def sort_blocks_to_4x4(blocks_info):
    blocks_info.sort(key=lambda x: x["center_y"])
    rows = []
    row_size = len(blocks_info) // 4
    
    for i in range(4):
        start = i * row_size
        end = start + row_size
        row_blocks = blocks_info[start:end]
        
        row_blocks.sort(key=lambda x: x["center_x"])
        rows.append(row_blocks)
    
    return rows

def parse_puzzle_from_image(img_path):
    try:
        try:
            net = init_npu_infer_model()
        except Exception as e:
            return None, f"模型加载失败：{str(e)}"
        
        blocks_info = extract_black_bg_blocks(img_path)
        if len(blocks_info) == 0:
            return None, "未检测到任何黑底块！"
        
        batch_start = time.time()
        
        for idx, block_info in enumerate(blocks_info):
            pred = infer_auto(net, block_info["path"])
            blocks_info[idx]["pred"] = pred if pred is not None else 0
        
        batch_end = time.time()
        batch_total_time = (batch_end - batch_start) * 1000
        batch_avg_time = batch_total_time / len(blocks_info)
        
        print("\n" + "="*60)
        print(f"【{DEVICE_TARGET}】批量识别耗时汇总")
        print(f"识别数字总数：{len(blocks_info)} 个")
        print(f"批量总耗时：{batch_total_time:.2f} ms")
        print(f"单数字平均耗时：{batch_avg_time:.2f} ms")
        print("="*60 + "\n")
        
        sorted_rows = sort_blocks_to_4x4(blocks_info)
        puzzle_matrix = []
        for row in sorted_rows:
            row_pred = [block["pred"] for block in row]
            puzzle_matrix.append(row_pred)
        
        puzzle_matrix = np.array(puzzle_matrix)
        
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
        
        return puzzle_matrix, f"识别成功！\n识别结果矩阵：\n{puzzle_matrix}\n\n【{DEVICE_TARGET}】批量总耗时：{batch_total_time:.2f}ms | 平均耗时：{batch_avg_time:.2f}ms/个"
    
    except Exception as e:
        return None, f"识别出错：{str(e)}"

def train_model():
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