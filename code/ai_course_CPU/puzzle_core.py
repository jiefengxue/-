import numpy as np
import heapq
import random
import os

#全局尺寸
#图片处理尺寸
INPUT_IMAGE_SIZE = (640, 640)    #上传图片强制尺寸
WARPED_SIZE = (480, 480)         #透视矫正后尺寸
CELL_SIZE_WARPED = WARPED_SIZE[0] // 4  #分割格子尺寸（120x120）
CELL_SIZE_MODEL = (50, 50)       #CNN输入尺寸

#模型/数据集配置
NUM_CLASSES = 16
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
WEIGHTS_DIR = "weights"
DATASET_DIR = "dataset/digit_data_fixed"
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

#UI配置
CELL_SIZE_UI = 70
CELL_FONT = ("微软雅黑", 18, "bold")
EMPTY_COLOR = "#f0f0f0"
NUMBER_COLOR = "#4CAF50"
STEP_DELAY = 0.4

#A*求解算法
class AStarNode:
    def __init__(self, state, g=0, h=0, parent=None, action=None):
        self.state = state
        self.g = g
        self.h = h
        self.f = g + h                                                      #代价函数
        self.parent = parent
        self.action = action
    
    def __lt__(self, other):
        return self.f < other.f

def manhattan_distance(state):
    goal_state = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,0]]) #目标状态
    target_pos = {val: (i, j) for i in range(4) for j, val in enumerate(goal_state[i])}
    total_dist = 0
    for i in range(4):
        for j in range(4):
            val = state[i][j]
            if val == 0:
                continue
            ti, tj = target_pos[val]
            total_dist += abs(i - ti) + abs(j - tj)                           #曼哈顿距离h
    return total_dist

def is_solvable(state):#是否可解
    flat_state = state.flatten()
    inversions = 0
    for i in range(15):
        if flat_state[i] == 0:
            continue
        for j in range(i + 1, 16):
            if flat_state[j] != 0 and flat_state[i] > flat_state[j]:
                inversions += 1
    blank_i = np.where(state == 0)[0][0]
    blank_row_1based = blank_i + 1
    return (inversions + blank_row_1based) % 2 == 0

def get_valid_neighbors(node):#给定当前节点，生成所有合法移动后的邻居节点
    neighbors = []
    state = node.state
    blank_i, blank_j = np.where(state == 0)
    blank_i, blank_j = blank_i[0], blank_j[0]
    
    valid_actions = [
        ("up", blank_i - 1, blank_j),
        ("down", blank_i + 1, blank_j),
        ("left", blank_i, blank_j - 1),
        ("right", blank_i, blank_j + 1)
    ]
    
    for action_name, new_i, new_j in valid_actions:
        if 0 <= new_i < 4 and 0 <= new_j < 4:
            new_state = state.copy()
            new_state[blank_i][blank_j], new_state[new_i][new_j] = new_state[new_i][new_j], new_state[blank_i][blank_j]
            new_node = AStarNode(
                state=new_state,
                g=node.g + 1,
                h=manhattan_distance(new_state),
                parent=node,
                action=action_name
            )
            neighbors.append(new_node)
    return neighbors

def solve_15puzzle(initial_state):#--------------------------------------------------------算法
    if initial_state.shape != (4, 4):
        return None, "输入状态必须是4x4矩阵"
    
    goal_state = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,0]])
    if np.array_equal(initial_state, goal_state):
        return [], "已处于目标状态（步数：0）"
    
    if not is_solvable(initial_state):
        return None, "该状态无解"
    
    open_list = []
    closed_set = set()
    initial_node = AStarNode(initial_state, h=manhattan_distance(initial_state))
    heapq.heappush(open_list, initial_node)
    closed_set.add(tuple(initial_state.flatten()))
    
    while open_list:
        current_node = heapq.heappop(open_list)
        for neighbor in get_valid_neighbors(current_node):
            neighbor_flat = tuple(neighbor.state.flatten())
            if neighbor_flat not in closed_set:
                if np.array_equal(neighbor.state, goal_state):
                    path = []
                    current = neighbor
                    while current.parent:
                        path.append(current.action)
                        current = current.parent
                    return path[::-1], f"求解成功（步数：{len(path)}）"
                closed_set.add(neighbor_flat)
                heapq.heappush(open_list, neighbor)
    
    return None, "搜索失败（未找到解）"

#随机生成15数码状态 目标状态开始，随机移动空白格指定步数，最后检查是否可解，不可解则递归重新生成-----一定有解
def generate_random_15puzzle(shuffle_steps=50):
    goal_state = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,0]])
    current_state = goal_state.copy()
    
    def get_valid_actions(state):
        blank_i, blank_j = np.where(state == 0)
        blank_i, blank_j = blank_i[0], blank_j[0]
        actions = []
        if blank_i > 0:
            actions.append(("up", blank_i - 1, blank_j))
        if blank_i < 3:
            actions.append(("down", blank_i + 1, blank_j))
        if blank_j > 0:
            actions.append(("left", blank_i, blank_j - 1))
        if blank_j < 3:
            actions.append(("right", blank_i, blank_j + 1))
        return actions
    
    for _ in range(shuffle_steps):
        valid_actions = get_valid_actions(current_state)
        if not valid_actions:
            break
        action_name, new_i, new_j = random.choice(valid_actions)
        blank_i, blank_j = np.where(current_state == 0)
        blank_i, blank_j = blank_i[0], blank_j[0]
        current_state[blank_i][blank_j], current_state[new_i][new_j] = current_state[new_i][new_j], current_state[blank_i][blank_j]
    
    if not is_solvable(current_state):
        return generate_random_15puzzle(shuffle_steps=shuffle_steps)
    return current_state