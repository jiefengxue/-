import tkinter as tk
from tkinter import ttk, messagebox, filedialog, Frame, Label
import numpy as np
import os
import time

from puzzle_core import (
    solve_15puzzle, generate_random_15puzzle, is_solvable,
    CELL_SIZE_UI, CELL_FONT, EMPTY_COLOR, NUMBER_COLOR, STEP_DELAY
)
from puzzle_vision import parse_puzzle_from_image, train_model

#可视化面板
class StepVisualizer(Frame):
    def __init__(self, parent):
        super().__init__(parent, bd=3, relief=tk.GROOVE, bg="#f8f8f8")
        self.parent = parent
        self.current_state = None
        self._create_title()
        self._create_grid()
        
        
    def _create_title(self):#标题
        self.title_label = Label(
            self, text="15数码求解可视化", font=("微软雅黑", 16, "bold"), bg="#f8f8f8"
        )
        self.title_label.pack(pady=10)
        
    def _create_grid(self):#格子
        grid_frame = Frame(self, bg="#f8f8f8")
        grid_frame.pack(pady=5)
        self.cell_labels = []
        for i in range(4):
            row_labels = []
            for j in range(4):
                cell = Label(
                    grid_frame, width=6, height=3, font=CELL_FONT,
                    bg=EMPTY_COLOR, relief=tk.RAISED, bd=2, fg="#000000"
                )
                cell.grid(row=i, column=j, padx=3, pady=3)
                row_labels.append(cell)
            self.cell_labels.append(row_labels)
    
    def update_state(self, state, step_num=0, action="初始状态"):#更新状态
        self.current_state = state.copy()
        self.title_label.config(text=f"步骤 {step_num}：{action}")
        for i in range(4):
            for j in range(4):
                val = state[i][j]
                cell = self.cell_labels[i][j]
                if val == 0:
                    cell.config(text="", bg=EMPTY_COLOR)
                else:
                    cell.config(text=str(val), bg=NUMBER_COLOR, fg="white")
        self.update_idletasks()
       
#主界面
class PuzzleSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("15数码问题")
        self.root.geometry("800x650")
        self.root.configure(bg="#f0f0f0")
        self.puzzle_state = None
        self.visualizer = None
        self.setup_ui()

    def setup_ui(self):
        #功能按钮区
        top_frame = Frame(self.root, bg="#f0f0f0")
        top_frame.pack(pady=20)
        
        #模型训练按钮
        self.train_btn = ttk.Button(
            top_frame, text="生成数据集+训练模型", command=self.train_model_click, width=18
        )
        self.train_btn.grid(row=0, column=0, padx=8)
        
        #核心功能按钮
        self.random_btn = ttk.Button(
            top_frame, text="生成随机状态", command=self.generate_random, width=14
        )
        self.random_btn.grid(row=0, column=1, padx=8)
        
        self.manual_btn = ttk.Button(
            top_frame, text="手动输入状态", command=self.manual_input, width=14
        )
        self.manual_btn.grid(row=0, column=2, padx=8)
        
        self.upload_btn = ttk.Button(
            top_frame, text="上传图片识别", command=self.upload_image, width=14
        )
        self.upload_btn.grid(row=0, column=3, padx=8)
        
        self.solve_btn = ttk.Button(
            top_frame, text="启动可视化求解", command=self.start_solve, width=14, state="disabled"
        )
        self.solve_btn.grid(row=0, column=4, padx=8)

        # 中间可视化面板
        self.visualizer = StepVisualizer(self.root)
        self.visualizer.pack(pady=20, fill=tk.BOTH, expand=True, padx=50)

    def train_model_click(self):#模型训练显示
        self.root.config(cursor="watch")
        self.root.update()
        try:
            train_model()
            messagebox.showinfo("成功", "数据集生成+模型训练完成！")
        except Exception as e:
            messagebox.showerror("失败", f"训练出错：{str(e)}")
        self.root.config(cursor="")

    def update_state_display(self, puzzle):#更新状态显示
        self.puzzle_state = puzzle
        self.visualizer.update_state(puzzle, step_num=0, action="初始状态")
        self.solve_btn.config(state="normal")

    def generate_random(self):#生成随机状态
        step_window = tk.Toplevel(self.root)
        step_window.title("设置打乱步数")
        step_window.geometry("300x180")
        step_window.grab_set()
        step_window.configure(bg="#f0f0f0")

        Label(step_window, text="打乱步数（10-200）：", font=("微软雅黑", 12), bg="#f0f0f0").pack(pady=15)
        step_var = tk.StringVar(value="50")
        step_entry = ttk.Entry(step_window, textvariable=step_var, font=("微软雅黑", 12), width=10)
        step_entry.pack(pady=5)
        step_entry.focus()

        def confirm_steps():#输入框与确认步数
            try:
                steps = int(step_var.get().strip())
                if 10 <= steps <= 200:
                    step_window.destroy()
                    self.root.config(cursor="watch")
                    self.root.update()
                    puzzle = generate_random_15puzzle(shuffle_steps=steps)
                    self.root.config(cursor="")
                    self.update_state_display(puzzle)
                else:
                    messagebox.showerror("错误", "步数必须在10-200之间！")
            except ValueError:
                messagebox.showerror("错误", "请输入有效数字！")

        ttk.Button(step_window, text="确认", command=confirm_steps).pack(pady=15)
        step_window.bind("<Return>", lambda e: confirm_steps())

    def manual_input(self):#手动输入状态
        input_window = tk.Toplevel(self.root)
        input_window.title("手动输入状态")
        input_window.geometry("400x250")
        input_window.grab_set()
        input_window.configure(bg="#f0f0f0")

        Label(input_window, text="输入4x4数字（空格分隔，0=空白）：", font=("微软雅黑", 12), bg="#f0f0f0").pack(pady=10)
        input_var = tk.StringVar(value="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0")
        input_entry = ttk.Entry(input_window, textvariable=input_var, font=("微软雅黑", 10), width=40)
        input_entry.pack(pady=5)

        def confirm_input():#输入框逻辑
            try:
                nums = list(map(int, input_var.get().split()))
                if len(nums) != 16 or set(nums) != set(range(16)):
                    messagebox.showerror("错误", "必须输入0-15不重复的16个数字！")
                    return
                puzzle = np.array(nums).reshape(4, 4)
                if not is_solvable(puzzle):
                    messagebox.showerror("错误", "该状态无解！")
                    return
                input_window.destroy()
                self.update_state_display(puzzle)
            except ValueError:
                messagebox.showerror("错误", "请输入数字+空格格式！")

        ttk.Button(input_window, text="确认", command=confirm_input).pack(pady=10)

    def upload_image(self):#上传图片识别
        file_path = filedialog.askopenfilename(
            title="选择15数码图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")]
        )
        if not file_path:
            return
        
        self.root.config(cursor="watch")
        self.root.update()
        
        try:
            puzzle, msg = parse_puzzle_from_image(file_path)
            self.root.config(cursor="")
            if puzzle is not None:
                if not is_solvable(puzzle):
                    messagebox.showerror("识别结果", f"{msg}\n该状态无解，无法进行求解！")
                    return
                #可解则正常展示
                self.update_state_display(puzzle)
                messagebox.showinfo("识别结果", msg)
            else:
                messagebox.showerror("识别失败", msg)
        except Exception as e:
            self.root.config(cursor="")
            messagebox.showerror("错误", f"图片识别出错：{str(e)}")

    def start_solve(self):#更新状态显示
        if self.puzzle_state is None:
            return
        self.solve_btn.config(state="disabled")
        self.root.config(cursor="watch")

        def solve_async():
            #先调用求解算法获取动作序列
            actions, msg = solve_15puzzle(self.puzzle_state)
            self.root.config(cursor="")
            
            if actions is None:
                messagebox.showerror("错误", msg)
                self.solve_btn.config(state="normal")
                return
            
            if not actions:
                messagebox.showinfo("提示", "当前已是目标状态！")
                self.solve_btn.config(state="normal")
                return
            
            #复制初始状态，用于逐步更新
            current_state = self.puzzle_state.copy()
            
            #定义递归函数，逐步执行动作
            def execute_step(step_idx=0):
                if step_idx >= len(actions):
                    #所有步骤执行完毕
                    messagebox.showinfo("完成", f"求解成功！总步数：{len(actions)}")
                    self.solve_btn.config(state="normal")
                    return
                
                #获取当前步骤的动作
                action = actions[step_idx]
                step_num = step_idx + 1
                
                #找到空白格位置
                blank_i, blank_j = np.where(current_state == 0)
                blank_i, blank_j = blank_i[0], blank_j[0]
                
                #执行动作，更新状态
                if action == 'up':
                    current_state[blank_i][blank_j], current_state[blank_i-1][blank_j] = current_state[blank_i-1][blank_j], current_state[blank_i][blank_j]
                elif action == 'down':
                    current_state[blank_i][blank_j], current_state[blank_i+1][blank_j] = current_state[blank_i+1][blank_j], current_state[blank_i][blank_j]
                elif action == 'left':
                    current_state[blank_i][blank_j], current_state[blank_i][blank_j-1] = current_state[blank_i][blank_j-1], current_state[blank_i][blank_j]
                elif action == 'right':
                    current_state[blank_i][blank_j], current_state[blank_i][blank_j+1] = current_state[blank_i][blank_j+1], current_state[blank_i][blank_j]
                
                
                self.visualizer.update_state(current_state, step_num, action)
                
            
                self.root.after(int(STEP_DELAY * 1000), execute_step, step_idx + 1)
            
           
            execute_step()


        self.root.after(0, solve_async)


if __name__ == "__main__":
    #屏蔽TK警告
    os.environ["TK_SILENCE_DEPRECATION"] = "1"
    
    try:
        import ctypes
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd != 0:
            ctypes.windll.user32.ShowWindow(hwnd, 0)
    except:
        pass
    
    root = tk.Tk()
    app = PuzzleSolverApp(root)
    root.mainloop()