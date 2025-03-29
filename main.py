import cv2
import numpy as np
import pyautogui
import threading
import tkinter as tk
from tkinter import ttk, filedialog
import keyboard
import time
import os
import json

CONFIG_FILE = "config.json"
IMAGE_DIR = "images"

if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {"start": "]", "stop": "alt"}

def save_config(hotkeys):
    with open(CONFIG_FILE, "w") as f:
        json.dump(hotkeys, f)

def get_image_path(filename):
    return os.path.join(IMAGE_DIR, filename)

class ClickApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Auto Clicker")
        self.hotkeys = load_config()
        self.detector = ImageDetector(self.update_message)

        self.label = tk.Label(root, text="クリック対象画像を選択してください")
        self.label.pack(pady=10)

        self.template_var = tk.StringVar()
        self.template_box = ttk.Combobox(root, textvariable=self.template_var, state="readonly")
        self.template_box.pack(pady=5)
        self.template_box.bind("<<ComboboxSelected>>", self.select_template)
        self.update_image_list()

        tk.Button(root, text="画像を追加", command=self.add_image).pack()
        tk.Button(root, text="画像を削除", command=self.delete_image).pack()

        self.message_label = tk.Label(root, text="")
        self.message_label.pack(pady=5)

        self.hotkey_labels = {}
        self.hotkey_entries = {}
        self.setup_hotkey_ui("開始キー", "start")
        self.setup_hotkey_ui("停止キー", "stop")

        self.register_hotkeys()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_hotkey_ui(self, label_text, key):
        frame = tk.Frame(self.root)
        frame.pack()
        
        label = tk.Label(frame, text=f"{label_text}: {self.hotkeys[key]}")
        label.pack(side=tk.LEFT)
        self.hotkey_labels[key] = label
        
        entry = tk.Entry(frame)
        entry.pack(side=tk.LEFT)
        self.hotkey_entries[key] = entry
        
        tk.Button(frame, text=" 設定 ", command=lambda: self.set_hotkey(key)).pack(side=tk.LEFT)
    
    def set_hotkey(self, key):
        value = self.hotkey_entries[key].get().strip()
        if len(value) != 1:
            self.update_message("エラー: キーは1文字のみ")
            return
        
        keyboard.remove_hotkey(self.hotkeys[key]) 
        self.hotkeys[key] = value
        save_config(self.hotkeys)
        self.hotkey_labels[key].config(text=f"{key.capitalize()}キー: {value}")
        self.register_hotkeys()
    
    def register_hotkeys(self):
        keyboard.add_hotkey(self.hotkeys["start"], lambda: self.detector.start())
        keyboard.add_hotkey(self.hotkeys["stop"], lambda: self.detector.stop())

    def update_image_list(self):
        images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".png")]
        self.template_box["values"] = images
    
    def add_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("PNG images", "*.png")])
        if file_path:
            os.rename(file_path, get_image_path(os.path.basename(file_path)))
            self.update_image_list()
    
    def delete_image(self):
        if self.template_var.get():
            os.remove(get_image_path(self.template_var.get()))
            self.update_image_list()
    
    def select_template(self, event=None):
        self.detector.set_template(get_image_path(self.template_var.get()))
        self.update_message("開始キーで実行、停止キーで終了")
    
    def update_message(self, message):
        self.message_label.config(text=message)
    
    def on_closing(self):
        self.detector.stop()
        self.root.destroy()
        
class ImageDetector:
    def __init__(self, update_message_callback):
        self.running = False
        self.thread = None
        self.template_path = None
        self.template = None
        self.update_message = update_message_callback
    
    def set_template(self, template_path):
        self.template_path = template_path
        self.template = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
    
    def start(self):
        if not self.template_path:
            return
        self.running = True
        self.update_message("実行中")
        self.thread = threading.Thread(target=self.detect_image, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        self.update_message("停止中")
        if self.thread and self.thread.is_alive():
            self.thread.join()
        self.thread = None

    def detect_image(self):
        if self.template is None:
            return

        sift = cv2.SIFT_create()
        kp_template, des_template = sift.detectAndCompute(self.template, None)
        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

        while self.running:
            time.sleep(0.1)
            screenshot = pyautogui.screenshot()
            img_gray = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGR2GRAY)
            kp_screen, des_screen = sift.detectAndCompute(img_gray, None)

            if des_screen is None or len(des_screen) < 10:
                time.sleep(0.1)
                continue

            matches = flann.knnMatch(des_template, des_screen, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.6 * n.distance]

            if len(good_matches) >= 10:
                x, y = np.mean([kp_screen[m.trainIdx].pt for m in good_matches], axis=0)
                pyautogui.moveTo(int(x), int(y), duration=0.1)
                pyautogui.click()
            

if __name__ == "__main__":
    root = tk.Tk()
    app = ClickApp(root)
    root.mainloop()
