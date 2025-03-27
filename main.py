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


class CookieClickerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Auto Clicker")
        self.template_path = None
        self.running = False
        self.thread = None
        self.hotkeys = {"start": "]", "stop": "alt"}

        self.load_config()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.label = tk.Label(root, text="クリック対象画像を選択してください")
        self.label.pack(pady=10)

        self.template_var = tk.StringVar()
        self.template_box = ttk.Combobox(
            root, textvariable=self.template_var, state="readonly"
        )
        self.template_box.pack(pady=5)
        self.template_box.bind("<<ComboboxSelected>>", self.select_template)
        self.update_image_list()

        self.add_image_button = tk.Button(
            root, text="画像を追加", command=self.add_image
        )
        self.add_image_button.pack()
        self.delete_image_button = tk.Button(
            root, text="画像を削除", command=self.delete_image
        )
        self.delete_image_button.pack()

        self.message_label = tk.Label(root, text="")
        self.message_label.pack(pady=5)

        self.start_key_label = tk.Label(root, text=f"開始キー: {self.hotkeys['start']}")
        self.start_key_label.pack()
        self.start_key_entry = tk.Entry(root)
        self.start_key_entry.pack()
        self.start_key_button = tk.Button(
            root, text="開始キーを設定", command=self.set_start_key
        )
        self.start_key_button.pack()

        self.stop_key_label = tk.Label(root, text=f"停止キー: {self.hotkeys['stop']}")
        self.stop_key_label.pack()
        self.stop_key_entry = tk.Entry(root)
        self.stop_key_entry.pack()
        self.stop_key_button = tk.Button(
            root, text="停止キーを設定", command=self.set_stop_key
        )
        self.stop_key_button.pack()

        self.register_hotkeys()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                self.hotkeys = json.load(f)

    def save_config(self):
        with open(CONFIG_FILE, "w") as f:
            json.dump(self.hotkeys, f)

    def register_hotkeys(self):
        if hasattr(self, "start_hotkey") and self.start_hotkey:
            keyboard.remove_hotkey(self.start_hotkey)
        if hasattr(self, "stop_hotkey") and self.stop_hotkey:
            keyboard.remove_hotkey(self.stop_hotkey)

        self.start_hotkey = self.hotkeys["start"]
        self.stop_hotkey = self.hotkeys["stop"]

        keyboard.add_hotkey(self.start_hotkey, self.start_detection)
        keyboard.add_hotkey(self.stop_hotkey, self.stop_detection)

    def set_start_key(self):
        key = self.start_key_entry.get().strip()
        if len(key) != 1:
            self.update_message("エラー: 開始キーは1文字のみ")
            return
        self.hotkeys["start"] = key
        self.start_key_label.config(text=f"開始キー: {key}")
        self.save_config()
        self.register_hotkeys()

    def set_stop_key(self):
        key = self.stop_key_entry.get().strip()
        if len(key) != 1:
            self.update_message("エラー: 停止キーは1文字のみ")
            return
        self.hotkeys["stop"] = key
        self.stop_key_label.config(text=f"停止キー: {key}")
        self.save_config()
        self.register_hotkeys()

    def update_image_list(self):
        images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".png")]
        self.template_box["values"] = images

    def add_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("PNG images", "*.png")])
        if file_path:
            file_name = os.path.basename(file_path)
            dest_path = os.path.join(IMAGE_DIR, file_name)
            os.rename(file_path, dest_path)
            self.update_image_list()

    def delete_image(self):
        if self.template_var.get():
            os.remove(os.path.join(IMAGE_DIR, self.template_var.get()))
            self.update_image_list()

    def select_template(self, event=None):
        self.template_path = os.path.join(IMAGE_DIR, self.template_var.get())
        self.label.config(text=f"選択済み: {self.template_path}")
        self.message_label.config(text="開始キーで実行、停止キーで終了")

    def start_detection(self):
        if not self.template_path:
            return
        self.running = True
        self.update_message("稼働中")
        if not self.thread or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.detect_cookie, daemon=True)
            self.thread.start()

    def stop_detection(self):
        self.running = False
        self.update_message("停止中")
        if self.thread and self.thread.is_alive():
            self.thread.join()
        self.thread = None

    def detect_cookie(self):
        template = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            self.update_message("エラー: 画像を開けません")
            return

        sift = cv2.SIFT_create()
        kp_template, des_template = sift.detectAndCompute(template, None)
        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

        while self.running:
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
                self.root.after(0, self.update_message, "クリック処理中...")

            time.sleep(0.1)

    def update_message(self, message):
        self.message_label.config(text=message)

    def on_closing(self):
        self.stop_detection()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CookieClickerApp(root)
    root.mainloop()