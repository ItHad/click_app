import os
import threading
import json
import time
import queue
import numpy as np
import tkinter as tk
from tkinter import filedialog
from enum import Enum
import cv2
import pyautogui
import keyboard
import shutil
import random
from sklearn.cluster import DBSCAN
from PIL import Image, ImageTk


CONFIG_FILE = "config.json"
IMAGE_DIR = "images"

# UI設定
PAD_LARGE = 10
PAD_MEDIUM = 5
PAD_SMALL = 2
HOTKEY_ENTRY_WIDTH = 10

# 処理インターバル設定 (秒)
SCAN_INTERVAL_SECONDS = 0.1
POST_CLICK_DELAY_SECONDS = 0.1
MESSAGE_QUEUE_TIMEOUT_SECONDS = 1.0

# 画像検出(SIFT/FLANN)パラメータ設定
KNN_MATCH_K = 2
LOWE_RATIO_THRESHOLD = 0.7
MIN_GOOD_MATCHES = 10
MIN_KEYPOINTS_ON_SCREEN = 10
MIN_DESCRIPTORS_FOR_MATCH = 2

# FLANNのパラメータ
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_PARAMS = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
FLANN_SEARCH_PARAMS = dict(checks=50)


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
        self.detector = None
        self.message_queue = queue.Queue()
        self.selected_image_files = []
        self.hotkey_labels = {}
        self.hotkey_entries = {}

        self.preview_window = None
        self.last_hovered_index = -1
        self.hovered_listbox = None

        self.setup_ui()
        self.update_available_images_listbox()
        self.register_hotkeys()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_message_thread()

    def setup_ui(self):
        list_frame = tk.Frame(self.root)
        list_frame.pack(pady=PAD_LARGE, padx=PAD_LARGE, fill=tk.X)

        available_frame = tk.Frame(list_frame)
        tk.Label(available_frame, text="利用可能な画像").pack()
        self.available_listbox = tk.Listbox(
            available_frame, selectmode=tk.EXTENDED, exportselection=False
        )
        self.available_listbox.bind("<Motion>", self._show_preview)
        self.available_listbox.bind("<Leave>", self._hide_preview)
        self.available_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        available_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        button_frame = tk.Frame(list_frame)
        tk.Button(button_frame, text="→", command=self.add_to_selection).pack(
            pady=PAD_MEDIUM
        )
        tk.Button(button_frame, text="←", command=self.remove_from_selection).pack(
            pady=PAD_MEDIUM
        )
        button_frame.pack(side=tk.LEFT, padx=PAD_LARGE)

        selected_frame = tk.Frame(list_frame)
        tk.Label(selected_frame, text="クリック対象 (優先度順)").pack()
        self.selected_listbox = tk.Listbox(
            selected_frame, selectmode=tk.EXTENDED, exportselection=False
        )
        self.selected_listbox.bind("<Motion>", self._show_preview)
        self.selected_listbox.bind("<Leave>", self._hide_preview)
        self.selected_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        selected_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        image_manage_frame = tk.Frame(self.root)
        image_manage_frame.pack(pady=PAD_MEDIUM)
        tk.Button(image_manage_frame, text="画像を追加", command=self.add_image).pack(
            side=tk.LEFT, padx=PAD_MEDIUM
        )
        tk.Button(
            image_manage_frame, text="画像を削除", command=self.delete_image
        ).pack(side=tk.LEFT, padx=PAD_MEDIUM)

        self.message_label = tk.Label(
            self.root, text="ホットキーで操作を開始/停止します"
        )
        self.message_label.pack(pady=PAD_MEDIUM)
        self.setup_hotkey_ui("startキー", "start")
        self.setup_hotkey_ui("stopキー", "stop")

    def setup_hotkey_ui(self, label_text, key):
        frame = tk.Frame(self.root)
        frame.pack(pady=PAD_SMALL)
        label = tk.Label(frame, text=f"{label_text}: {self.hotkeys[key]}")
        label.pack(side=tk.LEFT)
        self.hotkey_labels[key] = label
        entry = tk.Entry(frame, width=HOTKEY_ENTRY_WIDTH)
        entry.pack(side=tk.LEFT, padx=PAD_MEDIUM)
        self.hotkey_entries[key] = entry
        tk.Button(frame, text="設定", command=lambda: self.set_hotkey(key)).pack(
            side=tk.LEFT
        )

    def set_hotkey(self, key):
        new_value = self.hotkey_entries[key].get().strip()
        if not new_value:
            self.update_message("エラー: キーが入力されていません")
            return

        old_value = self.hotkeys[key]

        try:
            keyboard.remove_hotkey(old_value)
        except KeyError:
            pass

        self.hotkeys[key] = new_value
        save_config(self.hotkeys)
        self.hotkey_labels[key].config(text=f"{key.capitalize()}キー: {new_value}")

        callback = (
            self._handle_start_hotkey if key == "start" else self._handle_stop_hotkey
        )
        keyboard.add_hotkey(new_value, callback)

        self.update_message(f"{key.capitalize()}キーを '{new_value}' に設定しました")

    def _handle_start_hotkey(self):
        if not self.selected_image_files:
            self.update_message("エラー: クリック対象の画像が選択されていません")
            return

        if self.detector and self.detector.running:
            self.update_message("既に実行中です")
            return

        templates = []
        for filename in self.selected_image_files:
            path = get_image_path(filename)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                templates.append(image)
            else:
                self.update_message(f"警告: {filename} を読み込めませんでした")

        if not templates:
            self.update_message("エラー: 有効な画像を読み込めませんでした")
            return

        self.detector = ImageDetector(
            templates, self.message_queue, self.handle_detection
        )
        self.detector.start()

    def _handle_stop_hotkey(self):
        if self.detector and self.detector.running:
            self.detector.stop()
        else:
            self.update_message("実行されていません")

    def register_hotkeys(self):
        keyboard.add_hotkey(self.hotkeys["start"], self._handle_start_hotkey)
        keyboard.add_hotkey(self.hotkeys["stop"], self._handle_stop_hotkey)

    def update_available_images_listbox(self):
        self.available_listbox.delete(0, tk.END)
        images = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".png")])
        for image in images:
            self.available_listbox.insert(tk.END, image)

    def update_selected_images_listbox(self):
        self.selected_listbox.delete(0, tk.END)
        for image_name in self.selected_image_files:
            self.selected_listbox.insert(tk.END, image_name)

    def add_to_selection(self):
        selected_indices = self.available_listbox.curselection()
        for i in selected_indices:
            image_name = self.available_listbox.get(i)
            if image_name not in self.selected_image_files:
                self.selected_image_files.append(image_name)
        self.update_selected_images_listbox()

    def remove_from_selection(self):
        selected_indices = self.selected_listbox.curselection()
        for i in sorted(selected_indices, reverse=True):
            del self.selected_image_files[i]
        self.update_selected_images_listbox()

    def _show_preview(self, event):
        listbox = event.widget
        index = listbox.nearest(event.y)

        if index == self.last_hovered_index and listbox == self.hovered_listbox:
            return

        self._hide_preview()

        if index < 0 or index >= listbox.size():
            return

        self.last_hovered_index = index
        self.hovered_listbox = listbox

        filename = listbox.get(index)
        image_path = get_image_path(filename)

        try:
            img = Image.open(image_path)
            img.thumbnail((200, 200))

            self.preview_window = tk.Toplevel(self.root)
            self.preview_window.overrideredirect(True)

            x = event.x_root + 20
            y = event.y_root - 20
            self.preview_window.geometry(f"+{x}+{y}")

            photo = ImageTk.PhotoImage(img)

            frame = tk.Frame(self.preview_window, relief="solid", borderwidth=1)
            frame.pack()

            label = tk.Label(frame, image=photo)
            label.image = photo
            label.pack()
        except Exception:
            self._hide_preview()

    def _hide_preview(self, event=None):
        if self.preview_window:
            self.preview_window.destroy()
            self.preview_window = None
        self.last_hovered_index = -1
        self.hovered_listbox = None

    def add_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("PNG images", "*.png")])
        if not file_path:
            return

        basename = os.path.basename(file_path)
        filename, ext = os.path.splitext(basename)
        new_path = get_image_path(basename)
        count = 1
        while os.path.exists(new_path):
            new_basename = f"{filename}_{count}{ext}"
            new_path = get_image_path(new_basename)
            count += 1

        shutil.copy(file_path, new_path)
        self.update_available_images_listbox()
        self.update_message(f"{os.path.basename(new_path)} を追加しました")

    def delete_image(self):
        selected_indices = self.available_listbox.curselection()
        if not selected_indices:
            self.update_message("削除する画像をリストから選択してください")
            return

        for i in sorted(selected_indices, reverse=True):
            image_name = self.available_listbox.get(i)
            try:
                os.remove(get_image_path(image_name))
                if image_name in self.selected_image_files:
                    self.selected_image_files.remove(image_name)
            except OSError as e:
                self.update_message(f"エラー: {e}")

        self.update_available_images_listbox()
        self.update_selected_images_listbox()
        self.update_message("選択した画像を削除しました")

    def handle_detection(self, x, y):
        pyautogui.moveTo(x, y, duration=0.1)
        pyautogui.click()
        time.sleep(POST_CLICK_DELAY_SECONDS)

    def update_message_thread(self):
        def update():
            while True:
                try:
                    message = self.message_queue.get(
                        timeout=MESSAGE_QUEUE_TIMEOUT_SECONDS
                    )
                    if self.root.winfo_exists():
                        self.update_message(message)
                except queue.Empty:
                    if not self.root.winfo_exists():
                        break
                except Exception:
                    break

        threading.Thread(target=update, daemon=True).start()

    def update_message(self, message):
        if self.root.winfo_exists():
            self.message_label.config(text=message)

    def on_closing(self):
        if self.detector:
            self.detector.stop()
        keyboard.unhook_all()
        self.root.destroy()


class ImageDetector:
    def __init__(self, templates, message_queue, on_detect_callback):
        self.running = False
        self.thread = None
        self.templates = templates
        self.message_queue = message_queue
        self.on_detect = on_detect_callback

    def start(self):
        if self.running:
            return
        self.running = True
        self.message_queue.put(DetectorState.RUNNING.value)
        self.thread = threading.Thread(target=self.detect_images, daemon=True)
        self.thread.start()

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.message_queue.put(DetectorState.STOPPED.value)
        if self.thread and self.thread.is_alive():
            self.thread.join()
        self.thread = None

    def detect_images(self):
        try:
            sift = cv2.SIFT_create()
            flann = cv2.FlannBasedMatcher(FLANN_INDEX_PARAMS, FLANN_SEARCH_PARAMS)

            prepared_templates = []
            for tpl in self.templates:
                kp, des = sift.detectAndCompute(tpl, None)
                if des is not None and len(kp) > 0:
                    h, w = tpl.shape[:2]
                    prepared_templates.append(
                        {"kp": kp, "des": des.astype(np.float32), "h": h, "w": w}
                    )
                else:
                    self.message_queue.put(
                        "警告: 特徴点を検出できない画像がありました。"
                    )

            if not prepared_templates:
                self.message_queue.put("エラー: 有効なテンプレートがありません。")
                self.running = False
                return

        except cv2.error:
            self.message_queue.put(
                "エラー: SIFTが利用できません (opencv-contrib-pythonが必要)"
            )
            self.running = False
            return

        while self.running:
            time.sleep(SCAN_INTERVAL_SECONDS)
            screenshot = pyautogui.screenshot()
            img_gray = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGR2GRAY)
            kp_screen, des_screen = sift.detectAndCompute(img_gray, None)

            if des_screen is None or len(kp_screen) < MIN_KEYPOINTS_ON_SCREEN:
                continue

            des_screen = des_screen.astype(np.float32)

            for tpl_data in prepared_templates:
                if (
                    len(tpl_data["des"]) < MIN_DESCRIPTORS_FOR_MATCH
                    or len(des_screen) < MIN_DESCRIPTORS_FOR_MATCH
                ):
                    continue

                matches = flann.knnMatch(tpl_data["des"], des_screen, k=KNN_MATCH_K)

                good_matches = []
                for m, n in (match for match in matches if len(match) == KNN_MATCH_K):
                    if m.distance < LOWE_RATIO_THRESHOLD * n.distance:
                        good_matches.append(m)

                if len(good_matches) >= MIN_GOOD_MATCHES:
                    match_points = np.float32(
                        [kp_screen[m.trainIdx].pt for m in good_matches]
                    )

                    tpl_h, tpl_w = tpl_data["h"], tpl_data["w"]
                    eps = np.sqrt(tpl_h**2 + tpl_w**2)

                    clusters = DBSCAN(eps=eps, min_samples=MIN_GOOD_MATCHES).fit(
                        match_points
                    )
                    labels = clusters.labels_

                    valid_clusters = [label for label in labels if label != -1]
                    if not valid_clusters:
                        continue

                    grouped_points = {label: [] for label in set(valid_clusters)}
                    for i, label in enumerate(labels):
                        if label != -1:
                            grouped_points[label].append(match_points[i])

                    random_cluster_label = random.choice(list(grouped_points.keys()))
                    chosen_cluster_points = grouped_points[random_cluster_label]

                    x, y = np.mean(chosen_cluster_points, axis=0)
                    self.on_detect(int(x), int(y))
                    break


class DetectorState(Enum):
    STOPPED = "停止中"
    RUNNING = "実行中"


if __name__ == "__main__":
    root = tk.Tk()
    app = ClickApp(root)
    root.mainloop()
