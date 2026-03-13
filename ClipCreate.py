import os
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import threading
import time

# ---------- Настройки ----------
CLIP_LENGTH_SEC = 300  # длина клипа в секундах (5 минут)
MIN_LENGTH_SEC = 360   # минимальная длина видео для нарезки
CHECK_INTERVAL = 60    # интервал проверки новых видео в секундах

# ---------- Функции для видео ----------
def get_video_duration(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(result.stdout.strip())
    except:
        return 0

def split_video(video_path, output_base, log_widget, log_file, stop_flag, progress_bar):
    duration = get_video_duration(video_path)
    if duration < MIN_LENGTH_SEC:
        msg = f"⏭ Пропуск {video_path.name} — короткое видео ({int(duration)} сек)"
        log(msg, log_widget, log_file)
        return

    output_folder = output_base / video_path.stem
    output_folder.mkdir(exist_ok=True)
    parts = int(duration // CLIP_LENGTH_SEC) + (1 if duration % CLIP_LENGTH_SEC > 0 else 0)
    log(f"✂ Нарезка {video_path.name} на {parts} частей ({int(duration)} сек)", log_widget, log_file)

    for i in range(parts):
        if stop_flag['stop']:
            log(f"⏹ Нарезка {video_path.name} прервана пользователем", log_widget, log_file)
            break
        start = i * CLIP_LENGTH_SEC
        output_file = output_folder / f"{video_path.stem}_part{i+1}.mp4"
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-ss", str(start),
            "-t", str(CLIP_LENGTH_SEC),
            "-c", "copy",
            str(output_file)
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        percent = int(((i+1)/parts)*100)
        log(f"📌 Клип {i+1}/{parts} готов ({percent}%)", log_widget, log_file)
        if progress_bar:
            progress_bar['value'] = percent

# ---------- Лог ----------
def log(message, widget=None, log_file=None):
    print(message)
    if widget:
        widget.configure(state='normal')
        widget.insert(tk.END, message + "\n")
        widget.configure(state='disabled')
        widget.see(tk.END)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")

# ---------- UI ----------
class VideoCutterApp:
    def __init__(self, master):
        self.master = master
        master.title("🎬 Video Auto Cutter")

        self.source_folder = None
        self.output_folder = None
        self.log_file = "video_cut_log.txt"
        self.auto_running = False
        self.processed_files = set()
        self.current_stop_flag = None
        self.progress_bar = None

        # Кнопки выбора папок
        tk.Button(master, text="Выбрать папку с видео", command=self.choose_source).pack(pady=5)
        tk.Button(master, text="Выбрать папку для клипов", command=self.choose_output).pack(pady=5)

        # Кнопки управления
        tk.Button(master, text="Запустить авто-обработку", command=self.start_auto).pack(pady=5)
        tk.Button(master, text="Остановить авто", command=self.stop_auto).pack(pady=5)
        tk.Button(master, text="Прервать текущую нарезку", command=self.stop_current_video).pack(pady=5)

        # Кнопка скачать лог
        tk.Button(master, text="Скачать лог", command=self.save_log).pack(pady=5)

        # Прогресс-бар
        self.progress_bar = ttk.Progressbar(master, orient='horizontal', length=400, mode='determinate')
        self.progress_bar.pack(pady=5)

        # Лог-бар
        self.log_widget = scrolledtext.ScrolledText(master, width=80, height=20, state='disabled')
        self.log_widget.pack(padx=10, pady=10)

    def choose_source(self):
        folder = filedialog.askdirectory(title="Выберите папку с исходными видео")
        if folder:
            self.source_folder = Path(folder)
            log(f"📁 Исходная папка: {folder}", self.log_widget, self.log_file)

    def choose_output(self):
        folder = filedialog.askdirectory(title="Выберите папку для сохранения клипов")
        if folder:
            self.output_folder = Path(folder)
            log(f"📂 Папка для клипов: {folder}", self.log_widget, self.log_file)

    def start_auto(self):
        if not self.source_folder or not self.output_folder:
            messagebox.showerror("Ошибка", "Выберите исходную и папку для клипов")
            return
        if self.auto_running:
            messagebox.showinfo("Info", "Авто-обработка уже запущена")
            return
        self.auto_running = True
        threading.Thread(target=self.auto_process, daemon=True).start()
        log("ℹ Авто-обработка запущена", self.log_widget, self.log_file)

    def stop_auto(self):
        self.auto_running = False
        log("⏹ Авто-обработка остановлена", self.log_widget, self.log_file)

    def stop_current_video(self):
        if self.current_stop_flag:
            self.current_stop_flag['stop'] = True
            log("⏹ Запрос на остановку текущего видео отправлен", self.log_widget, self.log_file)

    def auto_process(self):
        while self.auto_running:
            new_videos = []
            for video_file in self.source_folder.glob("*.*"):
                if video_file.suffix.lower() in [".avi", ".mp4", ".mkv", ".mov"]:
                    if video_file.name not in self.processed_files:
                        new_videos.append(video_file)
            if new_videos:
                log(f"ℹ Найдено новых видео: {len(new_videos)}", self.log_widget, self.log_file)
                for video_file in new_videos:
                    self.current_stop_flag = {'stop': False}
                    split_video(video_file, self.output_folder, self.log_widget, self.log_file, self.current_stop_flag, self.progress_bar)
                    self.processed_files.add(video_file.name)
                    self.progress_bar['value'] = 0
                    self.current_stop_flag = None
            else:
                log("ℹ Новых видео не найдено", self.log_widget, self.log_file)
            time.sleep(CHECK_INTERVAL)

    def save_log(self):
        save_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            title="Сохранить лог"
        )
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(self.log_widget.get("1.0", tk.END))
            messagebox.showinfo("Готово", f"Лог сохранен в {save_path}")

# ---------- Запуск ----------
if __name__ == "__main__":
    root = tk.Tk()
    app = VideoCutterApp(root)
    root.mainloop()
