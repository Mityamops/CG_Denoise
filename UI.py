import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from main import TV_reg
from illumination_gradient import remove_illumination_gradient_lowpass
import os
import threading

def is_grayscale(image):
    # Проверяем количество каналов в изображении
    if len(image.shape) == 2:
        return True
    elif len(image.shape) == 3:
        # Проверяем, есть ли три канала (RGB) и все ли каналы одинаковы
        return image.shape[2] == 1 or (image[:, :, 0] == image[:, :, 1]).all() and (image[:, :, 0] == image[:, :, 2]).all()
    else:
        return False

class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Main Window")
        self.geometry("1600x900")

        self.create_widgets()

    def create_widgets(self):
        self.file_label = tk.Label(self, text="No file selected")
        self.file_label.grid(pady=20, row=1, column=0)
        self.choose_file = tk.Button(self, text="выбрать файл", command=self.select_file)
        self.choose_file.grid(pady=10, row=0, column=0)
        self.choose_square = tk.Button(self, text="выбрать квадрат", command=self.select_square)
        self.choose_square.grid(pady=10, row=0, column=1)

        self.button_start = tk.Button(self, text="начать", command=self.open_window1)
        self.button_start.grid(pady=10, row=2, column=0)

    def select_square(self):
        self.file_label.config(text='images/square.png')

    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="Select a file",
            filetypes=(("PNG files", "*.png"), ("JPG files", "*.jpg"), ("TIFF files", "*.tif"), ("All files", "*.*"))
        )
        if file_path:
            self.file_label.config(text=file_path)

    def open_window1(self):
        window1 = Window1(self, self.file_label['text'])
        window1.grab_set()

class Window1(tk.Toplevel):
    def __init__(self, parent, filename):
        super().__init__(parent)
        self.title("Window 1")
        self.geometry("1600x900")
        self.image = filename
        self.create_widgets(filename)

    def create_widgets(self, filename):
        self.button = tk.Button(self, text='показать фото', command=lambda: self.display_image(filename))
        self.button.pack(pady=20)

        self.eq_hist = tk.BooleanVar()
        self.il_grad = tk.BooleanVar()
        self.resize = tk.BooleanVar()

        self.hist_but = tk.Checkbutton(self, text="Выравнивание гистограммы", variable=self.eq_hist)
        self.hist_but.pack(pady=10)

        self.il_but = tk.Checkbutton(self, text="засветка градиента", variable=self.il_grad)
        self.il_but.pack(pady=10)

        self.resize_but = tk.Checkbutton(self, text="Resize", variable=self.resize, command=self.toggle_resize)
        self.resize_but.pack(pady=10)

        self.methods = tk.StringVar()
        self.methods.set("FR")  # Устанавливаем значение по умолчанию
        options = ["FR", "PR", "DY", "BKS", "BKG", "BKY"]
        self.dropdown = tk.OptionMenu(self, self.methods, *options)
        self.dropdown.pack(pady=10)

        tk.Label(self, text='введите mu:').pack(pady=10)
        self.entry_mu = tk.Entry(self)
        self.entry_mu.pack(pady=10)

        self.start_algorithm = tk.Button(self, text='убрать шум', command=self.open_window_algorithm)
        self.start_algorithm.pack(pady=10)

        self.image_label = tk.Label(self)
        self.image_label.pack(pady=20)

        # Инициализация полей для ресайза
        self.resize_frame = tk.Frame(self)
        self.resize_width_label = tk.Label(self.resize_frame, text="Width:")
        self.resize_width_entry = tk.Entry(self.resize_frame)
        self.resize_height_label = tk.Label(self.resize_frame, text="Height:")
        self.resize_height_entry = tk.Entry(self.resize_frame)
        self.resize_percent_label = tk.Label(self.resize_frame, text="Percent:")
        self.resize_percent_entry = tk.Entry(self.resize_frame)

    def toggle_resize(self):
        if self.resize.get():
            self.resize_frame.pack(pady=10)
            self.resize_width_label.pack(side=tk.LEFT)
            self.resize_width_entry.pack(side=tk.LEFT)
            self.resize_height_label.pack(side=tk.LEFT)
            self.resize_height_entry.pack(side=tk.LEFT)
            self.resize_percent_label.pack(side=tk.LEFT)
            self.resize_percent_entry.pack(side=tk.LEFT)
        else:
            self.resize_frame.pack_forget()

    def open_window_algorithm(self):
        method = self.methods.get()
        mu = float(self.entry_mu.get())
        hist = self.eq_hist.get()
        ilum_grad = self.il_grad.get()
        resize = self.resize.get()
        resize_width = self.resize_width_entry.get()
        resize_height = self.resize_height_entry.get()
        resize_percent = self.resize_percent_entry.get()

        print(method, mu, hist, ilum_grad, resize, resize_width, resize_height, resize_percent)

        # Загрузка изображения
        image = np.asarray(Image.open(self.image))

        # Ресайз изображения, если включен флаг
        if resize:
            if resize_width and resize_height:
                width = int(resize_width)
                height = int(resize_height)
                image = cv2.resize(image, (width, height))
            elif resize_percent:
                percent = float(resize_percent) / 100.0
                width = int(image.shape[1] * percent)
                height = int(image.shape[0] * percent)
                image = cv2.resize(image, (width, height))

        # Создаем окно ожидания
        waiting_window = WaitingWindow(self)
        waiting_window.grab_set()

        # Запускаем функцию TV_reg в отдельном потоке
        thread = threading.Thread(target=self.run_tv_reg, args=(image, method, mu, hist, ilum_grad, waiting_window))
        thread.start()

    def run_tv_reg(self, image, method, mu, hist, ilum_grad, waiting_window):
        if not is_grayscale(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if hist:
            image = cv2.equalizeHist(image)
        if ilum_grad:
            image = remove_illumination_gradient_lowpass(image)

        denoised = TV_reg(image, method, mu)

        # Закрываем окно ожидания
        waiting_window.destroy()

        # Открываем окно с результатом
        window2 = Window_algorithm(self, denoised, method, mu, hist, ilum_grad)
        # window2.grab_set()

    def display_image(self, file_path):
        image_window = ImageWindow(self, file_path)
        # Убираем вызов grab_set() для окна с изображением

class ImageWindow(tk.Toplevel):
    def __init__(self, parent, image_or_path):
        super().__init__(parent)
        self.title("Image Viewer")

        self.original_image = None
        self.current_image = None

        self.create_widgets(image_or_path)

    def create_widgets(self, image_or_path):
        if isinstance(image_or_path, str):
            # Если передан путь к файлу
            self.original_image = Image.open(image_or_path)
        else:
            # Если передано изображение
            self.original_image = Image.fromarray(image_or_path)

        self.current_image = self.original_image.copy()

        # Получаем размеры изображения
        width, height = self.current_image.size

        # Устанавливаем размер окна в зависимости от размеров изображения
        self.geometry(f"{width}x{height}")

        # Создаем фрейм для кнопок
        self.button_frame = tk.Frame(self)
        self.button_frame.pack(side=tk.TOP, fill=tk.X)

        self.equalize_button = tk.Button(self.button_frame, text="Выравнивание гистограммы", command=self.equalize_histogram)
        self.equalize_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.revert_button = tk.Button(self.button_frame, text="Вернуть исходное изображение", command=self.revert_to_original)
        self.revert_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.close_button = tk.Button(self.button_frame, text="Close", command=self.destroy)
        self.close_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.photo = ImageTk.PhotoImage(self.current_image)

        self.image_label = tk.Label(self, image=self.photo)
        self.image_label.image = self.photo  # Keep a reference to avoid garbage collection
        self.image_label.pack(pady=20)

    def equalize_histogram(self):
        image_array = np.array(self.current_image)
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        if image_array.dtype != np.uint8:
            image_array = (image_array ).astype(np.uint8)
        equalized_image = cv2.equalizeHist(image_array)
        self.current_image = Image.fromarray(equalized_image)
        self.update_image()

    def revert_to_original(self):
        self.current_image = self.original_image.copy()
        self.update_image()

    def update_image(self):
        self.photo = ImageTk.PhotoImage(self.current_image)
        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo  # Keep a reference to avoid garbage collection

class WaitingWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Processing")
        self.geometry("300x100")

        self.label = tk.Label(self, text="Подождите, идет обработка...")
        self.label.pack(pady=20)

class Window_algorithm(tk.Toplevel):
    def __init__(self, parent, image, method, mu, hist, ilum):
        super().__init__(parent)
        self.title("Window 2")
        self.geometry("200x150")

        self.start_algorithm(image, method, mu, hist, ilum)

    def start_algorithm(self, image, method, mu, hist, ilum):
        self.label = tk.Label(self, text=f"Method: {method}\nMu: {mu}")
        self.label.pack(pady=10)

        self.save_button = tk.Button(self, text="Save Image", command=lambda: self.save_image(image))
        self.save_button.pack(pady=10)

        self.close_button = tk.Button(self, text="Close", command=self.destroy)
        self.close_button.pack(pady=10)

        # Отображение обработанного изображения
        self.display_image(image)

    def display_image(self, image):
        image_window = ImageWindow(self, image)
        # Убираем вызов grab_set() для окна с изображением

    def save_image(self, image):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=(("PNG files", "*.png"), ("JPG files", "*.jpg"), ("TIFF files", "*.tiff"), ("All files", "*.*"))
        )
        if file_path:
            print(f"Saving image to {file_path}")
            # Преобразуем изображение в формат, поддерживаемый PIL
            if image.dtype == np.float64:
                image = (image ).astype(np.uint8)
            pil_image = Image.fromarray(image)
            pil_image.save(file_path)
            print(f"Image saved to {file_path}")

if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()
