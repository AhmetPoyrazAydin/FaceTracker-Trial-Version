import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
import time
import datetime
import sys
import os


class CameraTrackerApp:
    def __init__(self, master):
        """
        Uygulamanın temel ayarları:
        - Kamera açılır, yüz tespiti için Haar Cascade XML dosyası yüklenir.
        - Kontrol paneli ve video penceresi oluşturulur.
        - 14 geliştirme eklenmiştir.
        - Uygulamanın rengi siyah olarak ayarlanmıştır.
        - icon.ico dosyası kullanılarak uygulama ikonları eklenmiştir.
        - Cascade XML dosyasının exe ortamında doğru şekilde yüklenmesi için sys._MEIPASS kullanılır.
        """
        self.master = master
        self.master.title("Kontrol Paneli - Pencere Modu")
        self.master.configure(bg="black")
        # Ana pencereye icon ekleme (icon.ico dosyasının mevcut olduğundan emin olun)
        try:
            self.master.iconbitmap("icon.ico")
        except Exception as e:
            print("Icon ayarlanamadı: ", e)

        self.fullscreen = False  # Başlangıçta tam ekran modu kapalı
        self.last_update_time = time.time()
        self.fps = 0
        self.base_smoothing = 0.1
        self.current_roi = None  # Başlangıçta belirlenmedi

        # --- Style ayarları ---
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", background="black", foreground="white")
        style.configure("TLabel", background="black", foreground="white")

        # --- Kontrol Paneli Öğeleri ---
        self.toggle_button = ttk.Button(master, text="Tam Ekran", command=self.toggle_fullscreen)
        self.toggle_button.pack(pady=5)

        self.exit_button = ttk.Button(master, text="Çıkış", command=self.on_close)
        self.exit_button.pack(pady=5)

        self.snapshot_button = ttk.Button(master, text="Snapshot Al", command=self.take_snapshot)
        self.snapshot_button.pack(pady=5)

        self.interval_slider = tk.Scale(master, from_=5, to=50, orient=tk.HORIZONTAL,
                                        label="Güncelleme Aralığı (ms)", bg="black", fg="white",
                                        highlightbackground="black")
        self.interval_slider.set(10)
        self.interval_slider.pack(pady=5)

        self.log_text = tk.Text(master, height=5, width=50, bg="black", fg="white")
        self.log_text.pack(pady=5)

        # Klavye kısayolu: 'f' tuşu tam ekranı toggle eder.
        self.master.bind("<f>", lambda event: self.toggle_fullscreen())

        # --- Video Penceresi (Kamera Görüntüsü) ---
        self.video_window = tk.Toplevel(master)
        self.video_window.title("Kamera - Pencere Modu")
        self.video_window.configure(bg="black")
        try:
            self.video_window.iconbitmap("icon.ico")
        except Exception as e:
            print("Video penceresi için icon ayarlanamadı: ", e)
        self.video_label = tk.Label(self.video_window, bg="black")
        self.video_label.pack()

        # Data panel: Tam ekran modunda, kameranın altında veriler gösterilecek.
        self.data_panel = tk.Frame(self.video_window, bg="black")
        self.fps_label = tk.Label(self.data_panel, text="FPS: N/A", bg="black", fg="white")
        self.face_status_label = tk.Label(self.data_panel, text="Yüz: N/A", bg="black", fg="white")
        self.roi_label = tk.Label(self.data_panel, text="ROI: N/A", bg="black", fg="white")
        self.face_coord_label = tk.Label(self.data_panel, text="Yüz Koordinatları: N/A", bg="black", fg="white")
        self.smoothing_label = tk.Label(self.data_panel, text="Smoothing: N/A", bg="black", fg="white")
        self.timestamp_label = tk.Label(self.data_panel, text="Zaman: N/A", bg="black", fg="white")
        # Responsive layout: yan yana yerleştiriliyor
        self.fps_label.pack(side=tk.LEFT, padx=5)
        self.face_status_label.pack(side=tk.LEFT, padx=5)
        self.roi_label.pack(side=tk.LEFT, padx=5)
        self.face_coord_label.pack(side=tk.LEFT, padx=5)
        self.smoothing_label.pack(side=tk.LEFT, padx=5)
        self.timestamp_label.pack(side=tk.LEFT, padx=5)
        # Data panel başlangıçta gizlidir (sadece tam ekran modunda gösterilecek)

        # --- Kamera Başlatma ---
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.log_text.insert(tk.END, "Kamera açılamadı!\n")
            self.master.destroy()
            return

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # İlk ROI: tüm çerçeve
        self.current_roi = [0, 0, self.frame_width, self.frame_height]

        # Cascade XML dosyasını doğru şekilde yükleme
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(".")
        cascade_path = os.path.join(base_path, "haarcascade_frontalface_default.xml")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise IOError("Haar cascade dosyası yüklenemedi. Dosya yolu: " + cascade_path)

        # Hedef boyut: kare şeklinde görüntü (örnek: 400x400)
        self.display_size = 400

        self.current_display_frame = None  # Snapshot için kullanılacak

        # Başlangıçta update döngüsünü başlatıyoruz.
        self.update_frame()

    def update_frame(self):
        try:
            ret, frame = self.cap.read()
            if not ret:
                self.log_text.insert(tk.END, "Kare okunamadı!\n")
                self.master.after(self.interval_slider.get(), self.update_frame)
                return

            # Ters görüntüyü düzeltmek için flip yapıyoruz
            frame = cv2.flip(frame, 1)

            # FPS hesaplama
            current_time = time.time()
            dt = current_time - self.last_update_time
            self.last_update_time = current_time
            self.fps = 1.0 / dt if dt > 0 else 0

            # Yüz tespiti
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            if len(faces) > 0:
                # En büyük yüzü seçiyoruz
                face = max(faces, key=lambda rect: rect[2] * rect[3])
                (x, y, w, h) = face
                center_x = x + w / 2
                center_y = y + h / 2
                box_size = int(2 * max(w, h))
                target_x = int(center_x - box_size / 2)
                target_y = int(center_y - box_size / 2)
                target_roi = [target_x, target_y, box_size, box_size]
                face_coords = (x, y, w, h)
            else:
                target_roi = self.current_roi
                face_coords = None

            # Dinamik smoothing: mevcut ROI ile hedef arasındaki farkın büyüklüğüne göre alpha belirleniyor.
            current = np.array(self.current_roi, dtype=np.float32)
            target = np.array(target_roi, dtype=np.float32)
            diff = target - current
            magnitude = np.linalg.norm(diff)
            if magnitude > 50:
                alpha = 0.3
            elif magnitude > 20:
                alpha = 0.2
            else:
                alpha = self.base_smoothing

            self.current_roi = [
                int((1 - alpha) * self.current_roi[0] + alpha * target_roi[0]),
                int((1 - alpha) * self.current_roi[1] + alpha * target_roi[1]),
                int((1 - alpha) * self.current_roi[2] + alpha * target_roi[2]),
                int((1 - alpha) * self.current_roi[3] + alpha * target_roi[3])
            ]

            # ROI koordinatlarını sınırlandırma
            x_roi, y_roi, w_roi, h_roi = self.current_roi
            x_roi = max(x_roi, 0)
            y_roi = max(y_roi, 0)
            if x_roi + w_roi > self.frame_width:
                w_roi = self.frame_width - x_roi
            if y_roi + h_roi > self.frame_height:
                h_roi = self.frame_height - y_roi

            cropped_frame = frame[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi]
            display_frame = cv2.resize(cropped_frame, (self.display_size, self.display_size))
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            self.current_display_frame = display_frame.copy()

            img = Image.fromarray(display_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            # Data panel güncellemesi (tam ekran modunda)
            if self.fullscreen:
                self.fps_label.config(text=f"FPS: {self.fps:.1f}")
                if len(faces) > 0:
                    self.face_status_label.config(text=f"Yüz: {len(faces)}")
                    self.face_coord_label.config(text=f"Yüz Koordinatları: {face_coords}")
                    self.smoothing_label.config(text=f"Smoothing: {alpha:.2f}")
                else:
                    self.face_status_label.config(text="Yüz: Bulunamadı")
                    self.face_coord_label.config(text="Yüz Koordinatları: N/A")
                    self.smoothing_label.config(text=f"Smoothing: {alpha:.2f}")
                self.roi_label.config(text=f"ROI: {self.current_roi}")
                self.timestamp_label.config(text=f"Zaman: {datetime.datetime.now().strftime('%H:%M:%S')}")

        except Exception as e:
            self.log_text.insert(tk.END, f"Hata: {str(e)}\n")

        # Güncelleme aralığı slider değerine göre ayarlanıyor
        self.master.after(self.interval_slider.get(), self.update_frame)

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self.video_window.attributes("-fullscreen", self.fullscreen)
        if self.fullscreen:
            self.toggle_button.configure(text="Kare Ekran")
            self.video_window.title("Kamera - Tam Ekran Modu")
            # Data paneli tam ekran modunda göster
            self.data_panel.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
            self.master.title("Kontrol Paneli - Tam Ekran Modu")
        else:
            self.toggle_button.configure(text="Tam Ekran")
            self.video_window.title("Kamera - Pencere Modu")
            self.data_panel.pack_forget()
            self.master.title("Kontrol Paneli - Pencere Modu")

    def take_snapshot(self):
        try:
            if self.current_display_frame is not None:
                filename = datetime.datetime.now().strftime("snapshot_%Y%m%d_%H%M%S.png")
                img = Image.fromarray(self.current_display_frame)
                img.save(filename)
                self.log_text.insert(tk.END, f"Snapshot kaydedildi: {filename}\n")
        except Exception as e:
            self.log_text.insert(tk.END, f"Snapshot hatası: {str(e)}\n")

    def on_close(self):
        self.cap.release()
        self.video_window.destroy()
        self.master.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CameraTrackerApp(root)
    root.mainloop()
