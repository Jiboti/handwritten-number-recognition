import os
import customtkinter
from tkinter import filedialog
from datetime import datetime
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
from tkinter import Label, Button
import cv2
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import threading

REF_WIDTH = 1280
REF_HEIGHT = 720

app = customtkinter.CTk()
selected_file = None
progress_bar = None  # Global reference to progress bar widget
training_complete = False
training_thread = None
trained_model = None  # Global to store the trained model

def load_mnist():
    # Define transforms: normalize to [0,1] and ensure tensor format
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std for better convergence
    ])
    
    # Load training data
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Load test data (for potential future use, but we only need train for now)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    return train_loader, len(train_dataset), test_dataset  # Return loader for easy iteration, total samples

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def train_model_in_thread(app, progress_callback, complete_callback):
    global training_complete, trained_model
    train_loader, num_samples, _ = load_mnist()
    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 5
    num_batches = len(train_loader)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # Update progress every 100 batches
            if (batch_idx + 1) % 100 == 0 or batch_idx + 1 == num_batches:
                progress = ((epoch * num_batches + batch_idx + 1) / (num_epochs * num_batches))
                progress_callback(progress, epoch, num_epochs, batch_idx + 1, num_batches, loss.item())
        avg_loss = epoch_loss / num_batches
        progress = ((epoch + 1) * num_batches / (num_epochs * num_batches))
        progress_callback(progress, epoch + 1, num_epochs, num_batches, num_batches, avg_loss)
    torch.save(model.state_dict(), 'model.pt')
    trained_model = model
    complete_callback(model)

def update_progress_bar(progress, epoch, num_epochs, batch_idx, num_batches, loss):
    global progress_bar
    if progress_bar:
        progress_bar.set(progress)
    current_time = datetime.now().strftime("%H:%M:%S")
    message = f"{current_time}: Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{num_batches}, Loss: {loss:.4f}"
    file_msg = customtkinter.CTkLabel(
        master=technical_messages_container,
        text=message,
        font=("Arial", int(14 * sy)),
        text_color="yellow"
    )
    file_msg.pack(anchor="w", pady=2, padx=5)
    app.update_idletasks()  # Force GUI update

def on_training_complete(model):
    global progress_bar, training_complete, training_thread
    if progress_bar:
        progress_bar.destroy()
        progress_bar = None
    current_time = datetime.now().strftime("%H:%M:%S")
    message = f"{current_time}: Model training completed and saved."
    file_msg = customtkinter.CTkLabel(
        master=technical_messages_container,
        text=message,
        font=("Arial", int(14 * sy)),
        text_color="#90EE90"
    )
    file_msg.pack(anchor="w", pady=2, padx=5)
    training_complete = True
    training_thread = None
    app.update()

def start_training(app):
    global training_thread
    if not training_thread or not training_thread.is_alive():
        training_thread = threading.Thread(target=train_model_in_thread, args=(app, update_progress_bar, on_training_complete))
        training_thread.start()

def recognize_digits(thresh, digit_boxes, app):
    global progress_bar, training_complete, training_thread, trained_model
    model_path = 'model.pt'
    if not os.path.exists(model_path):
        current_time = datetime.now().strftime("%H:%M:%S")
        message = f"{current_time}: No pre-trained model found. Starting training..."
        file_msg = customtkinter.CTkLabel(
            master=technical_messages_container,
            text=message,
            font=("Arial", int(14 * sy)),
            text_color="yellow"
        )
        file_msg.pack(anchor="w", pady=2, padx=5)
        app.update()
        # Initialize progress bar before starting training
        progress_bar = customtkinter.CTkProgressBar(
            master=technical_messages_container,
            width=380 * sx,
            height=20 * sy,
            progress_color="#90EE90"
        )
        progress_bar.set(0)
        progress_bar.pack(anchor="w", pady=5, padx=5)
        training_complete = False
        start_training(app)
        while not training_complete:
            app.update()
            time.sleep(0.1)  # Small delay to prevent excessive CPU usage
        model = trained_model
    else:
        current_time = datetime.now().strftime("%H:%M:%S")
        message = f"{current_time}: Loading pre-trained model..."
        file_msg = customtkinter.CTkLabel(
            master=technical_messages_container,
            text=message,
            font=("Arial", int(14 * sy)),
            text_color="yellow"
        )
        file_msg.pack(anchor="w", pady=2, padx=5)
        app.update()
        model = Net()
        model.load_state_dict(torch.load(model_path))
        current_time = datetime.now().strftime("%H:%M:%S")
        message = f"{current_time}: Pre-trained model loaded successfully."
        file_msg = customtkinter.CTkLabel(
            master=technical_messages_container,
            text=message,
            font=("Arial", int(14 * sy)),
            text_color="#90EE90"
        )
        file_msg.pack(anchor="w", pady=2, padx=5)
        app.update()
    model.eval()
    digits = []
    for x_c, y_c, w_c, h_c in digit_boxes:
        digit_img = thresh[y_c:y_c + h_c, x_c:x_c + w_c]
        max_side = max(w_c, h_c)
        pad_h_top = (max_side - h_c) // 2
        pad_h_bottom = max_side - h_c - pad_h_top
        pad_w_left = (max_side - w_c) // 2
        pad_w_right = max_side - w_c - pad_w_left
        digit_img = cv2.copyMakeBorder(digit_img, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right, cv2.BORDER_CONSTANT, value=0)
        digit_img = cv2.resize(digit_img, (28, 28), interpolation=cv2.INTER_AREA)
        digit_img = digit_img.astype(np.float32) / 255.0
        digit_tensor = torch.from_numpy(digit_img).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output = model(digit_tensor)
            pred = output.argmax(dim=1, keepdim=True).item()
        digits.append(str(pred))
    result = ''.join(digits)

    current_time = datetime.now().strftime("%H:%M:%S")
    recent_entry = f"{current_time}: {result if result else 'No digits detected'}"
    recent_label = customtkinter.CTkLabel(
    master=recentText,
    text=recent_entry,
    font=("Arial", int(14 * sy)),
    text_color="white",
    anchor="w"
    )
    recent_label.pack(fill="x", pady=2, padx=5)

    for widget in outputFrame.winfo_children():
        widget.destroy()
    output_text = customtkinter.CTkLabel(
        master=outputFrame,
        text=result if result else "No digits detected",
        font=("Arial", int(40 * sy)),
        text_color="black"
    )
    output_text.pack(expand=True, fill="both")
    current_time = datetime.now().strftime("%H:%M:%S")
    message = f"{current_time}: Recognized number: {result}"
    file_msg = customtkinter.CTkLabel(
        master=technical_messages_container,
        text=message,
        font=("Arial", int(14 * sy)),
        text_color="#90EE90"
    )
    file_msg.pack(anchor="w", pady=2, padx=5)

def process_image(file_path, app, x=225, y=140, display_size=(600, 400),
                  min_width=10, min_height=10, adaptive_block=15, adaptive_C=8):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    output_path = os.path.join(processed_dir, "processed_output_TEST.png")

    current_time = datetime.now().strftime("%H:%M:%S")
    message = f"{current_time}: Uploading file in grayscale..."
    file_msg = customtkinter.CTkLabel(
        master=technical_messages_container,
        text=message,
        font=("Arial", int(14 * sy)),
        text_color="white"
    )
    file_msg.pack(anchor="w", pady=2, padx=5)

    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    
    current_time = datetime.now().strftime("%H:%M:%S")
    message = f"{current_time}: File uploaded in grayscale"
    file_msg = customtkinter.CTkLabel(
        master=technical_messages_container,
        text=message,
        font=("Arial", int(14 * sy)),
        text_color="white"
    )
    file_msg.pack(anchor="w", pady=2, padx=5)

    pixelized = cv2.resize(img, (600, 600), interpolation=cv2.INTER_NEAREST)

    current_time = datetime.now().strftime("%H:%M:%S")
    message = f"{current_time}: File resized"
    file_msg = customtkinter.CTkLabel(
        master=technical_messages_container,
        text=message,
        font=("Arial", int(14 * sy)),
        text_color="white"
    )
    file_msg.pack(anchor="w", pady=2, padx=5)

    coords = cv2.findNonZero(pixelized)
    if coords is None:
        print("⚠️ No content found in image. Using full image.")
        cropped = pixelized
    else:
        x0, y0, w, h = cv2.boundingRect(coords)
        cropped = pixelized[y0:y0+h, x0:x0+w]

    current_time = datetime.now().strftime("%H:%M:%S")
    message = f"{current_time}: Preprocessing, cleaning image..."
    file_msg = customtkinter.CTkLabel(
        master=technical_messages_container,
        text=message,
        font=("Arial", int(14 * sy)),
        text_color="#E9BF57"
    )
    file_msg.pack(anchor="w", pady=2, padx=5)

    blurred = cv2.GaussianBlur(cropped, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, adaptive_block, adaptive_C
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxed_image = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)

    current_time = datetime.now().strftime("%H:%M:%S")
    message = f"{current_time}: File processed"
    file_msg = customtkinter.CTkLabel(
        master=technical_messages_container,
        text=message,
        font=("Arial", int(14 * sy)),
        text_color="#90EE90"
    )
    file_msg.pack(anchor="w", pady=2, padx=5)

    digit_boxes = []
    h_img, w_img = thresh.shape
    for cnt in contours:
        x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
        if w_c < min_width or h_c < min_height:
            continue
        if w_c > w_img * 0.95 or h_c > h_img * 0.95:
            continue
        aspect_ratio = w_c / h_c
        if aspect_ratio < 0.2 or aspect_ratio > 1.2:
            continue
        digit_boxes.append((x_c, y_c, w_c, h_c))

    digit_boxes = sorted(digit_boxes, key=lambda b: b[0])

    for x_c, y_c, w_c, h_c in digit_boxes:
        cv2.rectangle(boxed_image, (x_c, y_c), (x_c+w_c, y_c+h_c), (0, 0, 255), 2)

    amount_boxes = len(digit_boxes)
    current_time = datetime.now().strftime("%H:%M:%S")
    message = f"{current_time}: {amount_boxes} number(s) detected!"
    file_msg = customtkinter.CTkLabel(
        master=technical_messages_container,
        text=message,
        font=("Arial", int(14 * sy)),
        text_color="#D74343"
    )
    file_msg.pack(anchor="w", pady=2, padx=5)

    success = cv2.imwrite(output_path, boxed_image)

    pil_img = Image.open(output_path)
    pil_img = pil_img.resize(display_size)
    photo = ImageTk.PhotoImage(pil_img)

    image_label = customtkinter.CTkLabel(app, image=photo, text="")
    image_label.place(x=x, y=y)
    image_label.image = photo

    recognize_digits(thresh, digit_boxes, app)

def openFinder():
    global selected_file
    selected_file = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
    if selected_file:
        filename = os.path.basename(selected_file)
        if "." in filename:
            name_part, ext = ".".join(filename.split(".")[:-1]), "." + filename.split(".")[-1]
        else:
            name_part, ext = filename, ""
        max_length = 20
        display_name = name_part[:max_length] + "... " + ext if len(name_part) > max_length else filename
        uploadFileLabel.configure(text=f"Selected file: {display_name}")

        current_time = datetime.now().strftime("%H:%M:%S")
        message = f"{current_time}: File '{display_name}' uploaded"
        file_msg = customtkinter.CTkLabel(
            master=technical_messages_container,
            text=message,
            font=("Arial", int(14 * sy)),
            text_color="white"
        )
        file_msg.pack(anchor="w", pady=2, padx=5)

        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            format_msg = customtkinter.CTkLabel(
                master=technical_messages_container,
                text="File format corresponds to required format ✅",
                font=("Arial", int(14 * sy)),
                text_color="#90EE90"
            )
            format_msg.pack(anchor="w", pady=2, padx=5)
        else:
            format_msg = customtkinter.CTkLabel(
                master=technical_messages_container,
                text="Incorrect file format",
                font=("Arial", int(14 * sy)),
                text_color="#D74343"
            )
            format_msg.pack(anchor="w", pady=2, padx=5)
        process_image(selected_file, app)

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Webcam Capture")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Could not open webcam.")
            self.root.destroy()
            return
        self.running = True
        self.label = Label(root)
        self.label.pack()
        self.capture_btn = Button(root, text="Take Picture", command=self.take_picture)
        self.capture_btn.pack(pady=10)
        self.close_btn = Button(root, text="Close Camera", command=self.close_camera)
        self.close_btn.pack(pady=5)
        self.update_frame()

    def update_frame(self):
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.label.imgtk = imgtk
                self.label.configure(image=imgtk)
            self.root.after(10, self.update_frame)

    def take_picture(self):
        ret, frame = self.cap.read()
        if ret:
            filename = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Saved: {filename}")
        process_image(filename, app)
        self.close_camera()

    def close_camera(self):
        self.running = False
        self.cap.release()
        self.root.destroy()

def start_camera():
    cam_window = tk.Toplevel(app)
    CameraApp(cam_window)

screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()
app.geometry(f"{screen_width}x{screen_height}")

sx = screen_width / REF_WIDTH
sy = screen_height / REF_HEIGHT

frame = customtkinter.CTkFrame(
    master=app,
    width=200*sx,
    height=screen_height,
    corner_radius=0,
    fg_color="gray25"
)
frame.place(x=0, y=0)

label = customtkinter.CTkLabel(
    master=app,
    text="Text Detector",
    width=180*sx,
    height=40*sy,
    font=("Arial", int(24*sy), "bold"),
    text_color=("black", "white"),
    bg_color="gray25"
)
label.place(x=10*sx, y=10*sy)

max_label_width = (830*sx) - (220*sx) - (20*sx)
test_font_size = int(28*sy)
from tkinter import font as tkFont
test_tk_font = tkFont.Font(family="Arial", size=test_font_size, weight="bold")
while test_tk_font.measure("Upload a file to get started (.png, .jpg, .jpeg)") > max_label_width and test_font_size > 10:
    test_font_size -= 1
    test_tk_font = tkFont.Font(family="Arial", size=test_font_size, weight="bold")

uploadFileLabel = customtkinter.CTkLabel(
    master=app,
    text="Upload a file to get started (.png, .jpg, .jpeg)",
    width=max_label_width,
    height=40*sy,
    font=("Arial", test_font_size, "bold"),
    text_color="white",
    anchor="w"
)
uploadFileLabel.place(x=220*sx, y=20*sy)

chooseFileButton = customtkinter.CTkButton(
    master=app,
    text="Choose a file",
    width=150*sx,
    height=40*sy,
    font=("Arial", int(16*sy)),
    fg_color="beige",
    text_color="black",
    command=lambda: print(openFinder())
)
chooseFileButton.place(x=260*sx, y=70*sy)

folderSymbol = customtkinter.CTkLabel(
    master=app,
    text="📁",
    width=40*sx,
    height=40*sy,
    font=("Arial", int(26*sy)),
    text_color=("black", "white")
)
folderSymbol.place(x=220*sx, y=65*sy)

recentText = customtkinter.CTkScrollableFrame(
    master=app,
    width=400*sx,
    height=200*sy,
    corner_radius=10,
    border_width=2
)
recentText.place(x=830*sx, y=20*sy)
recentText._scrollbar.grid_configure(padx=(0,5))

frameRecentText = customtkinter.CTkFrame(
    master=recentText,
    width=380*sx,
    height=30*sy,
    corner_radius=10,
    fg_color="gray25"
)
frameRecentText.pack(fill="x", pady=(0,5))

labelRecentText = customtkinter.CTkLabel(
    master=frameRecentText,
    text="Recently Detected Text",
    font=("Arial", int(18*sy), "bold")
)
labelRecentText.pack(pady=5)

technicalFrame = customtkinter.CTkScrollableFrame(
    master=app,
    width=400*sx,
    height=340*sy,
    corner_radius=10,
    border_width=2,
    fg_color="#614c4d"
)
technicalFrame.place(x=830*sx, y=270*sy)
technicalFrame._scrollbar.grid_configure(padx=(0,5))

frameTechnicalFrame = customtkinter.CTkFrame(
    master=technicalFrame,
    width=380*sx,
    height=30*sy,
    corner_radius=10,
    fg_color="#8a6d6e"
)
frameTechnicalFrame.pack(fill="x", pady=(0,5))

technical_messages_container = customtkinter.CTkFrame(
    master=technicalFrame,
    fg_color="transparent"
)
technical_messages_container.pack(fill="both", expand=True, anchor="nw")

labelTechnicalText = customtkinter.CTkLabel(
    master=frameTechnicalFrame,
    text="Technical actions",
    font=("Arial", int(18*sy), "bold")
)
labelTechnicalText.pack(pady=5)

outputLabel = customtkinter.CTkLabel(
    master=app,
    text="Output :",
    font=("Arial", int(28*sy), "bold")
)
outputLabel.place(x=220*sx, y=440*sy)

outputFrame = customtkinter.CTkFrame(
    master=app,
    width=590*sx,
    height=150*sy,
    corner_radius=10,
    fg_color="#AEAEAE"
)
outputFrame.place(x=220*sx, y=480*sy)

uploadImageViaWebcam = customtkinter.CTkButton(
    master=app,
    text="Upload an image via webcam",
    width=150*sx,
    height=40*sy,
    font=("Arial", int(16*sy)),
    fg_color="white",
    text_color="black",
    command=start_camera
)
uploadImageViaWebcam.place(x=420*sx, y=70*sy)

app.mainloop()