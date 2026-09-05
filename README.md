<h1 align="center">🔢 Handwritten Number Recognition System</h1>
<h4 align="center">Reading multi-digit handwritten numbers from an image — CNN + computer vision pipeline with a desktop GUI · COS30018 Intelligent Systems, Swinburne University</h4>

<p align="center">
  <img src="images/gui.png" alt="HNRS graphical interface" width="600"/>
</p>

---

## 🎯 Objective

Most tutorials stop at recognizing a *single* handwritten digit. This project goes one step further: reading a **complete multi-digit number** from a photo or a scan — which means the system has to find *where* the digits are before it can identify them.

Built as a four-person group project for the **Intelligent Systems (COS30018)** unit.

##  System architecture

The system is modular, with each stage feeding the next:

```
Image acquisition  →  Preprocessing  →  Segmentation  →  CNN recognition  →  Number
   (file / webcam)      (OpenCV)         (contours)        (PyTorch)
```

### 1. Preprocessing (OpenCV)
The raw image is converted to **grayscale**, resized to a fixed 600×600, then cleaned with a **Gaussian blur** to remove specks of noise. The key step is **adaptive Gaussian thresholding**: instead of one global cutoff, the threshold is computed per region, so the digits stay separated from the background even when one side of the photo is darker than the other. **Morphological opening and dilation** then remove residual noise and thicken the strokes.

### 2. Segmentation
`findContours()` locates candidate shapes, each wrapped in a bounding box. Boxes are then **filtered** — too small (noise), too large (background or overlapping shapes), or with an unrealistic aspect ratio — and finally **sorted left to right** so a multi-digit number is read in the right order.

### 3. Recognition (CNN, PyTorch)
Each segmented digit is padded to a square, resized to **28×28** and normalized to match the MNIST format, then classified by a **Convolutional Neural Network**: two convolutional layers (edges/curves, then loops and intersections), max pooling, two dropout layers against overfitting, and two fully connected layers producing a probability over the ten digits.

The model is trained on **MNIST** (70,000 labeled images) with **cross-entropy loss** and the **Adam** optimizer, then saved to `model.pt`. If no trained model is found on launch, the system **trains one automatically** — in a background thread, so the interface stays responsive and shows a live progress bar.

### 4. Interface (CustomTkinter)
A desktop GUI where the user can load an image from disk or capture one from the **webcam**, follow a live technical log of every processing step, and see the recognized number plus a history of recent results.

<p align="center">
  <img src="images/segmentation.png" alt="Digit segmentation with bounding boxes" width="600"/>
</p>

## 📈 Results & critical analysis

The system reliably reads clean, well-separated handwritten numbers. Testing on real handwriting also exposed the honest limits of the approach:

- **Digits written as a bare stroke** (a `1` drawn like an `I`) are sometimes dropped at the segmentation stage, having no distinctive contour.
- **Broken strokes** can split one digit into two boxes — a disconnected `3` read as two separate shapes.
- **Visually close pairs** get confused depending on handwriting style: `9`/`4`, `4`/`7`, `1`/`2`, `5`/`6`, `0`/`6`.

Most of these failures trace back to **segmentation and handwriting style rather than the classifier itself** — MNIST is clean, centered and uniform, while real handwriting is not. Training on augmented or more varied data, and making segmentation robust to broken strokes, are the natural next steps.

## 🛠️ Tech stack

`Python` · `PyTorch` · `OpenCV` · `NumPy` · `CustomTkinter` · `Pillow`

## ▶️ How to run

```bash
pip install -r requirements.txt
python TextDetectionFunctional.py
```

On first launch, if `model.pt` is missing the CNN trains itself on MNIST (a few minutes to ~20 min depending on your machine) and saves the weights for later runs.

📄 Full project report: [`Final_Report.pdf`](Final_Report.pdf)

---

<p align="center"><a href="https://github.com/Jiboti">⬅️ Back to profile</a></p>
