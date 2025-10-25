# Edge Detection Algorithms - Introduction to Computer Vision
## Project Description
Assignment 1 for the course Introduction to Computer Vision. This is a Python based project. This project implements an edge detection application using PyQt5, OpenCV, and NumPy. The application allows users to apply and adjust parameters for the following edge detection filters:  
- **Canny**: Lower/upper thresholds, Gaussian blur kernel size, and sigma.  
- **Sobel**: Kernel size and gradient direction (X, Y, or both).  
- **Laplacian**: Kernel size.
## Features
- Multiple edge detection algorithms (Sobel, Laplacian, Canny)
- Real-time parameter adjustment
- Dark/Light theme support
- Image upload and export functionality
- Clean, modern UI
## Requirements
- Python 3.7+
- OpenCV
- PyQt5
- NumPy
## Installation & Usage
1. Clone this repository:
```bash
git clone https://github.com/yourusername/edge-detection.git
```
2. Enter the project folder
```bash
cd edge-detector
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
Or install manually
```bash
pip install opencv-python pyqt5 numpy
```
4. Launch the application
```bash
python main.py
```
## Screenshots
1. Application in dark mode
<img width="959" height="506" alt="image" src="https://github.com/user-attachments/assets/79c918f6-3c6d-4add-a940-8ada46b36adb" />

2. Sobel filter on an image. UI in light mode
<img width="959" height="506" alt="image" src="https://github.com/user-attachments/assets/bff90ee8-b04c-4b8b-985f-81037d7ded20" />

3. Laplacian filter on an image. UI in dark mode
<img width="959" height="506" alt="image" src="https://github.com/user-attachments/assets/bce24d4a-5fc2-4a9f-87da-6af8c07128e1" />

4. Save image output window
<img width="959" height="505" alt="image" src="https://github.com/user-attachments/assets/9398dd49-c597-49a6-b7a7-51d701e58b58" />
