The dataset has 10 dynamic sign language gestures:
1. Hello
2. Thank You
3. Yes
4. Please
5. Come
6. No
7. Sorry
8. Stop
9. Water
10. Help      
Each gesture was recorded using a standard RGB webcam.
20 samples were captured per gesture to ensure sufficient variation.
Each sample consisted of 30 consecutive frames.

Total dataset size: 10 gestures × 20 samples × 30 frames = 6,000 images
Preprocessing Pipeline:
1. Grayscale conversion— reduces colour noise and computational load.
2. Gaussian blur— smooths out high-frequency noise before differencing.
3. Frame differencing— the difference between the previous frame and current frame is computed to isolate movement.
4. Motion mask generation— the result is a white silhouette of the moving hand against a black background.
5. Resize to 128×128 — standardises input size for the CNN.
6. Pixel normalisation — pixel values scaled to [0,1].

Dataset Structure:
dataset/
├── hello/
│   ├── sample_01/
│   │   ├── frame_01.jpg
│   │   ├── frame_02.jpg
│   │   └── ... (30 frames)
│   ├── sample_02/
│   └── ... (20 samples)
├── thankyou/
├── yes/
├── please/
├── come/
├── no/
├── sorry/
├── stop/
├── water/
└── help/
|__hello/
