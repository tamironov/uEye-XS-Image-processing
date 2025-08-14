# uEye-XS-Image-processing
iDS uEye XS Vision Tester
A Python-based GUI tool for vision testing using IDS uEye XS cameras.

**This application allows you to:**
- Select an ROI (Region of Interest) in a live camera feed
- Calibrate by capturing a set of reference images from the ROI
- Run automated tests comparing live frames against the reference images
- Detect pixel-level changes in the selected area (PASS/FAIL result)
- Uses CLAHE preprocessing and basic template alignment for improved accuracy

**Features:**
- Live camera preview (uEye XS)
- Mouse-based ROI selection
- 10× reference image capture for calibration
- CLAHE preprocessing for better contrast
- Template alignment to reduce ROI shift errors
- Pixel-difference detection with configurable thresholds
- Real-time PASS/FAIL display

**Requirements:**
You will need:
- Python 3.8+
- An IDS uEye XS camera with drivers installed
install IDS Software Suite 4.97 for Windows 32/64-bit
https://en.ids-imaging.com/download-details/AB02716.html

*Make sure to choose the correct camera model (in this case it was UI-1007XS-C)

- Required Python packages:
pip install customtkinter opencv-python numpy pillow pyueye

Note:
pyueye must be installed from IDS' SDK or Python bindings, not from PyPI.
Download the uEye SDK from IDS Imaging.

**How It Works**
1) Launch the application
The camera initializes with a short progress bar.
2)Select ROI
Click and drag on the video feed to mark the area of interest.
3)Calibrate
Click 📸 Calibrate (10x) to store 10 reference images from the ROI.
4)Start Testing
Click ▶️ Start Test to compare the live ROI against reference images.
The app will show:
PASS status ✅ if no significant change is detected
FAIL status ❌ if a change greater than the threshold is found
Reset
Use the Reset button to clear the ROI and start again.

**Key Parameters**
- Number of reference images: self.num_ref_images = 10
- Pixel change threshold: 0.5% (adjust in pixel_diff_change method)
- Calibration delay per image: 0.2 sec

**Example Use Cases**
Detecting defective pixels or missing segments on LCD/7-segment displays
Checking if a mechanical assembly has shifted
Monitoring a fixed test object for visual changes

**Notes:**
- Ensure your uEye XS camera is connected and recognized by the IDS uEye Cockpit software before running.
- The reference images are stored in ./reference/ for each run.
- CLAHE improves detection in varying lighting conditions.

