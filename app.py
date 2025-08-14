# app.py
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import os
import time

# Import the new modules
from camera_handler import CameraHandler
from image_processor import ImageProcessor

class VisionTestApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("uEye XS Vision Tester")
        self.geometry("1000x750")
        ctk.set_appearance_mode("dark")

        # Initialize modules
        self.image_processor = ImageProcessor()
        # CameraHandler will be initialized with callbacks for status updates and frame delivery
        self.camera_handler = CameraHandler(
            on_status_update=self.update_status,
            on_frame_ready=self.process_and_display_frame,
            on_init_progress=self.update_init_progress,
            on_init_complete=self.on_camera_init_complete
        )

        self.last_frame = None # Stores the last captured frame for ROI selection and calibration
        self.roi = None # Not directly used, roi_start/end define the ROI
        self.ref_folder = "./reference/" # Folder to store reference images
        self.num_ref_images = 10 # Number of reference images to capture for calibration
        self.testing = False # Flag to indicate if the test is currently running
        self.result_text = ctk.StringVar(value="Status: Initializing camera...") # Status message for the top label
        os.makedirs(self.ref_folder, exist_ok=True) # Ensure reference folder exists

        # --- GUI Elements ---

        # Status Label ABOVE the video feed (shows initializing & ready messages)
        self.status_label = ctk.CTkLabel(self, textvariable=self.result_text, font=("Arial", 20, "bold"), text_color="white")
        self.status_label.pack(pady=(15, 5))

        # Progress bar for initial camera setup/delay
        self.init_progress = ctk.CTkProgressBar(self, width=600)
        self.init_progress.set(0)
        self.init_progress.pack(pady=(0, 10))

        # Video Label - where the camera feed will be displayed
        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.pack(padx=10, pady=(0, 10))
        # Bind mouse events for ROI selection
        self.video_label.bind("<Button-1>", self.mark_roi_start)
        self.video_label.bind("<B1-Motion>", self.mark_roi_drag)
        self.video_label.bind("<ButtonRelease-1>", self.mark_roi_end)
        self.roi_rect = None # Placeholder for drawing ROI rectangle (not directly used by CTkLabel)

        # Buttons Frame to group control buttons
        buttons_frame = ctk.CTkFrame(self)
        buttons_frame.pack(pady=10)

        self.select_roi_button = ctk.CTkButton(buttons_frame, text="🎯 Select ROI", command=self.enable_roi_selection)
        self.select_roi_button.grid(row=0, column=0, padx=5)

        self.calibrate_button = ctk.CTkButton(buttons_frame, text="📸 Calibrate (10x)", command=self.calibrate_reference)
        self.calibrate_button.grid(row=0, column=1, padx=5)

        self.reset_button = ctk.CTkButton(buttons_frame, text="🔄 Reset", command=self.reset_app)
        self.reset_button.grid(row=0, column=2, padx=5)

        self.start_button = ctk.CTkButton(self, text="▶️ Start Test", command=self.start_testing)
        self.start_button.pack(pady=10)

        # Status label below start button for test results (PASS/FAIL)
        self.test_result_text = ctk.StringVar(value="")
        self.test_result_label = ctk.CTkLabel(self, textvariable=self.test_result_text, font=("Arial", 20, "bold"))
        self.test_result_label.pack(pady=(5, 15))

        # Progress bar for calibration (hidden initially)
        self.progress = ctk.CTkProgressBar(self, width=600)
        self.progress.set(0)
        self.progress.pack(pady=10)
        self.progress.pack_forget() # Hide it initially

        # --- Application State Variables ---
        self.running = True # Flag to control the main application loop and camera thread
        self.roi_start = None # (x,y) coordinate for the start of ROI selection
        self.roi_end = None # (x,y) coordinate for the end of ROI selection
        self.roi_enabled = False # Flag to indicate if ROI selection is active/complete

        # Start the camera capture in a separate thread to keep the GUI responsive
        self.capture_thread = threading.Thread(target=self.camera_handler.start_capture_loop, daemon=True)
        self.capture_thread.start()

    def update_status(self, text, color="white"):
        """Updates the main status label text and color."""
        self.result_text.set(text)
        self.status_label.configure(text_color=color)

    def update_test_result(self, text, color="white"):
        """Updates the test result label text and color."""
        self.test_result_text.set(text)
        self.test_result_label.configure(text_color=color)

    def update_init_progress(self, value):
        """Updates the initialization progress bar."""
        # Use .after() to ensure GUI updates happen on the main thread
        self.after(0, lambda: self.init_progress.set(value))
        self.after(0, self.update) # Force GUI update on main thread

    def on_camera_init_complete(self):
        """Callback executed when camera initialization is finished."""
        # Use .after() to ensure GUI updates happen on the main thread
        self.after(0, self.init_progress.pack_forget)

    def enable_roi_selection(self):
        """Prepares the app for ROI selection by the user."""
        self.update_status("Select ROI by dragging mouse over the video.", "yellow")
        self.update_test_result("")
        self.roi_enabled = False # Allow new ROI selection
        self.roi_start = None
        self.roi_end = None

    def reset_app(self):
        """Resets the application state, clearing ROI and test results."""
        self.roi_start = None
        self.roi_end = None
        self.roi_enabled = False
        self.testing = False
        self.progress.pack_forget() # Hide calibration progress bar
        self.update_status("Reset done. Please select ROI again.", "white")
        self.update_test_result("")

    def mark_roi_start(self, event):
        """Records the starting point of the ROI selection."""
        if not self.roi_enabled:
            self.roi_start = (event.x, event.y)

    def mark_roi_drag(self, event):
        """Updates the end point of the ROI selection while dragging."""
        if not self.roi_enabled and self.roi_start:
            self.roi_end = (event.x, event.y)

    def mark_roi_end(self, event):
        """Finalizes the ROI selection."""
        if not self.roi_enabled and self.roi_start:
            self.roi_end = (event.x, event.y)
            self.roi_enabled = True # Lock ROI after selection
            self.update_status("ROI selected. Ready to calibrate.", "yellow")

    def get_roi_box(self):
        """Calculates and returns the ROI coordinates (x, y, width, height)."""
        if self.roi_start and self.roi_end:
            x1, y1 = self.roi_start
            x2, y2 = self.roi_end
            # Ensure coordinates are in the correct order (top-left to bottom-right)
            return (min(x1,x2), min(y1,y2), abs(x2 - x1), abs(y2 - y1))
        return None

    def calibrate_reference(self):
        """
        Starts the calibration process in a separate thread to keep the GUI responsive.
        """
        if self.last_frame is None or not self.get_roi_box():
            self.update_status("Status: No frame or ROI ❌", "red")
            return

        # Start the actual calibration logic in a new thread
        calibration_thread = threading.Thread(target=self._run_calibration_threaded, daemon=True)
        calibration_thread.start()

    def _run_calibration_threaded(self):
        """
        Contains the core calibration logic, run in a separate thread.
        All GUI updates are scheduled using self.after() for thread safety.
        """
        x, y, w, h = self.get_roi_box()
        self.after(0, lambda: self.update_status("Calibrating... please wait", "yellow"))
        self.after(0, lambda: self.progress.set(0))
        self.after(0, lambda: self.progress.pack(pady=10)) # Show calibration progress bar
        self.after(0, lambda: self.update_test_result(""))

        # Loop to capture and save reference images
        for i in range(self.num_ref_images):
            # Ensure last_frame is not None before slicing
            if self.last_frame is not None:
                roi_img = self.last_frame[y:y+h, x:x+w]
                filename = os.path.join(self.ref_folder, f"ref_{i}.png")
                cv2.imwrite(filename, roi_img)
                self.after(0, lambda: self.progress.set((i+1)/self.num_ref_images))
                self.after(0, self.update) # Force GUI update
                time.sleep(0.2) # Small delay between captures

        self.after(0, self.progress.pack_forget) # Hide calibration progress bar
        self.after(0, lambda: self.update_status("Calibration complete ✅", "green"))


    def start_testing(self):
        """Initiates the vision test using the captured reference images."""
        if not self.get_roi_box():
            self.update_status("No ROI selected ❌", "red")
            return

        # Check if all reference images exist
        for i in range(self.num_ref_images):
            path = os.path.join(self.ref_folder, f"ref_{i}.png")
            if not os.path.exists(path):
                self.update_status("Missing reference images ❌", "red")
                return

        self.testing = True # Set testing flag to true
        self.update_status("Testing...", "yellow")
        self.update_test_result("")

    def process_and_display_frame(self, frame):
        """
        Callback function called by CameraHandler when a new frame is ready.
        Processes the frame, updates the display, and performs testing if active.
        """
        self.last_frame = frame.copy() # Store the current frame

        display_frame = frame.copy()
        # Draw ROI rectangle if selected
        if self.get_roi_box():
            x, y, w, h = self.get_roi_box()
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0,255,0), 2)

        # Convert frame to PhotoImage for CustomTkinter display
        img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img_pil = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        # Use .after() to ensure GUI updates happen on the main thread
        self.after(0, lambda: self.video_label.configure(image=img_pil))
        self.after(0, lambda: setattr(self.video_label, 'image', img_pil))


        if self.testing:
            # Get the live ROI
            x, y, w, h = self.get_roi_box()
            roi_live = frame[y:y+h, x:x+w]
            # Preprocess the live ROI using the ImageProcessor
            live_proc = self.image_processor.preprocess_with_clahe(roi_live)

            max_diff_ratio = 0
            change_detected = False

            # Compare live ROI with all reference images
            for i in range(self.num_ref_images):
                ref_path = os.path.join(self.ref_folder, f"ref_{i}.png")
                ref_img = cv2.imread(ref_path)
                if ref_img is None:
                    continue

                # Preprocess and align reference image
                ref_proc = self.image_processor.preprocess_with_clahe(ref_img)
                aligned_live = self.image_processor.align_images(ref_proc, live_proc)
                # Calculate pixel difference
                change, diff_ratio = self.image_processor.pixel_diff_change(ref_proc, aligned_live)

                max_diff_ratio = max(max_diff_ratio, diff_ratio)
                if change:
                    change_detected = True
                    break # Stop if a significant change is detected

            # Update test result based on detection
            if change_detected:
                self.after(0, lambda: self.update_test_result(f"FAIL ❌ (Diff={max_diff_ratio:.2%})", "red"))
            else:
                self.after(0, lambda: self.update_test_result(f"PASS ✅ (Diff={max_diff_ratio:.2%})", "green"))

            self.testing = False # Reset testing flag after one test cycle

    def on_closing(self):
        """Handles the application closing event, stopping the camera thread gracefully."""
        self.running = False # Signal the camera thread to stop
        self.camera_handler.stop_capture_loop() # Tell camera handler to stop
        self.update_status("Closing...", "yellow")
        self.update() # Force GUI update
        # Give a short delay for the camera thread to finish
        self.after(100, self._shutdown_cleanup)

    def _shutdown_cleanup(self):
        """Performs final cleanup before destroying the window."""
        try:
            # Wait for the capture thread to finish (with a timeout)
            if self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2.0)
        except Exception as e:
            print(f"Error during thread join: {e}")
        finally:
            # Destroy the CustomTkinter window
            self.destroy()
