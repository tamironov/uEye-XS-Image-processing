# camera_handler.py
from pyueye import ueye
import numpy as np
import cv2
import time

class CameraHandler:
    def __init__(self, on_status_update=None, on_frame_ready=None, on_init_progress=None, on_init_complete=None):
        """
        Initializes the CameraHandler.
        :param on_status_update: Callback function to update application status (text, color).
        :param on_frame_ready: Callback function to send captured frames to the application.
        :param on_init_progress: Callback function to update initialization progress.
        :param on_init_complete: Callback function to signal when camera initialization is complete.
        """
        self.hCam = ueye.HIDS(0) # Camera handle
        self.pcImageMemory = ueye.c_mem_p() # Pointer to image memory
        self.MemID = ueye.int() # ID of the image memory
        self.width, self.height = 640, 480 # Default camera resolution
        self.pitch = ueye.int() # Pitch of the image memory
        self.bitspixel = ueye.int(24) # 24 bits per pixel for BGR8_PACKED

        self.running = False # Flag to control the camera capture loop
        self.on_status_update = on_status_update
        self.on_frame_ready = on_frame_ready
        self.on_init_progress = on_init_progress
        self.on_init_complete = on_init_complete # New callback

    def _update_status(self, text, color="white"):
        """Helper to call the status update callback if provided."""
        if self.on_status_update:
            self.on_status_update(text, color)

    def _update_init_progress(self, value):
        """Helper to call the initialization progress update callback if provided."""
        if self.on_init_progress:
            self.on_init_progress(value)

    def _on_init_complete(self):
        """Helper to call the initialization complete callback if provided."""
        if self.on_init_complete:
            self.on_init_complete()

    def initialize_camera(self):
        """Initializes the uEye camera."""
        try:
            # Initialize camera
            ret = ueye.is_InitCamera(self.hCam, None)
            if ret != ueye.IS_SUCCESS:
                self._update_status(f"Camera Init Failed: {ret} ❌", "red")
                return False

            # Set color mode
            ret = ueye.is_SetColorMode(self.hCam, ueye.IS_CM_BGR8_PACKED)
            if ret != ueye.IS_SUCCESS:
                self._update_status(f"Set Color Mode Failed: {ret} ❌", "red")
                self.release_camera()
                return False

            # Set Area Of Interest (AOI) / resolution
            rect_aoi = ueye.IS_RECT()
            rect_aoi.s32X, rect_aoi.s32Y = ueye.int(0), ueye.int(0)
            rect_aoi.s32Width, rect_aoi.s32Height = ueye.int(self.width), ueye.int(self.height)
            ret = ueye.is_AOI(self.hCam, ueye.IS_AOI_IMAGE_SET_AOI, rect_aoi, ueye.sizeof(rect_aoi))
            if ret != ueye.IS_SUCCESS:
                self._update_status(f"Set AOI Failed: {ret} ❌", "red")
                self.release_camera()
                return False

            # Allocate image memory
            ret = ueye.is_AllocImageMem(self.hCam, self.width, self.height, self.bitspixel, self.pcImageMemory, self.MemID)
            if ret != ueye.IS_SUCCESS:
                self._update_status(f"Alloc Image Mem Failed: {ret} ❌", "red")
                self.release_camera()
                return False

            # Set image memory
            ret = ueye.is_SetImageMem(self.hCam, self.pcImageMemory, self.MemID)
            if ret != ueye.IS_SUCCESS:
                self._update_status(f"Set Image Mem Failed: {ret} ❌", "red")
                self.release_camera()
                return False

            # Set display mode
            ret = ueye.is_SetDisplayMode(self.hCam, ueye.IS_SET_DM_DIB)
            if ret != ueye.IS_SUCCESS:
                self._update_status(f"Set Display Mode Failed: {ret} ❌", "red")
                self.release_camera()
                return False

            # Start continuous video capture
            ret = ueye.is_CaptureVideo(self.hCam, ueye.IS_DONT_WAIT)
            if ret != ueye.IS_SUCCESS:
                self._update_status(f"Start Capture Video Failed: {ret} ❌", "red")
                self.release_camera()
                return False

            # Simulate initialization delay with progress bar
            delay_sec = 5
            steps = 50
            for i in range(steps):
                time.sleep(delay_sec / steps)
                self._update_init_progress((i+1)/steps)

            self._update_status("Camera Ready. Please select ROI.", "white")
            self._on_init_complete() # Signal completion
            self.running = True
            return True

        except Exception as e:
            self._update_status(f"Camera Initialization Error: {e} ❌", "red")
            self.release_camera()
            return False

    def start_capture_loop(self):
        """Starts the main camera frame capture loop."""
        if not self.initialize_camera():
            return

        while self.running:
            # Inquire image memory to get current frame data
            ret = ueye.is_InquireImageMem(self.hCam, self.pcImageMemory, self.MemID,
                                         ueye.int(self.width), ueye.int(self.height),
                                         self.bitspixel, self.pitch)
            if ret != ueye.IS_SUCCESS:
                # Handle error or continue if no new frame
                time.sleep(0.01) # Small delay to prevent busy-waiting
                continue

            # Get data from image memory
            array = ueye.get_data(self.pcImageMemory, self.width, self.height,
                                  self.bitspixel, self.pitch, copy=True)
            
            if array is None or array.size == 0:
                time.sleep(0.01)
                continue

            # Reshape the array into a BGR image (height, width, channels)
            frame = np.reshape(array, (self.height, self.width, 3))

            # Send the frame to the application for processing and display
            if self.on_frame_ready:
                self.on_frame_ready(frame)

            time.sleep(0.03) # Control frame rate

        self.release_camera() # Ensure camera is released when loop stops

    def stop_capture_loop(self):
        """Stops the camera capture loop."""
        self.running = False

    def release_camera(self):
        """Releases the camera resources."""
        if self.hCam:
            ueye.is_StopLiveVideo(self.hCam, ueye.IS_FORCE_VIDEO_STOP)
            ueye.is_FreeImageMem(self.hCam, self.pcImageMemory, self.MemID)
            ueye.is_ExitCamera(self.hCam)
            self._update_status("Camera Disconnected.", "gray")
            self.hCam = None # Clear handle
