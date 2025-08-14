# main.py
import sys
import os

# Add the parent directory to the sys.path to allow imports from app, camera_handler, and image_processor
# This is crucial if you run main.py directly from its directory and the other files are siblings.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from app import VisionTestApp

if __name__ == "__main__":
    # Create an instance of the VisionTestApp
    app = VisionTestApp()
    # Set the protocol for when the window is closed, ensuring proper camera shutdown
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    # Start the CustomTkinter event loop
    app.mainloop()
