import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import sys

# Attempt to import application modules
try:
    from app1_processor import App1Frame
    from app2_viewer import App2Frame
except ImportError as e:
    messagebox.showerror("Import Error", f"Failed to import a necessary module: {e}. Please ensure app1_processor.py and app2_viewer.py are in the same directory.")

class MainApplication(ttk.Window):
    """
    Central GUI application using ttkbootstrap for a modern, responsive full-screen interface.
    """
    def __init__(self):
        super().__init__(themename="flatly")
        self.title("Spectra Analysis Platform")

        # --- Fullscreen and Responsive Setup ---
        self.state('zoomed')  # Windows full-screen
        self.minsize(900, 600)  # Minimum size to keep layout stable
        self.bind("<F11>", self.toggle_fullscreen)
        self.bind("<Escape>", self.exit_fullscreen)

        # Configure the main grid for responsiveness
        # Row 0: Header Title
        # Row 1: Main Notebook
        self.rowconfigure(0, weight=0)  # Header row, no vertical expansion
        self.rowconfigure(1, weight=1)  # Notebook row, expands to fill
        self.columnconfigure(0, weight=1) # Single column, expands to fill

        self.fullscreen = True
        self.create_widgets()

    def toggle_fullscreen(self, event=None):
        """Toggle fullscreen mode with F11"""
        self.fullscreen = not self.fullscreen
        self.attributes("-fullscreen", self.fullscreen)
        return "break"

    def exit_fullscreen(self, event=None):
        """Exit fullscreen with Escape"""
        self.fullscreen = False
        self.attributes("-fullscreen", False)
        return "break"

    def create_widgets(self):
        # --- Header Title ---
        # This label is placed in the top row and configured to center its text.
        header_label = ttk.Label(
            self,
            text="Spectra Analysis Platform",
            font="-size 16 -weight bold",
            anchor="center" # Centers the text within the label
        )
        # Sticky 'ew' makes the label span horizontally, and 'anchor="center"'
        # ensures the text inside it is centered.
        header_label.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))


        # --- Notebook (tabbed interface) ---
        # Placed in the second row (row=1)
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))

        # Ensure tabs fill space proportionally
        self.notebook.rowconfigure(0, weight=1)
        self.notebook.columnconfigure(0, weight=1)

        # ----------------------------------------
        # Tab 1: Processing (App1Frame)
        # ----------------------------------------
        frame1 = ttk.Frame(self.notebook, padding=5)
        frame1.grid(row=0, column=0, sticky="nsew")
        self.notebook.add(frame1, text=" Stats Process")

        self.app1 = App1Frame(frame1)
        self.app1.pack(fill="both", expand=True)

        # ----------------------------------------
        # Tab 2: Combined Stats Viewer (App2Frame)
        # ----------------------------------------
        frame2 = ttk.Frame(self.notebook, padding=5)
        frame2.grid(row=0, column=0, sticky="nsew")
        self.notebook.add(frame2, text=" Combine & Plot Stats")

        self.app2 = App2Frame(frame2)
        self.app2.pack(fill="both", expand=True)

        # ----------------------------------------
        # Tab 3: CSV Plotting/Quality Control
        # ----------------------------------------
        try:
            from app3_cleaning_data import CompactCSVProcessor
            frame3 = ttk.Frame(self.notebook, padding=5)
            frame3.grid(row=0, column=0, sticky="nsew")
            self.notebook.add(frame3, text="Cleaning Test")

            self.app3 = CompactCSVProcessor(frame3)
            self.app3.pack(fill="both", expand=True)

        except ImportError as e:
            frame3_error = ttk.Frame(self.notebook, padding=20)
            self.notebook.add(frame3_error, text=" Cleaning Test(Error)")
            ttk.Label(
                frame3_error,
                text=f"Error loading App 3: {e}\nEnsure app3_cleaning_data.py is present.",
                bootstyle="danger"
            ).pack(expand=True)


if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()
