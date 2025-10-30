import os
import sys
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from threading import Thread

# Try to import matplotlib and configure for tkinter
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    plt.style.use('default')
except ImportError:
    # Define a dummy class for graceful failure if dependencies are missing
    class CompactCSVProcessor(ttk.Frame):
        def __init__(self, parent):
            super().__init__(parent, padding=20)
            self.pack(fill='both', expand=True)
            ttk.Label(self, text="MATPLOTLIB/NUMPY/PANDAS MISSING\nPlease install them using: pip install matplotlib numpy pandas ttkbootstrap", bootstyle="danger").pack(expand=True)
    
# --- Normal Class Definition starts here, only if imports succeeded ---

if 'matplotlib' in sys.modules:
    class CompactCSVProcessor(ttk.Frame): # Changed to inherit from ttk.Frame
        """
        CSV processor optimized for display within a ttkbootstrap tab.
        """

        def __init__(self, parent):
            super().__init__(parent) # Initialize as a Frame
            self.pack(fill='both', expand=True) # Essential for filling the notebook tab

            self.root = parent.winfo_toplevel() # Use the top-level window for dialogs/update

            # Application variables
            self.directory_path = ttk.StringVar()
            self.combined_df = None
            self.output_file = None
            self.fig = None
            self.canvas = None
            self.ax = None
            self.plot_saved_path = None # To store path of automatically saved plot

            # Physical constants
            self.c = 2.9979e8
            self.lba = 850e-9
            self.points = 4096
            self.len_spec = self.points * 2
            self.fnq = (self.c / self.lba) / 2
            self.max_v = ((2 * self.fnq) / self.c) * (1 / 100)
            self.spac = ((2 * self.fnq) / (self.len_spec * self.c)) * 0.01

            # Fixed ranges
            self.freq_range = [1000, 1550] 
            self.value_range = [-0.06, 0.06]
            
            # Color thresholds
            self.fail_threshold = 0.025
            self.pass_threshold = 0.02
            self.upper_fail_threshold = self.fail_threshold
            self.lower_fail_threshold = -self.fail_threshold
            self.upper_pass_threshold = self.pass_threshold
            self.lower_pass_threshold = -self.pass_threshold

            # Initialize modern GUI, passing self as the parent frame
            self.setup_modern_gui(parent=self) 


        def setup_modern_gui(self, parent):
            """Create modern GUI optimized for the tab area"""
            
            # Main container for the entire application layout
            main_frame = ttk.Frame(parent, padding=5)
            main_frame.pack(fill='both', expand=True)

            # Configure main layout: left panel for controls, right panel for plot
            main_frame.columnconfigure(0, weight=0, minsize=260)  # Control panel fixed width
            main_frame.columnconfigure(1, weight=1)              # Plot area expandable
            main_frame.rowconfigure(0, weight=1)

            # Left control panel
            self.setup_control_panel(main_frame)

            # Right plot panel
            self.setup_plot_panel(main_frame)

        def setup_control_panel(self, parent):
            """Setup modern control panel - 260px width"""

            # Control panel frame
            control_frame = ttk.Frame(parent, width=260, style='TFrame')
            control_frame.grid(row=0, column=0, sticky='ns', padx=(5, 2), pady=5)
            control_frame.pack_propagate(False)

            # Title section
            title_frame = ttk.Frame(control_frame)
            title_frame.pack(fill='x', pady=(8, 5))

            ttk.Label(
                title_frame,
                text="CSV QC Processor",
                bootstyle="primary",
                font=("Helvetica", 14, "bold")
            ).pack()

            ttk.Label(
                title_frame,
                text="Spectrometer Quality Control",
                bootstyle="secondary"
            ).pack()
            
            # File selection section
            file_frame = ttk.LabelFrame(
                control_frame,
                text="Select Folder",
                padding=(8, 5)
            )
            file_frame.pack(fill='x', padx=8, pady=(5, 8))

            # File input row
            input_frame = ttk.Frame(file_frame)
            input_frame.pack(fill='x')

            self.path_entry = ttk.Entry(
                input_frame,
                textvariable=self.directory_path,
                bootstyle="primary"
            )
            self.path_entry.pack(side='top', fill='x', pady=(0, 5))

            # Browse button
            browse_btn = ttk.Button(
                input_frame,
                text="Browse",
                command=self.browse_folder,
                bootstyle="primary"
            )
            browse_btn.pack(fill='x')

            # Process button
            process_frame = ttk.Frame(control_frame)
            process_frame.pack(fill='x', padx=8, pady=5)

            self.process_btn = ttk.Button(
                process_frame,
                text="Process & Plot",
                command=self.process_and_plot,
                bootstyle="success"
            )
            self.process_btn.pack(fill='x')

            # Control buttons section
            controls_frame = ttk.LabelFrame(
                control_frame,
                text="Plot Controls",
                padding=(8, 5)
            )
            controls_frame.pack(fill='x', padx=8, pady=(5, 8))
            
            # First row of buttons: Zoom In, Zoom Out
            button_row1 = ttk.Frame(controls_frame)
            button_row1.pack(fill='x', pady=2)

            self.zoom_in_btn = ttk.Button(
                button_row1,
                text="Zoom In",
                command=self.zoom_in,
                bootstyle="info-outline",
                state='disabled',
                width=10
            )
            self.zoom_in_btn.pack(side='left', padx=(0, 2), fill='x', expand=True)

            self.zoom_out_btn = ttk.Button(
                button_row1,
                text="Zoom Out",
                command=self.zoom_out,
                bootstyle="info-outline",
                state='disabled',
                width=10
            )
            self.zoom_out_btn.pack(side='right', padx=(2, 0), fill='x', expand=True)

            # Second row: Reset View button
            button_row2 = ttk.Frame(controls_frame)
            button_row2.pack(fill='x', pady=2)

            self.reset_btn = ttk.Button(
                button_row2,
                text="Reset View",
                command=self.reset_view,
                bootstyle="secondary",
                state='disabled'
            )
            self.reset_btn.pack(fill='x')

            # Third row: Save PNG button - this will now be for additional saves
            button_row3 = ttk.Frame(controls_frame)
            button_row3.pack(fill='x', pady=2)

            self.save_png_btn = ttk.Button(
                button_row3,
                text="Save PNG (Additional)",
                command=self.save_as_png,
                bootstyle="warning",
                state='disabled'
            )
            self.save_png_btn.pack(fill='x')

            # Status section
            status_frame = ttk.LabelFrame(
                control_frame,
                text="Status",
                padding=(8, 5)
            )
            status_frame.pack(fill='both', expand=True, padx=8, pady=(5, 8))

            # Progress bar
            self.progress = ttk.Progressbar(
                status_frame,
                mode='indeterminate',
                length=200,
                bootstyle="info"
            )
            self.progress.pack(pady=5)

            # Status label
            self.status_label = ttk.Label(
                status_frame,
                text="Ready to process files",
                wraplength=200,
                justify='center',
                bootstyle="secondary"
            )
            self.status_label.pack(fill='x', pady=2)

            # File count label
            self.file_count_label = ttk.Label(
                status_frame,
                text="No files processed",
                wraplength=200,
                justify='center',
                bootstyle="dark"
            )
            self.file_count_label.pack(fill='x')

        def setup_plot_panel(self, parent):
            """Setup plot panel to expand within the tab"""

            # Plot panel frame
            plot_frame = ttk.Frame(parent, style='TFrame')
            plot_frame.grid(row=0, column=1, sticky='nsew', padx=(2, 5), pady=5)

            # Configure plot frame to be expandable
            plot_frame.rowconfigure(1, weight=1)
            plot_frame.columnconfigure(0, weight=1)

            # Plot title
            title_frame = ttk.Frame(plot_frame)
            title_frame.grid(row=0, column=0, sticky='ew', pady=5)

            ttk.Label(
                title_frame,
                text="Data Visualization",
                bootstyle="primary"
            ).pack()

            # Plot content area - this is the key area for the plot
            self.plot_content = ttk.Frame(plot_frame, bootstyle="secondary")
            self.plot_content.grid(row=1, column=0, sticky='nsew', padx=5, pady=(0, 5))

            # Configure plot content for centering
            self.plot_content.rowconfigure(0, weight=1)
            self.plot_content.columnconfigure(0, weight=1)

            # Show placeholder
            self.show_modern_placeholder()

        def show_modern_placeholder(self):
            """Show modern placeholder for plot area"""
            self.clear_plot_area() # Ensure previous plot is cleared

            placeholder = ttk.Frame(self.plot_content, bootstyle="light")
            placeholder.grid(row=0, column=0, sticky='nsew')
            placeholder.rowconfigure(0, weight=1)
            placeholder.columnconfigure(0, weight=1)

            content_frame = ttk.Frame(placeholder)
            content_frame.grid(row=0, column=0)

            ttk.Label(
                content_frame,
                text="📊",
                bootstyle="secondary",
                font=("Helvetica", 30)
            ).pack()

            ttk.Label(
                content_frame,
                text="Plot Area - Select a Folder and Process",
                bootstyle="secondary",
                font=("Helvetica", 12)
            ).pack(pady=5)
            
        def clear_plot_area(self):
            """Clear the plot area"""
            for widget in self.plot_content.winfo_children():
                widget.destroy()

        def browse_folder(self):
            """Browse for folder selection"""
            folder_path = filedialog.askdirectory(
                title="Select folder with CSV files"
            )
            if folder_path:
                self.directory_path.set(folder_path)
                folder_name = os.path.basename(folder_path)
                self.update_status(f"Selected: {folder_name}")

        def update_status(self, message):
            """Update status message"""
            self.status_label.config(text=message)
            self.root.update()

        def update_file_count(self, message):
            """Update file count message"""
            self.file_count_label.config(text=message)
            self.root.update()

        def concatenate_csv_files(self, directory):
            """Process and concatenate CSV files. The combined file is automatically saved."""
            try:
                v_axis = np.arange(1, self.len_spec + 1) * (self.spac * 2)
                dataframes = []
                found_files = []

                self.update_status("Searching files...")

                # Find matching files
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        if '_sample_abs_' in file.lower() and file.endswith('.csv'):
                            file_path = os.path.join(root, file)
                            found_files.append((file_path, file))

                if not found_files:
                    return None, None

                self.update_file_count(f"Found: {len(found_files)} files")

                # Process files
                for i, (file_path, file_name) in enumerate(found_files):
                    self.update_status(f"Processing {i+1}/{len(found_files)}")

                    try:
                        df = pd.DataFrame(pd.read_csv(file_path, header=None))

                        if len(df.columns) <= len(v_axis):
                            df.columns = np.round(v_axis[:len(df.columns)], 4)
                        else:
                            extended_v_axis = list(np.round(v_axis, 4)) + [f"col_{j}" for j in range(len(v_axis), len(df.columns))]
                            df.columns = extended_v_axis

                        df['fn'] = file_name
                        df = df[['fn'] + [col for col in df.columns if col != 'fn']]
                        dataframes.append(df)

                    except Exception:
                        continue

                if not dataframes:
                    return None, None

                self.update_status("Combining files...")

                # Combine data
                combined_df = pd.concat(dataframes, axis=0, ignore_index=True)

                # Save combined file - AUTOMATICALLY SAVED
                folder_name = os.path.basename(directory.rstrip(os.sep))
                output_filename = f"{folder_name}_combined.csv"
                output_file = os.path.join(directory, output_filename)
                combined_df.to_csv(output_file, index=False)
                self.update_status(f"Combined data saved to {output_filename}")

                return output_file, combined_df

            except Exception:
                # Log or display the exception for debugging if needed
                return None, None

        
        def create_compact_plot(self, df):
            """
            Create plot optimized for compact display with conditional coloring,
            text overlay, and automatic saving.
            """
            passed_count = 0
            failed_count = 0
            total_samples_checked = 0

            try:
                self.update_status("Creating plot...")
                self.clear_plot_area()

                # Prepare data
                x_values = [float(col) for col in df.columns[1:] if col != 'fn']
                y_data = df.iloc[:, 1:]

                # Filter to frequency range
                freq_mask = [(x >= self.freq_range[0] and x <= self.freq_range[1]) for x in x_values]
                if any(freq_mask):
                    filtered_x = [x for x, mask in zip(x_values, freq_mask) if mask]
                    filtered_y_data = y_data.iloc[:, [i for i, mask in enumerate(freq_mask) if mask]]
                    x_values = filtered_x
                    y_data = filtered_y_data

                # --- Fix: increase margins and ensure layout stays visible ---
                self.fig = Figure(figsize=(8, 6), dpi=100, layout='constrained')
                self.fig.patch.set_facecolor('white')
                self.ax = self.fig.add_subplot(111)

                # Plot data with conditional coloring
                for i in range(len(df)):
                    y_row = y_data.iloc[i].values
                    is_failed = np.any(y_row > self.upper_fail_threshold) or np.any(y_row < self.lower_fail_threshold)
                    color = '#e74c3c' if is_failed else '#bdc3c7'
                    label = f"{df['fn'].iloc[i]} ({'Failed' if is_failed else 'Passed'})"

                    if i < 5:
                        if is_failed:
                            failed_count += 1
                        else:
                            passed_count += 1

                    self.ax.plot(x_values, y_row, color=color, linewidth=1.8, alpha=0.9, label=label)

                # Status overlay
                total_samples_checked = min(len(df), 5)
                status_text, status_color = "", ""

                if total_samples_checked > 0:
                    if passed_count >= 3:
                        status_text, status_color = "Cleaning Successful", "green"
                    elif failed_count >= 2:
                        status_text, status_color = "Cleaning Unsuccessful Repeat Cleaning", "red"

                if status_text:
                    self.ax.text(
                        0.5, 0.5, status_text,
                        ha='center', va='center',
                        transform=self.ax.transAxes,
                        fontsize=20, color=status_color,
                        fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5')
                    )

                # Reference lines
                self.ax.axhline(y=self.upper_pass_threshold, color='#2c3e50', linestyle='--', linewidth=2, alpha=0.8)
                self.ax.axhline(y=self.lower_pass_threshold, color='#2c3e50', linestyle='--', linewidth=2, alpha=0.8)
                self.ax.axhline(y=0, color='#34495e', linestyle='-', linewidth=1, alpha=0.6)

                # Ranges and labels
                self.ax.set_xlim(self.freq_range)
                self.ax.set_ylim(self.value_range)
                self.ax.set_xlabel('Frequency', fontsize=10, fontweight='bold', labelpad=10)
                self.ax.set_ylabel('Values', fontsize=10, fontweight='bold', labelpad=10)
                self.ax.set_title(
                    f'Data Visualization Freq: {self.freq_range[0]}–{self.freq_range[1]} | Values: {self.value_range[0]} to {self.value_range[1]}',
                    fontsize=9, fontweight='bold', pad=12
                )

                # Grid and styling
                self.ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                self.ax.set_facecolor('#fafbfc')

                # Legend (only if few signals)
                if len(df) <= 6:
                    handles, labels = self.ax.get_legend_handles_labels()
                    filtered = [(h, l) for h, l in zip(handles, labels) if "(Failed)" in l or "(Passed)" in l]
                    if filtered:
                        legend = self.ax.legend(*zip(*filtered), fontsize=7, loc='upper right', framealpha=0.9)
                        legend.get_frame().set_facecolor('#ffffff')

                # --- Fix: Use constrained layout and extra margin to prevent cropping ---
                self.fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.12)

                # Embed into Tkinter
                self.canvas = FigureCanvasTkAgg(self.fig, self.plot_content)
                self.canvas.draw()
                canvas_widget = self.canvas.get_tk_widget()
                canvas_widget.grid(row=0, column=0, sticky='nsew')

                # Enable buttons
                self.zoom_in_btn.config(state='normal')
                self.zoom_out_btn.config(state='normal')
                self.reset_btn.config(state='normal')
                self.save_png_btn.config(state='normal')

                # --- Fixed: Save PNG with full visible axes ---
                if self.directory_path.get():
                    plot_output_filename = "Cleaning.png"
                    self.plot_saved_path = os.path.join(self.directory_path.get(), plot_output_filename)
                    self.fig.savefig(
                        self.plot_saved_path,
                        dpi=150,
                        bbox_inches=None,
                        pad_inches=0.3,
                        facecolor='white',
                        edgecolor='none'
                    )
                    self.update_status(f"Plot auto-saved as {plot_output_filename}")

                return True

            except Exception as e:
                messagebox.showerror("Plot Error", f"Failed to create plot: {str(e)}")
                return False

        def zoom_in(self):
            """Zoom in functionality"""
            if self.ax:
                xlim = self.ax.get_xlim()
                ylim = self.ax.get_ylim()

                x_center = (xlim[0] + xlim[1]) / 2
                y_center = (ylim[0] + ylim[1]) / 2
                x_range = (xlim[1] - xlim[0]) * 0.7
                y_range = (ylim[1] - ylim[0]) * 0.7

                self.ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
                self.ax.set_ylim(y_center - y_range/2, y_center + y_range/2)

                self.canvas.draw()
                self.update_status("Zoomed in")

        def zoom_out(self):
            """Zoom out functionality"""
            if self.ax:
                xlim = self.ax.get_xlim()
                ylim = self.ax.get_ylim()

                x_center = (xlim[0] + xlim[1]) / 2
                y_center = (ylim[0] + ylim[1]) / 2
                x_range = (xlim[1] - xlim[0]) * 1.4
                y_range = (ylim[1] - ylim[0]) * 1.4

                # Apply limits
                x_new_min = max(self.freq_range[0], x_center - x_range/2)
                x_new_max = min(self.freq_range[1], x_center + x_range/2)
                y_new_min = max(self.value_range[0], y_center - y_range/2)
                y_new_max = min(self.value_range[1], y_center + y_range/2)

                self.ax.set_xlim(x_new_min, x_new_max)
                self.ax.set_ylim(y_new_min, y_new_max)

                self.canvas.draw()
                self.update_status("Zoomed out")

        def reset_view(self):
            """Reset view to original ranges"""
            if self.ax:
                self.ax.set_xlim(self.freq_range)
                self.ax.set_ylim(self.value_range)
                self.canvas.draw()
                self.update_status("View reset")

        def save_as_png(self):
            """Save plot as PNG (for additional user saves)"""
            if not self.fig:
                messagebox.showerror("Error", "No plot to save!")
                return

            try:
                # Suggest the automatically saved file as a default
                initial_file = self.plot_saved_path if self.plot_saved_path else "custom_plot.png"

                file_path = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    filetypes=[("PNG images", "*.png"), ("All files", "*.*")],
                    title="Save Plot (Additional)",
                    initialfile=os.path.basename(initial_file),
                    initialdir=os.path.dirname(initial_file) if self.plot_saved_path else os.getcwd()
                )

                if file_path:
                    self.update_status("Saving additional PNG...")
                    self.fig.savefig(file_path, dpi=150, bbox_inches='tight',
                                     facecolor='white', edgecolor='none')
                    self.update_status("Additional PNG saved!")
                    messagebox.showinfo("Success", f"Plot saved:\n{file_path}")

            except Exception as e:
                messagebox.showerror("Error", f"Save failed: {str(e)}")
                self.update_status("Save failed")

        def process_and_plot(self):
            """Main processing function"""
            if not self.directory_path.get().strip():
                messagebox.showerror("Error", "Please select a folder first!")
                return

            if not os.path.exists(self.directory_path.get()):
                messagebox.showerror("Error", "Selected folder does not exist!")
                return

            Thread(target=self._process_thread, daemon=True).start()

        def _process_thread(self):
            """Processing thread"""
            try:
                self.progress.start()
                self.process_btn.config(state='disabled')

                # Process files (combined CSV is automatically saved inside this function)
                self.update_status("Processing...")
                self.output_file, self.combined_df = self.concatenate_csv_files(self.directory_path.get())

                if self.combined_df is None or self.combined_df.empty:
                    self.update_status("No files found")
                    self.update_file_count("No valid files")
                    messagebox.showwarning("No Files", "No valid CSV files found in the selected folder.")
                    return

                # Create plot
                self.update_status("Creating plot...")
                plot_success = self.create_compact_plot(self.combined_df)

                if not plot_success:
                    self.update_status("Plot failed")
                    return

                # Success
                self.update_status("Ready - Plot displayed and auto-saved")
                self.update_file_count(f"Processed: {len(self.combined_df)} files")

            except Exception as e:
                self.update_status("Error occurred")
                messagebox.showerror("Error", f"Processing failed: {str(e)}")

            finally:
                self.progress.stop()
                self.process_btn.config(state='normal')