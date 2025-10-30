import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import os
import matplotlib.pyplot as plt
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import datetime
import socket
import numpy as np  # ✅ For consistent Y-axis tick intervals
import re  # ✅ For numeric extraction


class App2Frame(ttk.Frame):
    """Stats File Combiner & Plotter with ttkbootstrap UI"""
    def __init__(self, parent):
        super().__init__(parent)
        self.combined_df = pd.DataFrame()
        self.output_file = ""
        self.folder_prefix = ""
        self.build_ui()

    def build_ui(self):
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill='both', expand=True)
        main_frame.columnconfigure(0, weight=1)

        ttk.Label(main_frame, text="Stats File Combiner & Plotter", bootstyle="primary", font=("Helvetica", 16, "bold")).pack(pady=10)
        ttk.Label(main_frame, text="1. Select folder with '_stats.xlsx' files. 2. Choose columns. 3. Plot.", bootstyle="secondary").pack(pady=5)

        ttk.Button(
            main_frame,
            text="📂 Select Stats Folder & Combine Files",
            command=self.select_and_process_folder,
            bootstyle="info",
            width=40
        ).pack(pady=15)

        dropdown_frame = ttk.Frame(main_frame)
        dropdown_frame.pack(pady=10)

        ttk.Label(dropdown_frame, text="Select column for Line Plot:").grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.line_var = tk.StringVar()
        self.line_dropdown = ttk.Combobox(dropdown_frame, textvariable=self.line_var, state="readonly", width=35, bootstyle="info")
        self.line_dropdown.grid(row=1, column=0, padx=10, pady=5)

        ttk.Label(dropdown_frame, text="Select column for Scatter Plot:").grid(row=0, column=1, padx=10, pady=5, sticky='w')
        self.scatter_var = tk.StringVar()
        self.scatter_dropdown = ttk.Combobox(dropdown_frame, textvariable=self.scatter_var, state="readonly", width=35, bootstyle="info")
        self.scatter_dropdown.grid(row=1, column=1, padx=10, pady=5)

        self.plot_button = ttk.Button(
            main_frame,
            text="📈 Generate & Save Plots",
            command=self.plot_data,
            bootstyle="success",
            width=30,
            state=tk.DISABLED
        )
        self.plot_button.pack(pady=20)

    def select_and_process_folder(self):
        folder_path = filedialog.askdirectory(title="Select Folder with Stats Files")
        if not folder_path:
            return

        self.combined_df = pd.DataFrame()
        error_occurred = False
        self.folder_prefix = folder_path

        for file in os.listdir(folder_path):
            if '_stats' in file.lower() and file.endswith('.xlsx') and 'Combined_Stability_Stats.xlsx' not in file:
                try:
                    file_path = os.path.join(folder_path, file)
                    df = pd.read_excel(file_path)

                    if 'Source File' not in df.columns:
                        df.insert(0, 'Source File', os.path.splitext(file)[0])

                    if 'peak_raw_maxima_amp' in df.columns and 'peak_raw_minima_amp' in df.columns:
                        df['final_amp'] = df['peak_raw_maxima_amp'] - df['peak_raw_minima_amp']

                    self.combined_df = pd.concat([self.combined_df, df], ignore_index=True)
                except Exception as e:
                    print(f"Error processing {file}: {e}")
                    error_occurred = True

        if not self.combined_df.empty:
            self.output_file = os.path.join(folder_path, 'Combined_Stability_Stats.xlsx')
            try:
                if os.path.exists(self.output_file):
                    confirm = messagebox.askyesno(
                        "File Exists",
                        f"The file {self.output_file} already exists.\nDo you want to overwrite it?"
                    )
                    if not confirm:
                        messagebox.showinfo("Cancelled", "Operation cancelled. File not overwritten.")
                        return

                self.combined_df.to_excel(self.output_file, index=False)
                messagebox.showinfo("Success", f"Files combined successfully.\nSaved at:\n{self.output_file}")
                self.update_column_options()
                self.plot_button.config(state=tk.NORMAL)

            except PermissionError:
                messagebox.showerror("File in Use", f"The file {self.output_file} is open or locked.\nPlease close it and try again.")
            except Exception as e:
                messagebox.showerror("Save Error", f"Could not save the file:\n{e}")
                error_occurred = True

        else:
            messagebox.showwarning("No Files", "No valid stats files were found or processed.")

        if error_occurred:
            messagebox.showwarning("Errors Occurred", "Some errors occurred during processing.\nCheck the console output.")

    def update_column_options(self):
        numeric_cols = self.combined_df.select_dtypes(include='number').columns.tolist()
        self.line_dropdown['values'] = numeric_cols
        self.scatter_dropdown['values'] = numeric_cols

        if 'final_amp' in numeric_cols:
            self.line_var.set('final_amp')
        elif numeric_cols:
            self.line_var.set(numeric_cols[0])

        if 'up_laser_zc_std' in numeric_cols:
            self.scatter_var.set('up_laser_zc_std')
        elif len(numeric_cols) > 1:
            self.scatter_var.set(numeric_cols[1])

    def plot_data(self):
        if self.combined_df.empty or not self.folder_prefix:
            messagebox.showerror("No Data/Folder", "No data available or folder path not set.")
            return

        line_col = self.line_var.get()
        scatter_col = self.scatter_var.get()

        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        fig.patch.set_facecolor('#f0f2f5')

        # Helper function to extract leading numeric part
        def extract_number(value):
            match = re.match(r'(\d+)', str(value))
            return int(match.group(1)) if match else value

        # --- Line Plot (Amplitude) ---
        if line_col and line_col in self.combined_df.columns:
            line_df = self.combined_df[['Source File', line_col]].dropna().copy()
            line_df['Numeric_ID'] = line_df['Source File'].apply(extract_number)

            y1 = line_df[line_col]
            x1_labels = line_df['Numeric_ID']
            x1_pos = range(len(y1))

            line_title = "Amplitude" if line_col == "final_amp" else f"{line_col} (Line Plot)"
            axs[0].plot(x1_pos, y1, marker='o', linestyle='-', color='#3498db', linewidth=2, markersize=5)
            axs[0].set_title(line_title, fontsize=14, fontweight='bold')
            axs[0].set_xlabel("Sample No.", fontsize=12)
            axs[0].set_ylabel(line_col, fontsize=12)
            axs[0].set_xticks(x1_pos)
            axs[0].set_xticklabels(x1_labels, rotation=45, ha='right', fontsize=8)

            # ✅ Force Y-axis to start at 0 with 0.1 step
            y_max = y1.max()
            y_limit = np.ceil(y_max * 1.1 * 10) / 10
            axs[0].set_ylim(0, y_limit)
            axs[0].set_yticks(np.arange(0, y_limit + 0.1, 0.1))
            axs[0].grid(True, linestyle='--', alpha=0.6)
            axs[0].set_facecolor('#ecf0f1')
        else:
            axs[0].text(0.5, 0.5, f"'{line_col}' not found or selected", ha='center', va='center', fontsize=12, color='red')
            axs[0].axis('off')

        # --- Scatter Plot ---
        if scatter_col and scatter_col in self.combined_df.columns:
            scatter_df = self.combined_df[['Source File', scatter_col]].dropna().copy()
            scatter_df['Numeric_ID'] = scatter_df['Source File'].apply(extract_number)

            y2 = scatter_df[scatter_col]
            x2_labels = scatter_df['Numeric_ID']
            x2_pos = range(len(y2))

            axs[1].scatter(x2_pos, y2, color='#e74c3c', s=20, alpha=0.7)
            axs[1].set_title(f"{scatter_col}", fontsize=14, fontweight='bold')
            axs[1].set_xlabel("Sample No.", fontsize=12)
            axs[1].set_ylabel(scatter_col, fontsize=12)
            axs[1].set_xticks(x2_pos)
            axs[1].set_xticklabels(x2_labels, rotation=45, ha='right', fontsize=8)
            axs[1].grid(True, linestyle='--', alpha=0.6)
            axs[1].set_facecolor('#ecf0f1')
        else:
            axs[1].text(0.5, 0.5, f"'{scatter_col}' not found or selected", ha='center', va='center', fontsize=12, color='red')
            axs[1].axis('off')

        plt.tight_layout()

        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name_prefix = os.path.basename(self.folder_prefix)
            plot_filename = f"Plot_{folder_name_prefix}.jpg"
            save_path = os.path.join(self.folder_prefix, plot_filename)
            fig.savefig(save_path, dpi=300, format='jpeg')
            messagebox.showinfo("Plot Saved", f"Plots generated and saved successfully as:\n{plot_filename}")
        except Exception as e:
            messagebox.showerror("Plot Save Error", f"Could not save the plot file:\n{e}")

        plt.show()


# ==============================================================
# --- Main Application Setup ---
# ==============================================================
if __name__ == "__main__":
    try:
        app = ttk.Window(title="Stats Data Tool", themename="flatly")
        app.geometry("800x600")
        app_frame = App2Frame(app)
        app_frame.pack(fill='both', expand=True)
        app.mainloop()
    except Exception as e:
        messagebox.showerror("Fatal Error", f"The application encountered a critical error: {e}")
