import os
import threading
import customtkinter as ctk
from tkinter import filedialog
from pathlib import Path

# Import your existing logic
import doc_generator

# Set the theme
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")
ctk.set_window_scaling(2.0)  # Scales the overall window
ctk.set_widget_scaling(2.0)  # Scales text and UI elements

class DocGenApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("OpenClaw Doc Generator")
        self.geometry("1000x850")

        # --- Input Variables ---
        self.input_path = ctk.StringVar()
        self.output_path = ctk.StringVar()
        self.api_url = ctk.StringVar(value="http://localhost:8000/v1")
        self.api_key = ctk.StringVar(value="dummy-key")
        self.scan_mode = ctk.StringVar(value="incremental")

        self.build_ui()

    def build_ui(self):
        base_font = ctk.CTkFont(size=14)
        header_font = ctk.CTkFont(size=24, weight="bold")
        # Header
        ctk.CTkLabel(self, text="Documentation Generator", font=header_font).pack(pady=(20, 10))
        # Input Directory
        self._build_file_picker("Codebase Input:", self.input_path, self.select_input)
        
        # Output Directory
        self._build_file_picker("Docs Output:", self.output_path, self.select_output)

        # API Settings
        api_frame = ctk.CTkFrame(self)
        api_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(api_frame, text="API URL:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        ctk.CTkEntry(api_frame, textvariable=self.api_url, width=350).grid(row=0, column=1, padx=10, pady=10)
        
        ctk.CTkLabel(api_frame, text="API Key:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        ctk.CTkEntry(api_frame, textvariable=self.api_key, width=350, show="*").grid(row=1, column=1, padx=10, pady=10)

        # Scan Mode
        mode_frame = ctk.CTkFrame(self)
        mode_frame.pack(fill="x", padx=20, pady=10)
        ctk.CTkRadioButton(mode_frame, text="Incremental Scan", variable=self.scan_mode, value="incremental").pack(side="left", padx=20, pady=10)
        ctk.CTkRadioButton(mode_frame, text="Full Scan", variable=self.scan_mode, value="full").pack(side="left", padx=20, pady=10)

        # Run Button
        self.run_btn = ctk.CTkButton(self, text="Generate Docs", command=self.start_generation, height=40)
        self.run_btn.pack(pady=20)

        # Status Label
        self.status_label = ctk.CTkLabel(self, text="Ready", text_color="gray")
        self.status_label.pack()

    def _build_file_picker(self, label_text, string_var, command):
        frame = ctk.CTkFrame(self)
        frame.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(frame, text=label_text, width=120, anchor="w").pack(side="left", padx=10, pady=10)
        ctk.CTkEntry(frame, textvariable=string_var, width=280, state="disabled").pack(side="left", padx=5)
        ctk.CTkButton(frame, text="Browse", width=80, command=command).pack(side="right", padx=10)

    def select_input(self):
        folder = filedialog.askdirectory()
        if folder: self.input_path.set(folder)

    def select_output(self):
        folder = filedialog.askdirectory()
        if folder: self.output_path.set(folder)

    def start_generation(self):
        if not self.input_path.get() or not self.output_path.get():
            self.status_label.configure(text="Error: Please select input and output directories.", text_color="red")
            return

        # Disable button to prevent double-clicks
        self.run_btn.configure(state="disabled", text="Running...")
        self.status_label.configure(text="Generating documentation... check console for details.", text_color="yellow")

        # Run the heavy LLM tasks in a background thread so the GUI doesn't freeze
        threading.Thread(target=self._run_doc_generator, daemon=True).start()

    def _run_doc_generator(self):
        try:
            # Overwrite the global variables in your doc_generator script
            doc_generator.CODEBASE = Path(self.input_path.get())
            doc_generator.DOCS_DIR = Path(self.output_path.get())
            doc_generator.VLLM_BASE_URL = self.api_url.get()
            doc_generator.VLLM_API_KEY = self.api_key.get()

            # Execute the script
            doc_generator.run(self.scan_mode.get())
            
            self.status_label.configure(text="Success! Docs generated.", text_color="green")
        except Exception as e:
            self.status_label.configure(text=f"Error: {e}", text_color="red")
        finally:
            self.run_btn.configure(state="normal", text="Generate Docs")

if __name__ == "__main__":
    app = DocGenApp()
    app.mainloop()