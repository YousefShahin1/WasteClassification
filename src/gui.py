"""
Modern GUI for Waste Classification Model
Train, Evaluate, and Predict with an intuitive interface
Uses FastAPI backend for all operations
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
from pathlib import Path
import torch
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import io
import requests
import time
import subprocess

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_TIMEOUT = 30


class WasteClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Waste Classification - AI Model Manager")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        # Variables
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        self.training_thread = None
        self.api_process = None
        self.api_running = False
        
        # Style configuration
        self.setup_styles()
        
        # Check API status
        self.check_api_status()
        
        # Create main container
        self.create_header()
        self.create_notebook()
        self.create_status_bar()
        
    def check_api_status(self):
        """Check if API server is running"""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                self.api_running = True
                return True
        except:
            self.api_running = False
        
        # Ask user if they want to start the API
        result = messagebox.askyesno(
            "API Server Not Running",
            "The API server is not running. Would you like to start it now?\n\n"
            "The API server is required for predictions and some features."
        )
        
        if result:
            self.start_api_server()
        
        return False
    
    def start_api_server(self):
        """Start the API server in background"""
        def run_api():
            try:
                self.log_train("Starting API server...\n")
                self.api_process = subprocess.Popen(
                    ['python', 'api.py'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=os.path.dirname(os.path.abspath(__file__))
                )
                
                # Wait a few seconds for server to start
                time.sleep(3)
                
                # Check if it's running
                if self.check_api_status():
                    self.log_train("✅ API server started successfully!\n")
                    self.update_status("API server running")
                else:
                    self.log_train("⚠️ API server may not have started correctly\n")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start API server: {str(e)}")
        
        threading.Thread(target=run_api, daemon=True).start()
        
    def setup_styles(self):
        """Configure modern styling"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        self.colors = {
            'primary': '#2196F3',
            'secondary': '#4CAF50',
            'accent': '#FF9800',
            'danger': '#F44336',
            'dark': '#333333',
            'light': '#ffffff',
            'bg': '#f5f5f5'
        }
        
        # Button styles
        style.configure('Primary.TButton',
                       background=self.colors['primary'],
                       foreground='white',
                       padding=10,
                       font=('Segoe UI', 10, 'bold'))
        
        style.configure('Success.TButton',
                       background=self.colors['secondary'],
                       foreground='white',
                       padding=10,
                       font=('Segoe UI', 10, 'bold'))
        
        style.configure('Danger.TButton',
                       background=self.colors['danger'],
                       foreground='white',
                       padding=10,
                       font=('Segoe UI', 10, 'bold'))
        
        # Frame styles
        style.configure('Card.TFrame',
                       background='white',
                       relief='flat')
        
        style.configure('TNotebook',
                       background=self.colors['bg'],
                       borderwidth=0)
        
        style.configure('TNotebook.Tab',
                       padding=[20, 10],
                       font=('Segoe UI', 10, 'bold'))
        
    def create_header(self):
        """Create header with title and device info"""
        header = tk.Frame(self.root, bg=self.colors['primary'], height=80)
        header.pack(fill='x', side='top')
        header.pack_propagate(False)
        
        title = tk.Label(header,
                        text="🗑️ Waste Classification AI",
                        font=('Segoe UI', 24, 'bold'),
                        bg=self.colors['primary'],
                        fg='white')
        title.pack(side='left', padx=20, pady=15)
        
        device_label = tk.Label(header,
                               text=f"Device: {self.device.upper()}",
                               font=('Segoe UI', 12),
                               bg=self.colors['primary'],
                               fg='white')
        device_label.pack(side='right', padx=20)
        
    def create_notebook(self):
        """Create tabbed interface"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.train_tab = self.create_train_tab()
        self.evaluate_tab = self.create_evaluate_tab()
        self.predict_tab = self.create_predict_tab()
        
        self.notebook.add(self.train_tab, text="🎯 Train Model")
        self.notebook.add(self.evaluate_tab, text="📊 Evaluate Model")
        self.notebook.add(self.predict_tab, text="🔍 Predict")
        
    def create_train_tab(self):
        """Create training tab"""
        tab = ttk.Frame(self.notebook, style='Card.TFrame')
        
        # Left panel - Configuration
        left_panel = ttk.Frame(tab, style='Card.TFrame')
        left_panel.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        
        # Configuration card
        config_card = self.create_card(left_panel, "Training Configuration")
        
        # Epochs
        epochs_frame = ttk.Frame(config_card)
        epochs_frame.pack(fill='x', pady=5)
        ttk.Label(epochs_frame, text="Epochs:", font=('Segoe UI', 10)).pack(side='left')
        self.epochs_var = tk.StringVar(value="45")
        ttk.Entry(epochs_frame, textvariable=self.epochs_var, width=10).pack(side='left', padx=10)
        
        # Batch size
        batch_frame = ttk.Frame(config_card)
        batch_frame.pack(fill='x', pady=5)
        ttk.Label(batch_frame, text="Batch Size:", font=('Segoe UI', 10)).pack(side='left')
        self.batch_var = tk.StringVar(value="32")
        ttk.Entry(batch_frame, textvariable=self.batch_var, width=10).pack(side='left', padx=10)
        
        # Learning rate
        lr_frame = ttk.Frame(config_card)
        lr_frame.pack(fill='x', pady=5)
        ttk.Label(lr_frame, text="Learning Rate:", font=('Segoe UI', 10)).pack(side='left')
        self.lr_var = tk.StringVar(value="0.001")
        ttk.Entry(lr_frame, textvariable=self.lr_var, width=10).pack(side='left', padx=10)
        
        # Hidden units
        hidden_frame = ttk.Frame(config_card)
        hidden_frame.pack(fill='x', pady=5)
        ttk.Label(hidden_frame, text="Hidden Units:", font=('Segoe UI', 10)).pack(side='left')
        self.hidden_var = tk.StringVar(value="64")
        ttk.Entry(hidden_frame, textvariable=self.hidden_var, width=10).pack(side='left', padx=10)
        
        # Resume training
        resume_frame = ttk.Frame(config_card)
        resume_frame.pack(fill='x', pady=10)
        self.resume_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(resume_frame, text="Resume from checkpoint", variable=self.resume_var).pack(side='left')
        
        # Buttons
        button_frame = ttk.Frame(config_card)
        button_frame.pack(fill='x', pady=20)
        
        self.train_btn = ttk.Button(button_frame,
                                    text="▶ Start Training",
                                    style='Success.TButton',
                                    command=self.start_training)
        self.train_btn.pack(side='left', padx=5)
        
        self.stop_btn = ttk.Button(button_frame,
                                   text="⏹ Stop Training",
                                   style='Danger.TButton',
                                   command=self.stop_training,
                                   state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        
        # Right panel - Output
        right_panel = ttk.Frame(tab, style='Card.TFrame')
        right_panel.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        # Output card
        output_card = self.create_card(right_panel, "Training Output")
        
        self.train_output = scrolledtext.ScrolledText(output_card,
                                                       height=25,
                                                       font=('Consolas', 9),
                                                       bg='#1e1e1e',
                                                       fg='#d4d4d4',
                                                       insertbackground='white')
        self.train_output.pack(fill='both', expand=True, pady=5)
        
        # Progress bar
        self.train_progress = ttk.Progressbar(output_card, mode='indeterminate')
        self.train_progress.pack(fill='x', pady=10)
        
        return tab
    
    def create_evaluate_tab(self):
        """Create evaluation tab"""
        tab = ttk.Frame(self.notebook, style='Card.TFrame')
        
        # Control panel
        control_panel = self.create_card(tab, "Evaluation Controls")
        control_panel.pack(fill='x', padx=10, pady=10)
        
        button_frame = ttk.Frame(control_panel)
        button_frame.pack(fill='x', pady=10)
        
        ttk.Button(button_frame,
                  text="ℹ️ API Info",
                  style='Primary.TButton',
                  command=self.show_api_info).pack(side='left', padx=5)
        
        ttk.Button(button_frame,
                  text="📈 Plot Accuracy",
                  style='Primary.TButton',
                  command=self.plot_accuracy).pack(side='left', padx=5)
        
        ttk.Button(button_frame,
                  text="📉 Plot Loss",
                  style='Primary.TButton',
                  command=self.plot_loss).pack(side='left', padx=5)
        
        ttk.Button(button_frame,
                  text="🎯 Confusion Matrix",
                  style='Primary.TButton',
                  command=self.show_confusion_matrix).pack(side='left', padx=5)
        
        ttk.Button(button_frame,
                  text="📊 Full Evaluation",
                  style='Success.TButton',
                  command=self.full_evaluation).pack(side='left', padx=5)
        
        # Results panel
        results_panel = self.create_card(tab, "Results")
        results_panel.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create matplotlib figure area
        self.eval_figure_frame = ttk.Frame(results_panel)
        self.eval_figure_frame.pack(fill='both', expand=True, pady=5)
        
        # Output text
        self.eval_output = scrolledtext.ScrolledText(results_panel,
                                                      height=10,
                                                      font=('Consolas', 9))
        self.eval_output.pack(fill='x', pady=5)
        
        return tab
    
    def create_predict_tab(self):
        """Create prediction tab"""
        tab = ttk.Frame(self.notebook, style='Card.TFrame')
        
        # Left panel - Image selection
        left_panel = ttk.Frame(tab, style='Card.TFrame')
        left_panel.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        
        # Controls
        control_card = self.create_card(left_panel, "Image Selection")
        
        ttk.Button(control_card,
                  text="📁 Select Image",
                  style='Primary.TButton',
                  command=self.select_image).pack(pady=10)
        
        # Image preview
        self.image_label = tk.Label(control_card, text="No image selected", bg='white')
        self.image_label.pack(pady=20, fill='both', expand=True)
        
        # Predict button
        self.predict_btn = ttk.Button(control_card,
                                      text="🔍 Predict",
                                      style='Success.TButton',
                                      command=self.predict,
                                      state='disabled')
        self.predict_btn.pack(pady=10)
        
        # Right panel - Results
        right_panel = ttk.Frame(tab, style='Card.TFrame')
        right_panel.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        # Results card
        results_card = self.create_card(right_panel, "Prediction Results")
        
        self.prediction_text = tk.Label(results_card,
                                       text="",
                                       font=('Segoe UI', 16, 'bold'),
                                       bg='white')
        self.prediction_text.pack(pady=20)
        
        self.confidence_text = tk.Label(results_card,
                                       text="",
                                       font=('Segoe UI', 12),
                                       bg='white')
        self.confidence_text.pack(pady=10)
        
        # Probabilities frame
        self.prob_frame = ttk.Frame(results_card)
        self.prob_frame.pack(fill='both', expand=True, pady=10)
        
        return tab
    
    def create_card(self, parent, title):
        """Create a card-style container"""
        frame = ttk.Frame(parent, style='Card.TFrame', relief='raised', borderwidth=1)
        frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        header = tk.Label(frame,
                         text=title,
                         font=('Segoe UI', 12, 'bold'),
                         bg='white',
                         anchor='w')
        header.pack(fill='x', padx=15, pady=(15, 5))
        
        content = ttk.Frame(frame, style='Card.TFrame')
        content.pack(fill='both', expand=True, padx=15, pady=(5, 15))
        
        return content
    
    def create_status_bar(self):
        """Create status bar"""
        self.status_bar = tk.Label(self.root,
                                  text="Ready",
                                  relief=tk.SUNKEN,
                                  anchor='w',
                                  font=('Segoe UI', 9),
                                  bg=self.colors['light'])
        self.status_bar.pack(side='bottom', fill='x')
        
    def update_status(self, message):
        """Update status bar message"""
        self.status_bar.config(text=message)
        self.root.update_idletasks()
        
    # Training methods
    def start_training(self):
        """Start model training via API"""
        if not self.api_running:
            messagebox.showerror("Error", "API server is not running. Please start it first.")
            return
        
        self.train_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.train_progress.start()
        self.train_output.delete(1.0, tk.END)
        
        def train():
            try:
                self.log_train("Starting training via API...")
                
                # Get parameters
                epochs = int(self.epochs_var.get())
                batch_size = int(self.batch_var.get())
                lr = float(self.lr_var.get())
                hidden_units = int(self.hidden_var.get())
                resume = self.resume_var.get()
                
                self.log_train(f"Configuration: Epochs={epochs}, Batch={batch_size}, LR={lr}, Hidden={hidden_units}")
                
                # Send training request to API
                response = requests.post(
                    f"{API_BASE_URL}/train",
                    json={
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate": lr,
                        "hidden_units": hidden_units,
                        "resume": resume
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    training_id = result['training_id']
                    self.log_train(f"✅ Training started! ID: {training_id}\n")
                    self.log_train("Monitoring training progress...\n")
                    
                    # Poll for training status
                    while True:
                        time.sleep(2)  # Poll every 2 seconds
                        
                        status_response = requests.get(f"{API_BASE_URL}/train/status", timeout=5)
                        if status_response.status_code == 200:
                            status = status_response.json()
                            
                            status_text = status['status']
                            message = status.get('message', '')
                            
                            if message:
                                self.log_train(f"Status: {status_text} - {message}")
                            
                            # Update progress
                            if status['total_epochs'] and status['total_epochs'] > 0:
                                current = status.get('current_epoch', 0)
                                total = status['total_epochs']
                                best_acc = status.get('best_accuracy', 0)
                                if best_acc:
                                    self.log_train(f"Epoch {current}/{total} - Best accuracy: {best_acc:.2f}%")
                            
                            # Check if training completed or failed
                            if status_text in ['completed', 'failed', 'stopped']:
                                if status_text == 'completed':
                                    best_acc = status.get('best_accuracy', 0)
                                    self.log_train(f"\n✅ Training complete! Best accuracy: {best_acc:.2f}%")
                                    messagebox.showinfo("Success", f"Training completed!\nBest accuracy: {best_acc:.2f}%")
                                elif status_text == 'failed':
                                    self.log_train(f"\n❌ Training failed: {message}")
                                    messagebox.showerror("Error", f"Training failed: {message}")
                                else:
                                    self.log_train(f"\n⏹ Training stopped: {message}")
                                break
                        else:
                            self.log_train(f"⚠️ Could not get training status")
                            time.sleep(5)
                            
                elif response.status_code == 409:
                    self.log_train("❌ Training is already in progress")
                    messagebox.showerror("Error", "Training is already in progress")
                else:
                    self.log_train(f"❌ Failed to start training: {response.status_code}")
                    messagebox.showerror("Error", f"Failed to start training: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                self.log_train("❌ Cannot connect to API server")
                messagebox.showerror("Error", "Cannot connect to API server. Please ensure it's running.")
            except Exception as e:
                self.log_train(f"\n❌ Error: {str(e)}")
                messagebox.showerror("Error", f"Training failed: {str(e)}")
            finally:
                self.train_progress.stop()
                self.train_btn.config(state='normal')
                self.stop_btn.config(state='disabled')
        
        self.training_thread = threading.Thread(target=train, daemon=True)
        self.training_thread.start()
        
    def stop_training(self):
        """Stop training via API"""
        if not self.api_running:
            messagebox.showerror("Error", "API server is not running")
            return
        
        try:
            response = requests.post(f"{API_BASE_URL}/train/stop", timeout=5)
            if response.status_code == 200:
                self.log_train("\n⏹ Stop request sent. Current epoch will complete.")
                messagebox.showinfo("Info", "Training stop requested. Current epoch will complete.")
            else:
                messagebox.showerror("Error", "Failed to stop training")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop training: {str(e)}")
        
    def log_train(self, message):
        """Log message to training output"""
        self.train_output.insert(tk.END, message + "\n")
        self.train_output.see(tk.END)
        self.root.update_idletasks()
        
    # Evaluation methods
    def show_api_info(self):
        """Show API model information"""
        if not self.api_running:
            messagebox.showwarning("Warning", "API server is not running")
            return
        
        try:
            response = requests.get(f"{API_BASE_URL}/model/info", timeout=API_TIMEOUT)
            if response.status_code == 200:
                info = response.json()
                
                self.eval_output.delete(1.0, tk.END)
                self.eval_output.insert(tk.END, "="*60 + "\n")
                self.eval_output.insert(tk.END, "MODEL INFORMATION\n")
                self.eval_output.insert(tk.END, "="*60 + "\n\n")
                self.eval_output.insert(tk.END, f"Model Path: {info['model_path']}\n")
                self.eval_output.insert(tk.END, f"Device: {info['device']}\n")
                self.eval_output.insert(tk.END, f"Number of Classes: {info['num_classes']}\n")
                self.eval_output.insert(tk.END, f"Classes: {', '.join(info['classes'])}\n")
                self.eval_output.insert(tk.END, f"Best Accuracy: {info['best_accuracy']:.2f}%\n\n")
                self.eval_output.insert(tk.END, "Architecture:\n")
                self.eval_output.insert(tk.END, info['model_architecture'])
            else:
                messagebox.showerror("Error", f"API error: {response.status_code}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get API info: {str(e)}")
    
    def plot_accuracy(self):
        """Plot accuracy curves"""
        try:
            history = torch.load('../models/training_history.pth')
            train_accs = history['train_accuracies']
            test_accs = history['test_accuracies']
            
            # Clear previous plot
            for widget in self.eval_figure_frame.winfo_children():
                widget.destroy()
            
            fig, ax = plt.subplots(figsize=(8, 5))
            epochs = range(1, len(train_accs) + 1)
            ax.plot(epochs, train_accs, label='Train Accuracy', marker='o')
            ax.plot(epochs, test_accs, label='Test Accuracy', marker='s')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Model Accuracy Over Epochs')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            canvas = FigureCanvasTkAgg(fig, self.eval_figure_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
            self.update_status("Accuracy plot displayed")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot accuracy: {str(e)}")
            
    def plot_loss(self):
        """Plot loss curves"""
        try:
            history = torch.load('../models/training_history.pth')
            train_losses = history['train_losses']
            test_losses = history['test_losses']
            
            # Clear previous plot
            for widget in self.eval_figure_frame.winfo_children():
                widget.destroy()
            
            fig, ax = plt.subplots(figsize=(8, 5))
            epochs = range(1, len(train_losses) + 1)
            ax.plot(epochs, train_losses, label='Train Loss', marker='o')
            ax.plot(epochs, test_losses, label='Test Loss', marker='s')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Model Loss Over Epochs')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            canvas = FigureCanvasTkAgg(fig, self.eval_figure_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
            self.update_status("Loss plot displayed")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot loss: {str(e)}")
            
    def show_confusion_matrix(self):
        """Show confusion matrix"""
        self.eval_output.delete(1.0, tk.END)
        self.eval_output.insert(tk.END, "Loading model and generating confusion matrix...\n")
        self.eval_output.insert(tk.END, "Running evaluation script...\n")
        
        def generate():
            try:
                import subprocess
                # Run the evaluate script which will generate the confusion matrix
                result = subprocess.run(['python', 'evaluate.py'],
                                      capture_output=True,
                                      text=True,
                                      cwd='.')
                self.eval_output.insert(tk.END, result.stdout)
                if result.stderr:
                    self.eval_output.insert(tk.END, "\n" + result.stderr)
                    
                self.eval_output.insert(tk.END, "\nConfusion matrix saved to confusion_matrix.png\n")
                messagebox.showinfo("Success", "Confusion matrix generated!")
            except Exception as e:
                self.eval_output.insert(tk.END, f"\nError: {str(e)}\n")
                messagebox.showerror("Error", f"Failed: {str(e)}")
        
        threading.Thread(target=generate, daemon=True).start()
        
    def full_evaluation(self):
        """Run full evaluation"""
        self.eval_output.delete(1.0, tk.END)
        self.eval_output.insert(tk.END, "Running full evaluation...\n")
        
        def evaluate():
            try:
                import subprocess
                result = subprocess.run(['python', 'evaluate.py'],
                                      capture_output=True,
                                      text=True,
                                      cwd='.')
                self.eval_output.insert(tk.END, result.stdout)
                if result.stderr:
                    self.eval_output.insert(tk.END, "\nErrors:\n" + result.stderr)
                messagebox.showinfo("Success", "Evaluation complete!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed: {str(e)}")
        
        threading.Thread(target=evaluate, daemon=True).start()
        
    # Prediction methods
    def select_image(self):
        """Select image for prediction"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )
        
        if file_path:
            self.selected_image_path = file_path
            
            # Display image
            img = Image.open(file_path)
            img.thumbnail((300, 300))
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
            
            self.predict_btn.config(state='normal')
            self.update_status(f"Image loaded: {os.path.basename(file_path)}")
            
    def predict(self):
        """Make prediction on selected image using API"""
        if not self.api_running:
            messagebox.showerror("Error", "API server is not running. Please start it first.")
            return
        
        try:
            # Send image to API
            with open(self.selected_image_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(
                    f"{API_BASE_URL}/predict",
                    files=files,
                    timeout=API_TIMEOUT
                )
            
            if response.status_code == 200:
                result = response.json()
                pred_class = result['predicted_class']
                confidence = result['confidence']
                probabilities = result['all_probabilities']
                
                # Display results
                self.prediction_text.config(text=f"Prediction: {pred_class.upper()}")
                self.confidence_text.config(text=f"Confidence: {confidence:.2f}%")
                
                # Clear previous probabilities
                for widget in self.prob_frame.winfo_children():
                    widget.destroy()
                
                # Display all probabilities
                tk.Label(self.prob_frame,
                        text="Class Probabilities:",
                        font=('Segoe UI', 10, 'bold'),
                        bg='white').pack(anchor='w', pady=5)
                
                for cls in self.classes:
                    prob = probabilities.get(cls, 0)
                    frame = ttk.Frame(self.prob_frame)
                    frame.pack(fill='x', pady=2)
                    
                    tk.Label(frame, text=f"{cls}:", width=10, anchor='w').pack(side='left')
                    
                    progress = ttk.Progressbar(frame, length=200, mode='determinate', value=prob)
                    progress.pack(side='left', padx=5)
                    
                    tk.Label(frame, text=f"{prob:.1f}%", width=8, anchor='e').pack(side='left')
                
                self.update_status(f"Predicted: {pred_class} ({confidence:.2f}%)")
            else:
                messagebox.showerror("Error", f"API error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            messagebox.showerror("Error", "Cannot connect to API server. Please ensure it's running.")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")


def main():
    """Launch the GUI"""
    root = tk.Tk()
    app = WasteClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
