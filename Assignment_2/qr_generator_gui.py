"""
QR Code Generator (Separate Window Version)
Company: Biox Systems
Description: Generates a QR code from a URL, displays it in a new window,
             and allows the user to download the QR code image.
Author: Arun Bhaskar Gadde
Date: 01/18/2026
"""

import qrcode
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

def generate_qr():
    """
    Generates a QR code and opens it in a separate window.
    """
    url = url_entry.get().strip()

    if not url:
        messagebox.showerror("Input Error", "Please enter a valid URL.")
        return

    # Generate QR code
    qr_image = qrcode.make(url).resize((300, 300))

    # Open new window to display QR code
    qr_window = tk.Toplevel(root)
    qr_window.title("Generated QR Code")
    qr_window.geometry("350x400")

    # Convert image for Tkinter
    qr_photo = ImageTk.PhotoImage(qr_image)

    qr_label = tk.Label(qr_window, image=qr_photo)
    qr_label.image = qr_photo
    qr_label.pack(pady=10)

    def save_qr():
        """
        Saves the QR code image to the user's system.
        """
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")],
            title="Save QR Code"
        )

        if file_path:
            qr_image.save(file_path)
            messagebox.showinfo("Success", "QR Code saved successfully.")

    # Download button
    save_button = tk.Button(
        qr_window,
        text="Download QR Code",
        command=save_qr
    )
    save_button.pack(pady=10)

# Main application window
root = tk.Tk()
root.title("QR Code Generator - Biox Systems")
root.geometry("400x250")

# UI Elements
tk.Label(root, text="Enter URL:", font=("Arial", 12)).pack(pady=10)
url_entry = tk.Entry(root, width=45)
url_entry.pack(pady=5)

tk.Button(
    root,
    text="Generate QR Code",
    command=generate_qr
).pack(pady=15)

# Run application
root.mainloop()
