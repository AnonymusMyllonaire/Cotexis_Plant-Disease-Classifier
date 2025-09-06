import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import time

def select_image_file():
    """Open a file dialog to select an image file"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Set up file types for dialog
    file_types = [
        ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
        ("JPEG files", "*.jpg *.jpeg"),
        ("PNG files", "*.png"),
        ("All files", "*.*")
    ]
    
    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select an image file to analyze",
        filetypes=file_types
    )
    
    root.destroy()
    return file_path

def ask_continue():
    """Ask user if they want to continue"""
    root = tk.Tk()
    root.withdraw()
    
    response = messagebox.askyesno(
        "Continue?",
        "Do you want to analyze another image?",
        icon='question'
    )
    
    root.destroy()
    return response

def load_and_preprocess_image(img_path, target_size=(128, 128)):
    """
    Load and preprocess a single image for prediction
    """
    # Check if image exists
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")
    
    # Load image
    img = image.load_img(img_path, target_size=target_size)
    
    # Convert to array and normalize
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    return img_array

def predict_image(model, img_path):
    """
    Predict the class of a single image
    """
    # Class names (should match training order)
    class_names = ['angular_leaf_spot', 'bean_rust', 'healthy']

    # Load and preprocess the image
    try:
        processed_image = load_and_preprocess_image(img_path)
        print(f"✓ Image loaded successfully: {os.path.basename(img_path)}")
    except Exception as e:
        print(f"✗ Error loading image: {e}")
        return None, None

    # Make prediction
    print("🔍 Analyzing image...")
    start_time = time.time()
    predictions = model.predict(processed_image, verbose=0)
    prediction_time = time.time() - start_time
    
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    # Get class name
    predicted_class_name = class_names[predicted_class]

    # Display results
    print(f"\n{'='*60}")
    print(f"📊 PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"📁 Image: {os.path.basename(img_path)}")
    print(f"🎯 Predicted class: {predicted_class_name.upper()}")
    print(f"✅ Confidence: {confidence:.2%}")
    print(f"⏱️  Prediction time: {prediction_time:.2f} seconds")
    print(f"\n📈 All class probabilities:")
    
    # Create a bar for visualization
    for i, (class_name, prob) in enumerate(zip(class_names, predictions[0])):
        bar = "█" * int(prob * 20)  # Visual bar
        percentage = f"{prob:.1%}"
        print(f"{class_name:20s}: {percentage:>6s} {bar}")
    
    print(f"{'='*60}")

    # Display the image with prediction
    display_prediction_result(img_path, predicted_class_name, confidence, predictions[0])
    
    return predicted_class_name, confidence

def display_prediction_result(img_path, prediction, confidence, probabilities):
    """Display the image with prediction results"""
    try:
        # Create a figure with subplots
        fig = plt.figure(figsize=(16, 8))
        
        # Main grid
        gs = plt.GridSpec(2, 3, figure=fig)
        
        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        img = image.load_img(img_path)
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Processed image
        ax2 = fig.add_subplot(gs[0, 1])
        processed_img = image.load_img(img_path, target_size=(128, 128))
        ax2.imshow(processed_img)
        ax2.set_title('Processed Image (128x128)', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # Prediction result
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.text(0.5, 0.8, 'PREDICTION RESULT', 
                ha='center', va='center', fontsize=16, fontweight='bold', transform=ax3.transAxes)
        ax3.text(0.5, 0.6, f'{prediction.upper()}', 
                ha='center', va='center', fontsize=20, fontweight='bold', color='green', transform=ax3.transAxes)
        ax3.text(0.5, 0.4, f'Confidence: {confidence:.2%}', 
                ha='center', va='center', fontsize=14, transform=ax3.transAxes)
        ax3.axis('off')
        
        # Probability bar chart
        ax4 = fig.add_subplot(gs[1, :])
        classes = ['Angular Leaf Spot', 'Bean Rust', 'Healthy']
        colors = ['red', 'orange', 'green']
        
        bars = ax4.bar(classes, probabilities, color=colors, alpha=0.7)
        ax4.set_ylabel('Probability', fontweight='bold')
        ax4.set_title('Class Probabilities', fontsize=14, fontweight='bold')
        ax4.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.2%}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Could not display image: {e}")
        print("Saving result to 'prediction_result.png'")
        plt.savefig('prediction_result.png')

def create_sample_images():
    """Create sample images for testing if no images available"""
    print("Creating sample images for testing...")
    
    sample_dir = "sample_images"
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create a few sample images with different colors
    colors = [(34, 139, 34), (50, 205, 50), (144, 238, 144)]  # Different greens
    for i, color in enumerate(colors):
        img = Image.new('RGB', (300, 200), color=color)
        img.save(f"{sample_dir}/sample_leaf_{i+1}.jpg")
    
    print(f"Sample images created in '{sample_dir}' folder")
    return f"{sample_dir}/sample_leaf_1.jpg"

def main():
    """Main function with continuous image processing"""
    print("🌱 Bean Leaf Disease Classifier")
    print("=" * 50)
    print("This program will continue analyzing images until you choose to quit.")
    print("=" * 50)
    
    # Check if model exists
    model_path = 'plant_model.h5'
    if not os.path.exists(model_path):
        print("❌ Model not found! Please train the model first.")
        print("Run: python plant_classifier.py")
        return
    
    # Load model once at the beginning
    try:
        model = load_model(model_path)
        print(f"✓ Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Please make sure you have trained the model first.")
        return
    
    analysis_count = 0
    
    while True:
        analysis_count += 1
        print(f"\n📸 Analysis #{analysis_count}")
        print("-" * 30)
        
        # Open file dialog
        img_path = select_image_file()
        
        if not img_path:
            print("❌ No file selected. Exiting.")
            break
        
        print(f"📁 Selected file: {os.path.basename(img_path)}")
        
        # Make prediction
        try:
            predict_image(model, img_path)
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            print("Please try another image.")
            continue
        
        # Ask if user wants to continue
        try:
            continue_analysis = ask_continue()
            if not continue_analysis:
                print("\n👋 Thank you for using the Bean Leaf Disease Classifier!")
                print(f"📊 Total images analyzed: {analysis_count}")
                break
        except:
            # If dialog fails, ask in console
            response = input("\nDo you want to analyze another image? (y/n): ").lower().strip()
            if response not in ['y', 'yes']:
                print("\n👋 Thank you for using the Bean Leaf Disease Classifier!")
                print(f"📊 Total images analyzed: {analysis_count}")
                break

def quick_test_mode():
    """Quick test mode without dialogs for debugging"""
    model_path = 'plant_model.h5'
    
    if not os.path.exists(model_path):
        print("Model not found. Please train first.")
        return
    
    try:
        model = load_model(model_path)
        print("Model loaded successfully!")
    except:
        print("Error loading model.")
        return
    
    print("Quick test mode: Enter image paths (or 'quit' to exit)")
    
    while True:
        img_path = input("\nEnter image path: ").strip()
        
        if img_path.lower() in ['quit', 'exit', 'q']:
            break
        
        if not os.path.exists(img_path):
            print("File not found. Please try again.")
            continue
        
        try:
            predict_image(model, img_path)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Simple command line interface
    parser = argparse.ArgumentParser(description='Predict plant disease from images')
    parser.add_argument('--image', type=str, help='Single image to predict')
    parser.add_argument('--quick', action='store_true', help='Use quick text-based mode')
    
    args = parser.parse_args()
    
    if args.image:
        # Single image mode
        model_path = 'plant_model.h5'
        if os.path.exists(model_path):
            try:
                model = load_model(model_path)
                predict_image(model, args.image)
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Model not found. Please train first.")
    elif args.quick:
        # Quick text mode
        quick_test_mode()
    else:
        # Interactive mode with file dialogs
        main()