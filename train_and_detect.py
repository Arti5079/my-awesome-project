import os
from ultralytics import YOLO
import cv2

def main():
    # 1. Load a lightweight pre-trained YOLOv8 nano model
    model = YOLO("yolov8n.pt")

    # 2. Train the model on your virtual dataset
    print("Starting training on virtual dataset...")
    model.train(
        data="dataset.yaml",
        epochs=50,          
        imgsz=640,
        batch=8
    )

    print("Training finished! Running validation/inference...")

    # 3. Test the trained model on a validation image
    trained_model_path = "runs/detect/train-4/weights/best.pt"
    
    if os.path.exists(trained_model_path):
        optimized_model = YOLO(trained_model_path)
        
        test_img = "fish_dataset/images/val/fish_normal_0.jpg"
        if os.path.exists(test_img):
            results = optimized_model(test_img)
            
            # Visualize and save results in a popup window
            res_plotted = results[0].plot()
            cv2.imshow("Virtual Fish AI Detection", res_plotted)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("Training weights not found. Check training logs above.")

if __name__ == "__main__":
    main()