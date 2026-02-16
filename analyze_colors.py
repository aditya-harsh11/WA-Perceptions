
import matplotlib.pyplot as plt
import numpy as np

def analyze_image():
    img = plt.imread('rgb/left000000.png')
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title("Frame 0 Analysis")
    plt.axis('off')
    plt.savefig('frame0_analysis.png')
    print("Saved frame0_analysis.png")

if __name__ == "__main__":
    analyze_image()
