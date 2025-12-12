import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Img url (Nowy link do zdjęcia samochodu - Ferrari 488 GTB)
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Ferrari_488_GTB_by_Alex_Penfold.jpg/1280px-Ferrari_488_GTB_by_Alex_Penfold.jpg"
# Imitate browser-like behavior to avoid error 403: Forbidden
header = {'User-Agent': 'Mozilla/5.0'}

# Fn to check img quality
def check_quality(img):
    # Get histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    p = hist / np.sum(hist)

    # Calc entropy using an equation
    entropy = -np.nansum(p * np.log2(p + 1e-12))

    # Calc difference between highest and lowest values
    used_values = np.where(hist > 0)[0]
    spread = used_values[-1] - used_values[0] if len(used_values) > 1 else 0

    # Calc final score
    quality_score = 0.5 * (entropy / 8.0) + 0.5 * (spread / 255.0)
    
    return entropy, spread, quality_score

def main():
    # Get the image by url and convert to used formats
    req = urllib.request.Request(url, headers=header)
    with urllib.request.urlopen(req) as response:
        image_data = response.read()
    
    arr = np.asarray(bytearray(image_data), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    
    # Konwersja formatów
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Display og photo
    plt.subplot(2, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Oryginalne zdjęcie")
    plt.axis('off')

    # Display grayscale histogram
    plt.subplot(2, 2, 2)
    plt.hist(img_gray.ravel(), bins = 256)
    plt.title("Histogram szarości")

    # Display RGB histogram
    plt.subplot(2, 2, 3)
    for i, c in enumerate(('r', 'g', 'b')):
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        plt.plot(hist, color=c)
    plt.show()

    # Get and print quality values and score
    entropy, spread, quality = check_quality(img_gray)
    
    print("\nJakość zdjęcia:")
    print(f"Entropia: {entropy:.3f} / 8.0")
    print(f"Rozpiętość histogramu: {spread} / 255")
    print(f"Ocena końcowa: {quality:.3f} (0-1)\n")

main()
