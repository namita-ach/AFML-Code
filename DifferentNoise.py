import numpy as np
import cv2
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0, sigma=25):
    gauss = np.random.normal(mean, sigma, image.shape).astype('uint8')
    noisy_image = cv2.add(image, gauss)
    return noisy_image

def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy_image = image.copy()
    total_pixels = image.size
    
    # Salt noise (set pixels to 255)
    salt_pixels = int(total_pixels * salt_prob)
    salt_coords = [np.random.randint(0, i-1, salt_pixels) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1], :] = 255
    
    # Pepper noise (set pixels to 0)
    pepper_pixels = int(total_pixels * pepper_prob)
    pepper_coords = [np.random.randint(0, i-1, pepper_pixels) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1], :] = 0
    
    return noisy_image

def add_masking(image, mask_prob=0.1):
    masked_image = image.copy()
    mask_size = int(mask_prob * image.size)
    mask_coords = [np.random.randint(0, i-1, mask_size) for i in image.shape]
    masked_image[mask_coords[0], mask_coords[1], :] = 0
    return masked_image

def add_dropout(image, dropout_prob=0.1):
    dropout_image = image.copy()
    dropout_mask = np.random.rand(*image.shape[:2]) < dropout_prob
    dropout_image[dropout_mask] = 0
    return dropout_image

def show_noisy_images(image, gaussian_params, salt_pepper_params, mask_params, dropout_params):
    gaussian_noisy = add_gaussian_noise(image, **gaussian_params)
    salt_pepper_noisy = add_salt_and_pepper_noise(image, **salt_pepper_params)
    masked_image = add_masking(image, **mask_params)
    dropout_image = add_dropout(image, **dropout_params)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(gaussian_noisy, cv2.COLOR_BGR2RGB))
    plt.title("Gaussian Noise")
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(salt_pepper_noisy, cv2.COLOR_BGR2RGB))
    plt.title("Salt & Pepper Noise")
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    plt.title("Masked Image")
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(dropout_image, cv2.COLOR_BGR2RGB))
    plt.title("Dropout Noise")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = '/content/rick.webp'
    image = cv2.imread(image_path)
    
    gaussian_params = {'mean': 0, 'sigma': 25}
    salt_pepper_params = {'salt_prob': 0.01, 'pepper_prob': 0.01}
    mask_params = {'mask_prob': 0.1}
    dropout_params = {'dropout_prob': 0.1}
    # change these and try
    
    show_noisy_images(image, gaussian_params, salt_pepper_params, mask_params, dropout_params)
