import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

class ImageSeriesLoader:
    def __init__(self, root_dir, image_extensions=('png', 'jpg', 'jpeg')):
        """
        root_dir: path containing subdirectories of image series
        image_extensions: valid image file extensions
        """
        self.root_dir = root_dir
        self.image_extensions = image_extensions
        # all images are in the same location
        # each unique series begins with a unique ID before the first '-'
        self.unique_dict = {}
        self.type_dict = {'drone':[], 'bird':[]}
        #load all images in the root directory
        for fname in [file for file in os.listdir(root_dir) if not file.startswith("._")]:
            if fname.lower().endswith(image_extensions):
                series_id = fname.split('-')[0]
                if int(series_id[3:])>43:
                    type = 'drone'
                else:
                    type = 'bird'

                if series_id not in self.unique_dict:
                    self.unique_dict[series_id] = [os.path.join(root_dir, fname)]
                    self.type_dict[type].append(series_id)
                else:
                    self.unique_dict[series_id].append(os.path.join(root_dir, fname))

    def load_series(self, type='drone', series_id=None):
        """
        Load a series of images by its unique ID
        Returns list of images
        """
        if type not in self.type_dict:
            raise ValueError(f"Type {type} not recognized. Valid types: {list(self.type_dict.keys())}")

        if type:
            series_ids = self.type_dict[type]
            #select a random series_id from the list
            if series_id is None:
                series_id = np.random.choice(series_ids)

        if series_id not in self.unique_dict:
            raise ValueError(f"Series ID {series_id} not found.")


        image_paths = sorted(self.unique_dict[series_id])
        images = [np.asarray(cv2.imread(p)) for p in image_paths]
        return images


def video_to_images(video_path, dest_dir, prefix="frame"):
    """
    Converts an MP4 video into a series of images.

    video_path: path to the input video file
    dest_dir: directory where images will be saved
    prefix: prefix for saved image filenames
    """
    # Ensure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Create subdir for this video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.join(dest_dir)
    os.makedirs(video_dir, exist_ok=True)

    frame_idx = 0
    spacing=10
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Save frame as image
        img_path = os.path.join(video_dir, f"{prefix}-{frame_idx:04d}.png")

        #open image, reduce in resolution to 500x500, then save
        frame = cv2.resize(frame, (500, 500))
        cv2.imwrite(img_path, frame)
        if frame_idx>50*spacing:
            break
        frame_idx += spacing

    cap.release()
    print(f"Saved {frame_idx} frames from {video_path} to {video_dir}")


def convert_videos_in_dir(src_dir, dest_dir, video_extensions=('mp4',)):
    """
    Converts all videos in src_dir to image sequences in dest_dir
    """
    for i,fname in enumerate(os.listdir(src_dir)):
        save_prefix = 'vid{}-'.format(i+43)
        if fname.lower().endswith(video_extensions):
            video_path = os.path.join(src_dir, fname)
            video_to_images(video_path, dest_dir, prefix=save_prefix)


def compute_dog(img,sigma1=1,sigma2=3):
    import scipy.ndimage
    blurred1 = scipy.ndimage.gaussian_filter(img, sigma=sigma1)
    blurred2 = scipy.ndimage.gaussian_filter(img, sigma=sigma2)
    dog = blurred1 - blurred2
    dog = (dog - dog.min()) / (dog.max() - dog.min()) * 255
    dog = dog.astype(np.uint8)
    return dog

def rgb_to_hsv(image):
    return np.asarray(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))

def rgb_to_gray(image):
    return np.asarray(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))

def otsu_threshold(image):
    """
    Computes Otsu's threshold for a grayscale image
    """
    thresh = threshold_otsu(image)
    binary = image > thresh
    return binary.astype(np.uint8)



if __name__ == "__main__":
    import scipy.ndimage
    from skimage.filters import threshold_otsu

    plt.switch_backend('tkagg')
    root_dir = './images_subdir/'
    #
    # convert_videos_in_dir(source_videos, root_dir)

    loader = ImageSeriesLoader(root_dir)
    views = 10
    switch = True

    lo=50
    hi=200
    sigma1=1
    sigma2=10

    #show the last three images
    for _ in range(views):

        img = loader.load_series(type='bird' if switch else 'drone')[0]
        gray = rgb_to_gray(img)
        dog1 = compute_dog(gray,sigma1=sigma1,sigma2=sigma2)# an intermediate pass filter between pixel lengths of 0.5 and 7
        otsu = otsu_threshold(gray)
        dog_intermediate = np.where((dog1>lo) & (dog1<hi), 1, 0)

        #fill holes in dog_intermediate
        filled = scipy.ndimage.binary_fill_holes(dog_intermediate).astype(np.uint8)

        diff_int_otsu = np.where(otsu!=dog_intermediate,1,0)
        otsu_minus_diff = np.where(diff_int_otsu!=otsu,1,0)

        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.imshow(gray)
        ax1.set_title("Original Image")
        ax1.axis('off')
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.imshow(dog1, cmap='gray')
        ax2.set_title("Band-Pass filter (DoG)")
        ax2.axis('off')
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.imshow(dog_intermediate, cmap='gray')
        ax3.set_title("Intermediate-thresholded BPF (itDoG)")
        ax3.axis('off')
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.imshow(otsu, cmap='gray')
        ax4.set_title("Otsu's Thresholding")
        ax4.axis('off')
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.imshow(diff_int_otsu, cmap='gray')
        ax5.set_title("Difference Intermediate and Otsu")
        ax5.axis('off')
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.imshow(filled, cmap='gray')
        ax6.set_title("Filled Holes itDOG")
        ax6.axis('off')
        plt.tight_layout()
        plt.show()

        switch = not switch

