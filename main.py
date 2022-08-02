import facerecognition as fr
from PIL import Image
import numpy as np
def main():
    img1_path = 'tests/dataset/img1.jpg'
    img2_path = 'tests/dataset/img2.jpg'

    #image = fr.detectFace(img2_path)
    #pil_image = Image.fromarray(np.uint8(image * 255))

    #pil_image.show()

    result = fr.find(img1_path, 'tests/dataset')
    print (result)
if __name__ == "__main__":
    main()