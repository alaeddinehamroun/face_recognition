import facerecognition as fr
def main():
    img1_path = 'tests/dataset/img1.jpg'
    img2_path = 'tests/dataset/img2.jpg'

    print(fr.verify(img1_path=img1_path,  img2_path=img2_path))

if __name__ == "__main__":
    main()