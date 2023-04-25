"""
@DATE: 2022/10/24
@Author  : ld
"""
import cv2
import os


def main():

    print(os.getenv("CUDA_SDK_ROOT_DIR"))
    # for key in cv2.cuda.__dict__:
    #     print(key)
    cv2.cuda.getDevice()



if __name__ == "__main__":
    pass
    main()
