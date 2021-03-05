import cv2
import numpy as np
import math
import sys
import huffman
from scipy.fftpack import dct 
np.set_printoptions(threshold=sys.maxsize)

def rgb2ycbcr(image):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = image.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)

def ycbcr2rgb(image):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = image.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)

def blockingOf8(image):
    heightDifference = image.shape[0] % 8
    widthDifference = image.shape[1] % 8
    if widthDifference != 0 or heightDifference != 0:
        top =  int(heightDifference/2)
        bottom =  heightDifference-top
        left = int(widthDifference/2)
        right = widthDifference - left
        return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REPLICATE)
    return image
        
def compressImage(image, blockSize, quantizationMatrix):

    wholeDctMatrix = np.zeros_like(image, dtype=float)
    N = blockSize
    coded = []
    currentBlock = np.zeros((N,N))
    dctMatrix = np.zeros_like(currentBlock)
    DCT_block_coef = np.zeros_like(currentBlock)

    for channel in range(0, 3):
        for i in range(0, image.shape[0], N):
            for j in range(0, image.shape[1], N):
                
                currentBlock = image[i:i+N,j:j+N, channel]

                dctMatrix = DCT(currentBlock)
                wholeDctMatrix[i:i+N,j:j+N, channel] = dctMatrix
                
                DCT_block_coef = np.matrix.round(dctMatrix @ currentBlock @ np.linalg.inv(dctMatrix))
                quantizedBlock = quantize(DCT_block_coef, quantizationMatrix)

                quantized_block_zigzag = (huffman.zigzag(quantizedBlock.astype(int)))
                coded += quantized_block_zigzag
    return huffman.encode_array(coded), wholeDctMatrix 

def reverse_DCT(quantizationMatrix, quantizedBlock):
    return np.multiply(quantizationMatrix, quantizedBlock)

def quantize(DCT_coef, quantizationMatrix):
    return np.matrix.round(np.divide(DCT_coef, quantizationMatrix))

def DCT(currentBlock):
    N = currentBlock.shape[0]
    dctMatrix = np.zeros((N,N))
    currentBlock -= 128

    for i in range(0, currentBlock.shape[0]):
        for j in range(0, currentBlock.shape[1]):
            if i == 0:
                dctMatrix[i][j] = 1 / math.sqrt(N)
            else:
                dctMatrix[i][j] = math.sqrt(2/N) * math.cos(((2*j+1)*i*math.pi) / (2*N))
    return dctMatrix

def calculateQuantizationMatrix(qualityLevel):
    Q50 = np.array([[16,11,10,16,24,40,51,61],
                    [12,12,14,19,26,58,60,55],
                    [14,13,16,24,40,57,69,56],
                    [14,17,22,29,51,87,80,62],
                    [18,22,37,56,68,109,103,77],
                    [24,35,55,64,81,104,113,92],
                    [49,64,78,87,103,121,120,101],
                    [72,92,95,98,112,100,130,99]])
    if qualityLevel < 50:
        Q = int(50/qualityLevel) * Q50
        Q [Q > 255] = 255
        return Q
    elif qualityLevel > 50:
        return (((100-qualityLevel)/50) * Q50).astype(int)
    else:
        return Q50.astype(int)
    
image = cv2.imread('lenna.bmp')

print("\nDimenzije slike:", image.shape, ", Prostor na disku:", sys.getsizeof(image)/1024, "kB.")

ycbcr_image = rgb2ycbcr(image)
ycbcr_image = blockingOf8(ycbcr_image)
compressed = np.empty_like(ycbcr_image)

quantizationMatrix = calculateQuantizationMatrix(20)
(codedImage, trim), wholeDctMatrix = compressImage(ycbcr_image, 8, quantizationMatrix)

decodedImage = huffman.decode(trim, codedImage)

k = 0
N = 8
for channel in range(0, 3):
    for i in range(0, compressed.shape[0], 8):
        for j in range(0, compressed.shape[1], 8):
            dctMatrix = wholeDctMatrix[i:i+N,j:j+N, channel]
            temp = huffman.reverse_zigzag(decodedImage[k:k+64], 8)
            R = reverse_DCT(quantizationMatrix, temp)
            compressed[i:i+N,j:j+N, channel] = np.matrix.round(np.linalg.inv(dctMatrix) @ R @ dctMatrix) + 128
            k += 64

compressed = ycbcr2rgb(compressed)

cv2.imshow('Originalna slika', image)
cv2.imshow('Kompresirana slika', compressed)
cv2.waitKey(0)
cv2.destroyAllWindows()