'''
CIS Username: tgkh12
Submodule: Image Processing
'''

##############################
import cv2 as cv
import numpy as np
import math
import random
from argparse import ArgumentParser
##############################

### Problem 1: Light Leak and Rainbow Light Leak ###

def gamma_correct(coeff):
    return np.array([((i / 255.0) ** coeff) * 255 for i in np.arange(0, 256)]).astype("uint8")

def problem1(inputImg, darkCoeff=2.5, blendCoeff=1, option=1):
    imgCopy = np.copy(inputImg)
    light_mask = cv.imread('./mask2.jpg', cv.IMREAD_COLOR)
    gamma_correction = gamma_correct(darkCoeff) # creates a gamma correction lookup table
    mask_correction = gamma_correct(15)
    mask = cv.LUT(light_mask, mask_correction)
    outputImg = cv.LUT(imgCopy, gamma_correction)
    a = cv.addWeighted(outputImg, 1, mask, 0.2, 5) ## can't use this function, have to implement on my own 

    return a

#####################################################

### Problem 2: Pencil/Charcoal Effect ###
# https://medium.com/@akumar5/computer-vision-gaussian-filter-from-scratch-b485837b6e09 <== Gaussian Blur Example
# https://www.askaswiss.com/2016/01/how-to-create-pencil-sketch-opencv-python.html

def add_salt_and_pepper(img, prob=0.5):
    output = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    rows, cols = output.shape
    threshold = 1 - prob

    for x in range (rows):
        for y in range (cols):
            random_val = random.random()
            if random_val > threshold:
                output[x, y] = 255
            elif random_val < prob:
                output[x, y] = 0
            else:
                output[x, y] = output[x, y]

    return output

def gaussian_noise(img):
    noise_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv.randn(noise_mask, 200, 10)
    noise_mask = apply_motion_blur(noise_mask, 2)
    return noise_mask

def random_uniform_noise(img):
    output = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv.randu(output, 0, 255)
    apply_motion_blur(output, 1)
    return output

def apply_motion_blur(img, dir):
    k_size = 100
    kernel = np.zeros((k_size, k_size))

    # horizontal
    if dir == 1:
        kernel[int((k_size - 1) / 2), :] = np.ones(k_size)
    # vertical
    elif dir == 2:
        kernel[:, int((k_size - 1) / 2)] = np.ones(k_size)

    kernel /= k_size
    motion_blur = cv.filter2D(img.astype(np.uint8), -1, kernel)
    return motion_blur

def dodge_naive(img, noise_mask):
    w, h = img.shape[:2]
    output = np.zeros((w, h), np.uint8)

    for i in range(w):
        for j in range(h):
            if noise_mask[i][j] == 255:
                output[i][j] = 255
            else:
                temp = (img[i][j] << 8) / (255 - noise_mask)

                if temp[i][j] > 255:
                    temp[i][j] = 255
                    output[i][j] = temp[i][j]

    return output

def problem2(inputImg, mode=1, blending=0.75):
    imgCopy = np.copy(inputImg)
    grayscale_img = cv.cvtColor(imgCopy, cv.COLOR_BGR2GRAY)
    alpha = blending
    beta = 1 - alpha
    # img_blurred = gaussianBlur(grayscale_img, 5)
    # filtered_img = cv.divide(grayscale_img, 255-img_blurred, scale=256) # efficient dodging technique

    if mode == 1:
        noise = add_salt_and_pepper(grayscale_img, 0.5)
        noise = apply_motion_blur(noise, 1)
        blended_img = np.uint8((alpha*grayscale_img) + (beta*noise))
        return blended_img

    elif mode == 2:
        noise = add_salt_and_pepper(grayscale_img, 0.5)
        noise = apply_motion_blur(noise, 1)

        red = np.uint8((alpha*grayscale_img) + (beta*gaussian_noise(grayscale_img)))
        green = np.uint8((alpha*grayscale_img) + (beta*random_uniform_noise(grayscale_img)))
        blue =  np.copy(grayscale_img)
        np.multiply(red, 0.6, out=red, casting='unsafe')
        np.multiply(green, 0.5, out=green, casting='unsafe')
        np.multiply(blue, 0, out=blue, casting='unsafe')
        blended_img = cv.merge([blue, green, red])
        return blended_img

    else:
        print('Invalid selection')


##########################################    

### Problem 3: Smoothing & Beautifying Filter ###

def median_filter(img, n):
    dim = (2*n) + 1
    margin = dim//2
    rows, cols, channels = img.shape
    output = np.copy(img)

    for x in range (margin, rows-margin):
        for y in range (margin, cols-margin):
            r_channel = []
            g_channel = []
            b_channel = []

            roi = img[x-margin:x+margin+1,y-margin:y+margin+1]

            for i in range (dim):
                r_channel += [roi[i][0][2], roi[i][1][2], roi[i][2][2]]
                g_channel += [roi[i][0][1], roi[i][1][1], roi[i][2][1]]
                b_channel += [roi[i][0][0], roi[i][1][0], roi[i][2][0]]

            r_channel.sort()
            g_channel.sort()
            b_channel.sort()

            mid = len(r_channel)//2
            output[x][y] = [b_channel[mid], g_channel[mid], r_channel[mid]]

    return output

def beautify(img):
    red_hist = np.zeros(256)
    blue_hist = np.zeros(256)
    green_hist = np.zeros(256)
    rows, cols, channels = img.shape

    for i in range(rows):
        for j in range(cols):
            #BGR
            blue_hist[img[i,j][0]] += 1
            green_hist[img[i,j][1]] += 1            
            red_hist[img[i,j][2]] += 1

    ## use univariate spline from scipy to map 

def problem3(inputImg, blur_amt=1):
    imgCopy = np.copy(inputImg)
    filtered_img = median_filter(imgCopy, blur_amt)
    beautified_img = beautify(inputImg)
    # return filtered_img

##################################################

### Problem 4: Face Swirl ###

def low_pass_filter(img):
    converted_img = np.float32(img)
    img_dft = cv.dft(converted_img, flags=cv.DFT_COMPLEX_OUTPUT)
    shift = np.fft.fftshift(img_dft) # brings low freq to the center

    r, c = img.shape
    center_r, center_c = r//2, c//2

    mask = np.zeros((r, c, 2), np.uint8)
    mask[center_r-30:center_r+30, center_c-30:center_c+30] = 1

    blend = shift*mask
    inv_shift = np.fft.ifftshift(blend)
    outputImg = cv.idft(inv_shift)
    outputImg = cv.magnitude(outputImg[:,:,0], outputImg[:,:,1])
    return outputImg

def warp_img(filtered_img, swirl_strength, swirl_radius):
    outputImg = np.copy(filtered_img)
    r, c = filtered_img.shape[0], filtered_img.shape[1]
    center_r, center_c = r//2 , c//2 

    for i in range(r):
        for j in range(c):
            relative_Y = i - center_r
            relative_X = j - center_c
            pixel_angle = 0

            if relative_X != 0:
                pixel_angle = math.atan(abs(relative_Y)/abs(relative_X))

                if relative_X > 0 and relative_Y < 0:
                    pixel_angle = 2*math.pi - pixel_angle
                elif relative_X <= 0 and relative_Y >= 0:
                    pixel_angle = math.pi - pixel_angle
                elif relative_X <= 0 and relative_Y < 0:
                    pixel_angle += math.pi
            else:
                if relative_Y >= 0:
                    pixel_angle = 2*math.pi
                else:
                    pixel_angle = 1.5*math.pi

            dist_from_center = math.sqrt((relative_X**2) + (relative_Y**2))
            swirl_amt = 1 - (dist_from_center / swirl_radius)

            ## checks if swirl transformation needs to be applied
            if swirl_amt > 0:
                twist = swirl_strength * swirl_amt * math.pi * 0.5
                pixel_angle += twist

                ori_X = int(math.floor(dist_from_center * math.cos(pixel_angle) + 0.5))
                ori_Y = int(math.floor(dist_from_center * math.sin(pixel_angle) + 0.5))
                ori_X += center_c
                ori_Y += center_r

                if ori_X < 0:
                    ori_X = 0
                elif ori_X >= c:
                    ori_X = c - 1

                if ori_Y < 0:
                    ori_Y = 0
                elif ori_Y >= c:
                    ori_Y = r - 1

                ## Bilinear Interpolation to restore image
                q12x = ori_X 
                q12y = ori_Y
                q22x = int(math.ceil(ori_X))
                q22y = q12y
                q22x = min(c-1, q22x)
                q22x = max(0, q22x)
                q11x = q12x
                q11y = int(math.ceil(ori_Y))
                q11y = min(r-1, q11y)
                q11y = max(0, q11y)
                q21x = q22x
                q21y = q11y

                top_left = filtered_img[q11y][q11x]
                bottom_left = filtered_img[q12y][q12x]
                top_right = filtered_img[q21y][q21x]
                bottom_right = filtered_img[q22y][q22x]

                b1, g1, r1 = top_left ## q11
                b2, g2, r2 = bottom_left ## q12
                b3, g3, r3 = top_right ## q21
                b4, g4, r4 = bottom_right ## q22

                if q21x == q11x:
                    factor1 = 1
                    factor2 = 0
                else:
                    factor1 = (q21x - ori_X)/(q21x - q11x)
                    factor2 = (ori_X - q11x)/(q21x - q11x)

                R1_red = factor1 * r1 + factor2 * r3
                R1_green = factor1 * g1 + factor2 * g3
                R1_blue = factor1 * b1 + factor2 * b3
                R2_red = factor1 * r2 + factor2 * r4
                R2_green = factor1 * g2 + factor2 * g4
                R2_blue = factor1 * b2 + factor2 * b4

                if q12y == q11y:
                    factor3 = 1
                    factor4 = 0
                else:
                    factor3 = (q12y - ori_Y)/(q12y - q11y)
                    factor4 = (ori_Y - q11y)/(q12y - q11y)

                P_red = factor3 * R1_red + factor4 * R2_red
                P_green = factor3 * R1_green + factor4 * R2_green
                P_blue = factor3 * R1_blue + factor4 * R2_blue

                P_red = min(255, P_red)
                P_red = max(0, P_red)
                P_blue = min(255, P_blue)
                P_blue = max(0, P_blue)
                P_green = min(255, P_green)
                P_green = max(0, P_green)

                final_pixel = [P_blue, P_green, P_red]
                outputImg[i][j] = final_pixel

    return outputImg

def problem4(inputImg, swirl_strength=1.5, swirl_radius=140):
    imgCopy = np.copy(inputImg)

    ## Pre-filtering
    b, g, r = cv.split(imgCopy)
    # b, g, r = low_pass_filter(b), low_pass_filter(g), low_pass_filter(r)
    filtered_img = cv.merge([b, g, r])

    ## Swirl image 
    face_swirl_img = warp_img(filtered_img, swirl_strength, swirl_radius)

    ## Inverse swirl


    ## Find the difference
    # img_diff = face_swirl_inv - face_swirl_img

    return face_swirl_img

#############################

def parse_arguments():
    parser = ArgumentParser(description='Allows for selection of different filters available for a provided image.')

    parser.add_argument('-i', '--in_img',
                        type=str,
                        help='Provide image (path to file) to be processed/filtered'
                        )
    parser.add_argument('-f', '--filter',
                        type=int,
                        choices=[1, 2, 3, 4],
                        default=0,
                        help='Filter selection'
                        )
    parser.add_argument('-p', '--params',
                        metavar='Parameters',
                        type=float,
                        nargs='+',
                        help='Parameters for chosen filter function'
                        )

    return parser.parse_args()

def main():
    args = parse_arguments()
    windowName = 'Processed Image'
    inputImg = cv.imread(args.in_img, cv.IMREAD_COLOR)

    if not inputImg is None:
        cv.namedWindow(windowName)
        flag = True
        params = args.params
        if not params:
            params = []

        if args.filter == 1:
            if len(params) < 3:
                print('\nInsufficient parameters provided ... Using defaults')
                outputImg = problem1(inputImg)
            else:
                outputImg = problem1(inputImg, params[0], params[1], params[2])
        elif args.filter == 2:
            if len(params) < 2:
                print('\nInsufficient parameters provided ... Using defaults')
                outputImg = problem2(inputImg)
            else:
                outputImg = problem2(inputImg, params[0], params[1])
        elif args.filter == 3:
            if len(params) < 1:
                print('\nInsufficient parameters provided ... Using defaults')
                outputImg = problem3(inputImg)
            else:
                outputImg = problem3(inputImg, params[0])            
        elif args.filter == 4:
            if len(params) < 2:
                print('\nInsufficient parameters provided ... Using defaults')
                outputImg = problem4(inputImg)
            else:
                outputImg = problem4(inputImg, params[0], params[1])
        else:
            print('\nInvalid selection')

        while flag:
            # cv.imshow('Input Image', inputImg)
            cv.imshow(windowName, outputImg)
            key = cv.waitKey(40) & 0xFF;

            if (key == ord('x')):
                flag = False;

    else:
        print('\nImage cannot be loaded')

    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
    