from multiprocessing import Pool
from PIL import Image
import numpy as np
import glob, os
import png  #To read/write 16-bit sample images

#Process the list of frames given. Used with threading
def process_frames(filelist):
    png_files_1, png_files_2 = filelist
    
    #Go through each frame
    for i in range(0,len(png_files_1)):
        print("Processing", png_files_1[i])

        #First we need to get the phase from video 2
        fin = open(os.path.join(CWD, "video_2_frames", png_files_2[i]), "rb")
        
        #Open cover image and get dimensions
        img_base_2 = Image.open(fin).convert('RGB')
        column_count_video_2, row_count_video_2 = img_base_2.size

        #Load each RGB channel from the image separately as greyscale
        img_r_2, img_g_2, img_b_2 = img_base_2.split()

        fin.close()

        #Perform Fourier transform on each channel
        #Red
        red_f_2 = np.fft.fft2(img_r_2)  #Do 2d FFT on image samples
        r_phase_2 = np.angle(red_f_2)    #Phase information

        #Green
        green_f_2 = np.fft.fft2(img_g_2)
        g_phase_2 = np.angle(green_f_2)

        #Blue
        blue_f_2 = np.fft.fft2(img_b_2)
        b_phase_2 = np.angle(blue_f_2)


        #Now we craft the magnitude from video 1
        fin = open(os.path.join(CWD, "video_1_frames", png_files_1[i]), "rb")
        
        #Open cover image and get dimensions
        img_base = Image.open(fin).convert('RGB')
        column_count, row_count = img_base.size #i.e. 1920, 1080 etc

        #Load each RGB channel from the image separately as greyscale
        img_r, img_g, img_b = img_base.split()

        fin.close()

        #Load as numpy arrays and scale them
        #Try and set the division const so we don't saturate on the resultant real image
        #Decreasing it makes the distrubtion "wider" giving brighter pixels that while making the phase video
        #Look better, make the magnitude space less defined and introduce artifacts
        DIVISION_CONST = 3500
        img_r = np.array(img_r, dtype=np.uint32)* int(column_count_video_2*row_count_video_2/(DIVISION_CONST))
        img_g = np.array(img_g, dtype=np.uint32)* int(column_count_video_2*row_count_video_2/(DIVISION_CONST))
        img_b = np.array(img_b, dtype=np.uint32)* int(column_count_video_2*row_count_video_2/(DIVISION_CONST))

        #Flip the image horizontally and vertically to add it in the bottom right quadrant later
        img_r_flipped = np.flipud(np.fliplr(img_r))
        img_g_flipped = np.flipud(np.fliplr(img_g))
        img_b_flipped = np.flipud(np.fliplr(img_b))

        
        #DC Offsets and grid for later. The DC offset is the sum of all pixel values in the real image so
        #increasing the middle value increases the overall brightness of the image.May need to change DC
        #location in grid for odd video 2 dimensions.
        
        #We want the DC value set so the distribution of brightness in the middle image peaks at 128 so frequencies
        #up and down have the maximum room to increase/decrease the value without being clipped
        DC_magnitudes = np.array([[0,0,0,0,0 ,0,0,0,0],
                                 [0,0,0,0,0 ,0,0,0,0],
                                 [0,0,0,0,0 ,0,0,0,0],
                                 [0,0,0,0,0 ,0,0,0,0],
                                 [0,0,0,0,130,0,0,0,0],
                                 [0,0,0,0,0 ,0,0,0,0],
                                 [0,0,0,0,0 ,0,0,0,0],
                                 [0,0,0,0,0 ,0,0,0,0],
                                 [0,0,0,0,0 ,0,0,0,0]])*(column_count_video_2*row_count_video_2)

        #Get cords of where to put array above such that the DC offset will be in the middle
        middle_start_x = (column_count_video_2//2) - 4  #i.e. 4 is half of the size of array above
        middle_start_y = (row_count_video_2//2) - 4
        middle_end_x = (column_count_video_2//2) + 5
        middle_end_y = (row_count_video_2//2) + 5

        #Make our canvas for the magnitude and copy the frame into the top left and bottom right
        #Red
        r_magnitude_shift = np.zeros((row_count_video_2, column_count_video_2))
        r_magnitude_shift[0:row_count, 0:column_count] = img_r
        r_magnitude_shift[(row_count_video_2-row_count):row_count_video_2, (column_count_video_2-column_count):column_count_video_2] = img_r_flipped

        #Now copy the DC offset to the middle
        r_magnitude_shift[middle_start_y:middle_end_y, middle_start_x:middle_end_x] = DC_magnitudes

        #Green
        g_magnitude_shift = np.zeros((row_count_video_2, column_count_video_2))
        g_magnitude_shift[0:row_count, 0:column_count] = img_g
        g_magnitude_shift[(row_count_video_2-row_count):row_count_video_2, (column_count_video_2-column_count):column_count_video_2] = img_g_flipped

        g_magnitude_shift[middle_start_y:middle_end_y, middle_start_x:middle_end_x] = DC_magnitudes

        #Blue
        b_magnitude_shift = np.zeros((row_count_video_2, column_count_video_2))
        b_magnitude_shift[0:row_count, 0:column_count] = img_b
        b_magnitude_shift[(row_count_video_2-row_count):row_count_video_2, (column_count_video_2-column_count):column_count_video_2] = img_b_flipped

        b_magnitude_shift[middle_start_y:middle_end_y, middle_start_x:middle_end_x] = DC_magnitudes

        #Now shift these back to the expected edge fourier numpy format for ifft later i.e. low freq @ corners instead of middle
        r_magnitude = np.fft.ifftshift(r_magnitude_shift)
        g_magnitude = np.fft.ifftshift(r_magnitude_shift)    
        b_magnitude = np.fft.ifftshift(r_magnitude_shift)



        ####Now we can combine the magnitude and phase and do the inverse FFT
        #Combine these into an array of complex numbers for each channel
        red_shift = (r_magnitude*np.exp(1j*r_phase_2)).reshape(row_count_video_2,column_count_video_2)
        green_shift = (g_magnitude*np.exp(1j*g_phase_2)).reshape(row_count_video_2,column_count_video_2)
        blue_shift = (b_magnitude*np.exp(1j*b_phase_2)).reshape(row_count_video_2,column_count_video_2)

        #Do the inverse 2D FFT to get the image pixels back
        img_r_array = np.fft.ifft2(red_shift)
        img_g_array = np.fft.ifft2(green_shift)
        img_b_array = np.fft.ifft2(blue_shift)

        #We need to manually round each pixel and limit it's range to 0-255 here before casting to uint8
        #Otherwise due to quantization errors it could be outside the range
        img_r_array = np.uint8(np.clip(np.round(np.abs(img_r_array.real)),0,255))
        img_g_array = np.uint8(np.clip(np.round(np.abs(img_g_array.real)),0,255))
        img_b_array = np.uint8(np.clip(np.round(np.abs(img_b_array.real)),0,255))

        #Merge the seperate RGB channels together for the final image in the format RGBRGBRGB#
        img_rgb = np.dstack(( img_r_array, img_g_array,img_b_array )).ravel()
        img_rgb = img_rgb.reshape(row_count_video_2,column_count_video_2*3)

        #Write the combined RGB channels to the final payload image
        os.chdir(output_folder) #For writing there later
        with open(png_files_1[i], "wb") as out:
            pngWriter = png.Writer(
                column_count_video_2, row_count_video_2, greyscale=False, alpha=False, bitdepth=8
            )
            pngWriter.write(out, img_rgb)



        #We also want to actually look at the resultant magnitude space from this image
        fin = open(png_files_1[i], "rb")    #The image to read

        #Convert to fourier space
        img_base = Image.open(fin)
        column_count, row_count = img_base.size

        #Load each RGB channel from the image as greyscale
        img_r, img_g, img_b = img_base.split()
        fin.close()
        
        #Perform fourier transform on each channel
        #Red
        red_f = np.fft.fft2(img_r)
        red_fshift = np.fft.fftshift(red_f) #Shift low freq to center for visualisation
        red_magnitude = np.abs(red_fshift)


        #Green
        green_f = np.fft.fft2(img_g)
        green_fshift = np.fft.fftshift(green_f)
        green_magnitude = np.abs(green_fshift)


        #Blue
        blue_f = np.fft.fft2(img_b)
        blue_fshift = np.fft.fftshift(blue_f)
        blue_magnitude = np.abs(blue_fshift)


        #Scale each mag/phase array to the 8 bit image range
        #For mag scale by N image pixels
        r_magnitude = red_magnitude/(column_count*row_count)
        g_magnitude = green_magnitude/(column_count*row_count)
        b_magnitude = blue_magnitude/(column_count*row_count)


        #Convert 0-1 values above to 0-255 for 8 bit image
        r_magnitude = np.uint8(255*r_magnitude)
        g_magnitude = np.uint8(255*g_magnitude)
        b_magnitude = np.uint8(255*b_magnitude)


        #Transpose to fix orientation of resulting image
        r_magnitude = r_magnitude.T
        g_magnitude = g_magnitude.T
        b_magnitude = b_magnitude.T

        #Combine the seperate R,G,B channels to form final images
        mag_img_array = np.dstack( (r_magnitude, g_magnitude, b_magnitude) )
        mag_img_array = np.hstack(mag_img_array)

        #Scale the magnitude image so we can actually see it
        mag_img_array = np.uint8(np.clip((np.round((mag_img_array+1))*10),0,255))


        #Write the magnitude
        os.chdir(output_recovered_folder)
        mag_png_file_output = open(png_files_1[i], "wb")
        pngWriter = png.Writer(
            column_count, row_count, greyscale=False, alpha=False, bitdepth=8
        )
        pngWriter.write(mag_png_file_output, mag_img_array)

        mag_png_file_output.close()
        
    
CWD = os.getcwd()
frames_folder_1 = os.path.join(CWD, "video_1_frames")
frames_folder_2 = os.path.join(CWD, "video_2_frames")
output_folder = os.path.join(CWD, "output_frames")
output_recovered_folder = os.path.join(CWD, "output_recovered_frames")
THREAD_COUNT = 4    #Can increase to speed up

#Only run this section if it's the main script running, not a sub-process from multiprocessing
"""
Note video_1 should be about 0.9 * 1/4 of the size of video_2 and the same aspect ratio
"""
if __name__ == '__main__':

    #Get a list of all frames from both folders
    png_files_1 = []
    png_files_2 = []
    
    os.chdir(frames_folder_1)
    for file in glob.glob("*.png"):
        png_files_1.append(file)

    os.chdir(frames_folder_2)
    for file in glob.glob("*.png"):
        png_files_2.append(file)
        
    #Sort them numerically
    png_files_1.sort(key=lambda x: int(x.split('.')[0]))
    png_files_2.sort(key=lambda x: int(x.split('.')[0]))

    #Split them into THREAD_COUNT # of chunks
    png_1_chunks = np.array_split(png_files_1, THREAD_COUNT)
    png_2_chunks = np.array_split(png_files_2, THREAD_COUNT)

    #Combine those into a list of [[chunks1, chunks2], [chunks1, chunks2]] etc to pass in a single argument
    chunk_list = []
    for i in range(0, len(png_1_chunks)):
        chunk_list.append([png_1_chunks[i],png_2_chunks[i]])

    #Change CWD back for threads
    os.chdir(CWD) #For writing there later
    with Pool(THREAD_COUNT) as p:
        p.map(process_frames, chunk_list)


