from multiprocessing import Pool
from PIL import Image
import numpy as np
import glob, os
import png

#For use with MT, will process the list of frames given
def process_frames(filelist):
    
    os.chdir(output_folder)
    png_files_1, png_files_2 = filelist
    
    for i in range(0,len(png_files_1)):
        
        fin = open(os.path.join(CWD, "video_1_frames", png_files_1[i]), "rb")
        print("Processing", png_files_1[i])

        #Load and convert the first video to fourier space
        #Open cover image and get dimensions
        img_base = Image.open(fin).convert('RGB')
        column_count, row_count = img_base.size

        #Load each RGB channel from the image separately as greyscale
        img_r, img_g, img_b = img_base.split()

        #Perform 2d fft on each channel
        #Red
        red_f = np.fft.fft2(img_r)
        r_magnitude = np.abs(red_f)

        #Green
        green_f = np.fft.fft2(img_g)
        g_magnitude = np.abs(green_f)

        #Blue
        blue_f = np.fft.fft2(img_b)
        b_magnitude = np.abs(blue_f)

        fin.close()

        #############
        
        #Do the same for the second video but get its phase instead
        fin = open(os.path.join(CWD, "video_2_frames", png_files_2[i]), "rb")

        #Open cover image and get dimensions
        img_base_2 = Image.open(fin).convert('RGB')
        column_count, row_count = img_base.size

        #Load each RGB channel from the image separately as greyscale
        img_r_2, img_g_2, img_b_2 = img_base_2.split()

        #Perform 2d fft on each channel
        #Red
        red_f_2 = np.fft.fft2(img_r_2)
        r_phase_2 = np.angle(red_f_2)

        #Green
        green_f_2 = np.fft.fft2(img_g_2)
        g_phase_2 = np.angle(green_f_2)

        #Blue
        blue_f_2 = np.fft.fft2(img_b_2)
        b_phase_2 = np.angle(blue_f_2)

        fin.close()

        
        #Now combine both the magnitude and phase into an array of complex numbers for each channel
        red_shift = (r_magnitude*np.exp(1j*r_phase_2)).reshape(row_count,column_count)
        green_shift = (g_magnitude*np.exp(1j*g_phase_2)).reshape(row_count,column_count)
        blue_shift = (b_magnitude*np.exp(1j*b_phase_2)).reshape(row_count,column_count)

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
        img_rgb = img_rgb.reshape(row_count,column_count*3)

        #Write the combined RGB channels to the final payload image
        with open(png_files_1[i], "wb") as out:
            pngWriter = png.Writer(
                column_count, row_count, greyscale=False, alpha=False, bitdepth=8
            )
            pngWriter.write(out, img_rgb)


#Directory boilerplate
CWD = os.getcwd()
frames_folder_1 = os.path.join(CWD, "video_1_frames")
frames_folder_2 = os.path.join(CWD, "video_2_frames")
output_folder = os.path.join(CWD, "output_frames")
THREAD_COUNT = 3    #Can increase to speed up

#Only run this section if it's the main script running, not a sub-process from multiprocessing
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
        
