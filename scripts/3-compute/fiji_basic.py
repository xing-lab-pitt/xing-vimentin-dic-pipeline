# @String(label="Enter a filename pattern describing the TIFFs to process") pattern
# @File(label="Select the output location", style="directory") output_dir
# @String(label="Experiment name (base name for output files)") experiment_name
# @Float(label="Flat field smoothing parameter (0 for automatic)", value=0.1) lambda_flat
# @Float(label="Dark field smoothing parameter (0 for automatic)", value=0.01) lambda_dark


# pattern = '''/Volumes/Fusion0_dat/dante_weikang_imaging_data/rpe_pcna_p21_72hr_time_lapse/TRITC_data/rpe_pcna_p21_72hr_time_lapse_T{time}_XY{tile}_{channel}.tif'''
import collections
import os
import re
import sys

import BaSiC_ as Basic
from ij import IJ, ImagePlus, ImageStack, WindowManager
from ij.io import Opener
from ij.macro import Interpreter

MAX_IMG_IN_MEM = 100
DEFAULT_CHANNEL = 0


def enumerate_filenames(pattern, time_range=(0, float('inf'),)):
    """Return filenames matching pattern (a str.format pattern containing
    {channel} and {tile} placeholders).
    Returns a list of lists, where the top level is indexed by channel number
    and the bottom level is sorted filenames for that channel.
    """
    (base, pattern) = os.path.split(pattern)
    regex = re.sub(r'{([^:}]+)(?:[^}]*)}', r'(?P<\1>.*?)', pattern)
    print("---------------------------------------")
    print('regex:', regex)
    tiles = set()
    channels = set()
    num_images = 0
    # Dict[channel: int, List[filename: str]]
    filenames = collections.defaultdict(list)
    for f in os.listdir(base):
        match = re.match(regex, f)
        if match:
            gd = match.groupdict()
            tile = str(gd['tile'])
            if 'channel' in gd:
                channel = str(gd['channel'])
            else:
                print(
                    'warning: channel not found, use default channel=',
                    DEFAULT_CHANNEL)
                channel = DEFAULT_CHANNEL

            time = str(gd['time'])
            if not(time_range[0] <= int(time) <= time_range[1]):
                continue
            tiles.add(tile)
            channels.add(channel)
            filenames[channel].append(os.path.join(base, f))
            num_images += 1
    # if len(tiles) * len(channels) != num_images:
    #    raise Exception("Missing some image files")
    filenames = [
        sorted(filenames[channel])
        for channel in sorted(filenames.keys())
    ]
    return filenames


def get_filename_from_path(path, extension=False):
    '''
    returns filename from a path, without extension
    '''
    basename = os.path.basename(path)
    res = os.path.splitext(basename)  # filename, ext
    if not extension:
        res = res[0]
    return res


def make_dir(path, abort=True):
    if os.path.exists(path):
        print(path + ' : exists')
        if abort:
            exit(0)
        elif os.path.isdir(path):
            print(path + ' : is a directory, continue using the old one')
            return False
        else:
            print(path + ' : is not a directory, creating one')
            os.makedirs(path)
            return True
    else:
        os.makedirs(path)
        return True


def main():
    make_dir(str(output_dir), abort=False)
    template = '%s/%s-%%s.tif' % (output_dir, experiment_name)
    Interpreter.batchMode = True

    if (lambda_flat == 0) ^ (lambda_dark == 0):
        print("ERROR: Both of lambda_flat and lambda_dark must be zero,"
              " or both non-zero.")
        return
    lambda_estimate = "Automatic" if lambda_flat == 0 else "Manual"

    #import pdb; pdb.set_trace()
    print "Loading images..."

    # Note: filenames are file paths here
    filenames = enumerate_filenames(pattern)
    print('loaded filenames:', filenames)
    num_channels = len(filenames)
    image = Opener().openImage(filenames[0][0])
    width = image.width
    height = image.height
    image.close()

    # Pre-allocate the output profile images, since we have all the dimensions.
    ff_image = IJ.createImage("Flat-field", width, height, num_channels, 32)
    df_image = IJ.createImage("Dark-field", width, height, num_channels, 32)

    print("\n\n")

    # BaSiC works on one channel at a time, so we only read the images from one
    # channel at a time to limit memory usage.
    for channel in range(num_channels):
        num_images = len(filenames[channel])

        def handle_image_stack(stack, stack_image_num, stack_filenames):
            input_image = ImagePlus("input", stack)
            IJ.run(input_image, "Enhance Contrast...", "saturated=0.3 normalize process_all");
            
            # BaSiC seems to require the input image is actually the ImageJ
            # "current" image, otherwise it prints an error and aborts.
            WindowManager.setTempCurrentImage(input_image)
            basic.exec(
                input_image, None, None,
                "Estimate shading profiles", "Estimate both flat-field and dark-field",
                # "Automatic", 0.5, 0.5, # default
                lambda_estimate, lambda_flat, lambda_dark,
                "Ignore", "Compute shading and correct images"
            )
            # IJ.saveAsTiff(input_image, template % ('input'))
            input_image.close()
            # Copy the pixels from the BaSiC-generated profile images to the
            # corresponding channel of our output images.
            ff_channel = WindowManager.getImage(
                "Flat-field:%s" % input_image.title)
            ff_image.slice = channel + 1
            ff_image.getProcessor().insert(ff_channel.getProcessor(), 0, 0)

            df_channel = WindowManager.getImage(
                "Dark-field:%s" % input_image.title)
            df_image.slice = channel + 1
            df_image.getProcessor().insert(df_channel.getProcessor(), 0, 0)

            tiffsPath = '%s/%s/' % (output_dir,
                                    ('%s_channel' % (experiment_name)) + str(channel))
            make_dir(tiffsPath, abort=False)

            corrected_channel = WindowManager.getImage(
                "Corrected:%s" % input_image.title)
            # IJ.saveAsTiff(corrected_channel, template % ('corrected-tttt'))
            for j in range(stack_image_num):
                IJ.run(
                    corrected_channel,
                    "Duplicate...",
                    "title=temp_stack duplicate range=%s-%s" %
                    (j + 1, j + 1))  # note 1 base in Fiji
                temp_stack = WindowManager.getImage("temp_stack")
                temp_path = os.path.join(
                    tiffsPath, get_filename_from_path(
                        stack_filenames[j], extension=False))
                print('temp_path:', temp_path)
                IJ.saveAs(temp_stack, "Tiff", temp_path)
                temp_stack.close()
            # IJ.run(corrected_channel, "Image Sequence... ", "format=TIFF digits=3 save=%s" % ('%s' % (tiffsPath)));
            ff_channel.close()
            df_channel.close()
            corrected_channel.close()
            print("\n\n")
        print "Processing channel %d/%d..." % (channel + 1, num_channels)
        print "%d images in current channel" % (num_images)
        print "==========================="
        num_stacks = (num_images // MAX_IMG_IN_MEM) + 1

        for stack_id in range(num_stacks):
            opener = Opener()
            start_filename_index = stack_id * MAX_IMG_IN_MEM
            end_filename_index = None
            if stack_id == num_stacks - 1:
                end_filename_index = len(filenames[channel]) - 1
            else:
                end_filename_index = (stack_id + 1) * MAX_IMG_IN_MEM - 1
            stack_image_num = end_filename_index - start_filename_index + 1
            stack = ImageStack(width, height,
                               stack_image_num)
            # The internal initialization of the BaSiC code fails when we invoke it via
            # scripting, unless we explicitly set a the private 'noOfSlices' field.
            # Since it's private, we need to use Java reflection to access it.
            Basic_noOfSlices = Basic.getDeclaredField('noOfSlices')
            Basic_noOfSlices.setAccessible(True)
            basic = Basic()
            Basic_noOfSlices.setInt(basic, stack_image_num)
            stack_filenames = filenames[channel][start_filename_index:end_filename_index + 1]
            for i, filename in enumerate(stack_filenames):
                print "Loading image %d/%d: %s" % (i + 1, stack_image_num, filename)
                image = opener.openImage(filename)
                stack.setProcessor(image.getProcessor(), i + 1)
            handle_image_stack(stack, stack_image_num, stack_filenames)
            del stack

    ff_filename = template % 'ffp'
    IJ.saveAsTiff(ff_image, ff_filename)
    ff_image.close()
    df_filename = template % 'dfp'
    IJ.saveAsTiff(df_image, df_filename)
    df_image.close()
    WindowManager.setTempCurrentImage(None)
    print "Done!"


main()
