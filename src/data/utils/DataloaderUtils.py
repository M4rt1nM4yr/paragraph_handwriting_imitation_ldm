
import torch
import numpy as np

def single_src_pad_mask(size, modeParagraph, z_shape, image):
    mask = torch.zeros(z_shape, dtype=torch.bool)

    # TODO adjust this later. It might already work with below implementation
    if modeParagraph > 0:
        return mask

    picture = image
    aspect_ratio_target = size[0] / size[1]
    aspect_ratio_input = picture.height / picture.width
    aspect_rev = int(size[1] / size[0])

    if aspect_ratio_input < aspect_ratio_target:
        # pad height

        final_width = picture.width
        for i in range(aspect_rev):
            current_width = final_width + i
            if current_width % aspect_rev == 0:
                final_width = current_width
                break

        final_height = int(aspect_ratio_target * final_width)

        final_height = int(size[0]*picture.width/size[1])
    else:

        final_height = picture.height
      #  final_width = final_height * aspect_rev
   #     final_width = int(size[0]/size[1] * picture.width/picture.height *picture.width)
        final_width = int(size[1]*picture.height/size[0])

    ratio_pad_height = picture.height / final_height
    ratio_pad_width = picture.width / final_width
    #TODO +1
    padding_start_height = min(int(ratio_pad_height * z_shape[0]+0.9999) , z_shape[0])
    padding_start_width = min(int(ratio_pad_width * z_shape[1]+0.9999) , z_shape[1])

    mask[:, padding_start_width:] = True
    mask[padding_start_height:, :] = True
    return mask


"""
        Creates the src_pad_mask.

"""


def src_pad_mask(size, modeParagraph, z_shape,sample_names,images):
    src_key_padding = dict()

    # Only relevant for pre-training. Just gonna leave this empty since this will barely contain any padding
    if modeParagraph > 0:
        for s in sample_names:
            mask = torch.zeros(z_shape, dtype=torch.bool)
            src_key_padding[s] = mask

    # Paragraphs. Important bit
    # Based on the aspect ratio of the input image and the target aspect ratio we compute the padding mask
    # TODO check whether this works for 512x1024
    if modeParagraph == 0:
        for s in sample_names:
            picture = images[s]
            mask = torch.zeros(z_shape, dtype=torch.bool)

            aspect_ratio_target = size[0] / size[1]
            aspect_ratio_input = picture.height / picture.width
            aspect_rev = int(size[1] / size[0])

            if aspect_ratio_input < aspect_ratio_target:
                # pad height

                final_width = picture.width
                for i in range(aspect_rev):
                    curren_width = final_width + i
                    if curren_width % aspect_rev == 0:
                        final_width = curren_width
                        break

                final_height = int(aspect_ratio_target * final_width)
            else:

                final_height = picture.height
                final_width = final_height * aspect_rev

            ratio_pad_height = picture.height / final_height
            ratio_pad_width = picture.width / final_width
            padding_start_height = min(int(ratio_pad_height * z_shape[0]) + 1, z_shape[0])
            padding_start_width = min(int(ratio_pad_width * z_shape[1]) + 1, z_shape[1])

            mask[:, padding_start_width:] = True
            mask[padding_start_height:, :] = True

            src_key_padding[s] = mask

    return src_key_padding


def mk_padding_interpolation(x,z_shape, INTERPOLATION_CUTOFF = 0.99,invert=False):
    # interpolation time
    padding = torch.zeros(z_shape, dtype=torch.bool)


    avg_horizontal = np.mean(x,axis=1)/255#torch.mean(x[0], dim=1)
    avg_vertical = np.mean(x, axis=0) / 255

    padding_start_horizontal = x.shape[0]
    padding_start_vertical = x.shape[1]

    #Check where the text starts for the height
    for j in range(x.shape[0] - 1, 0, -1):
        if avg_horizontal[j] < INTERPOLATION_CUTOFF:
            padding_start_horizontal = j
            break

    #Compute padding start for height
    padding_start_horizontal = min(padding_start_horizontal + 5, x.shape[0])
    ratio_pad_height = padding_start_horizontal / x.shape[0]
    padding_start_height = min(int(ratio_pad_height * z_shape[0]) + 1,z_shape[0])


    #Check where the text starts for the width
    for j in range(x.shape[1] - 1, 0, -1):
        if avg_vertical[j] < INTERPOLATION_CUTOFF:
            padding_start_vertical = j
            break

    #Compute the padding start for width
    padding_start_vertical = min(padding_start_vertical + 5, x.shape[1])
    ratio_pad_width = padding_start_vertical / x.shape[1]
    # Extra +1 here. So it doesn't accidentally delete single dots at the end of lines. This is very generous and probably not needed
    padding_start_width= min(int(ratio_pad_width * z_shape[1]) + 1 +1,z_shape[1])

    padding[padding_start_height:, :] = True
    padding[:,padding_start_width:] = True

    return padding
