import glob
import math
import os
import random
import secrets

import cv2
import markovgen
import numpy as np
from PIL import Image
from tqdm import tqdm
from trdg.generators import GeneratorFromStrings

import config


def generate_random_text(textfile, n):
    """Generate long text samples, remove quote characters from them."""
    text_donor = open(textfile)
    markov = markovgen.Markov(text_donor)
    markov_text = list(set([markov.generate_markov_text(max_size=1000, backward=True).replace('"', '').replace("'", '')
                            for _ in tqdm(range(n))]))
    return markov_text


def get_pictures(root_dir):
    """Paths to all images in the folder."""
    paths_to_images = glob.glob(root_dir + '/**/*.jpg', recursive=True)
    return paths_to_images


def yolo2tlrb(x, y, w, h, width, height):
    """Conversion from yolo format to [top, left, right, bottom] format."""
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return round(x1*width), round(y1*height), round(x2*width), round(y2*height)


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """Resize image to given height/width."""
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def get_random_color(max_value=255):
    """Random color generation."""
    r = lambda: random.randint(0, max_value)
    random_color = ('#%02X%02X%02X' % (r(), r(), r()))
    return random_color


def get_simple_random_color():
    """Generating a random dark color from a list of the most popular font colors."""
    r = random.sample([(0, 0, 0), (54, 69, 79), (2, 48, 32), (48, 25, 52), (52, 52, 52),
                       (27, 18, 18), (40, 40, 43), (25, 25, 112), (53, 57, 53)], 1)[0]
    random_color = ('#%02X%02X%02X' % r)
    return random_color


def first_nonzero(arr, axis, threshed=False, mask_val=0, invalid_val=-1):
    """Getting the first non-null element in the array along the selected axis."""

    # depending on the arguments, choose what to consider as the background - black or threshold color
    if not threshed:
        mask = arr != mask_val
    else:
        mask = arr > mask_val
    val = np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)
    return np.where(mask.any(axis=axis), val, invalid_val)


def last_nonzero(arr, axis, threshed=False, mask_val=0, invalid_val=-1):
    """Getting the last non-null element in the array along the selected axis."""

    # depending on the arguments, choose what to consider as the background - black or threshold color
    if not threshed:
        mask = arr != mask_val
    else:
        mask = arr > mask_val
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def add_media(background, array, df, cls, folder, media_top_n_choice=10):
    """Adding a image, chart, diagram or table to the background."""
    result = np.copy(background)
    media = array[array[:, -1] == cls]
    for b, bbox in enumerate(media):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        ratio = w/h
        square = w*h
        # we do not take into account already used crops and crops that will have to be heavily upscaled
        good_squares = df[(df['square'] > square / 4) & (~df['is_used'])]

        # check that there are crops left in the dataframe that can be selected
        if not len(good_squares):
            return result, None

        # do not take those crops whose aspect ratio does not fit the target bbox
        if w < h:
            candidates = good_squares[(good_squares['width'] < good_squares['height'])]
        else:
            candidates = good_squares[(good_squares['width'] >= good_squares['height'])]
        # find the top k crops whose aspect ratio is closest to the target bbox
        candidates = candidates.sort_values(by='ratio', key=lambda x: abs(x-ratio))
        candidates = candidates.head(media_top_n_choice)
        # if such a crop is found, draw it and mark in the dataframe that we used it
        if len(candidates):
            choice = candidates.sample(1)
            df['is_used'].iloc[choice.index[0]] = True
        else:
            return None, df

        # paste the crop into the target bbox
        load_media = cv2.imread(os.path.join(folder, choice['name'][choice.index[0]]))
        load_media = cv2.resize(load_media, (w, h))
        result[bbox[1]:bbox[3], bbox[0]:bbox[2]] = load_media

    return result, df


def add_logo_or_icon(background, array, df, cls, folder, top_n_choice=10, colour_smoothing_threshold=64):
    """Adding (with transparency) a logo or icon to the background."""
    result = np.copy(background)
    logo_or_icon = array[array[:, -1] == cls]
    for b, bbox in enumerate(logo_or_icon):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        ratio = w/h
        square = w*h
        # we do not take into account already used crops and crops that will have to be heavily upscaled
        good_squares = df[(df['square'] > square / 4) & (~df['is_used'])]

        # check that there are crops left in the dataframe that can be selected
        if not len(good_squares):
            return result, None

        # do not take those crops whose aspect ratio does not fit the target bbox
        if w < h:
            candidates = good_squares[(good_squares['width'] < good_squares['height'])]
        else:
            candidates = good_squares[(good_squares['width'] >= good_squares['height'])]
        # find the top k crops whose aspect ratio is closest to the target bbox
        candidates = candidates.sort_values(by='ratio', key=lambda x: abs(x-ratio))
        candidates = candidates.head(top_n_choice)
        # if such a crop is found, draw it and mark in the dataframe that we used it
        if len(candidates):
            choice = candidates.sample(1)
            df['is_used'].iloc[choice.index[0]] = True
        else:
            return None, df

        # paste the crop into the target bbox
        logo_or_icon = cv2.imread(os.path.join(folder, choice['name'][choice.index[0]]))
        logo_or_icon = cv2.resize(logo_or_icon, (w, h))

        # remove the black background from the crop
        roi = np.copy(result[bbox[1]:bbox[3], bbox[0]:bbox[2]])
        image2gray = cv2.cvtColor(logo_or_icon, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(image2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # insert pixels from the background of the slide instead of the pixels of the black background of the crop
        background_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        image2_fg = cv2.bitwise_and(logo_or_icon, logo_or_icon, mask=mask)
        dst = cv2.add(background_bg, image2_fg)
        result[bbox[1]:bbox[3], bbox[0]:bbox[2]] = dst

        # additionally replace inside the resulting bbox too dark pixels, the remnants of the crop borders
        clear_crop = result[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        black_pixels = np.where((clear_crop[:, :, 0] < colour_smoothing_threshold)
                                & (clear_crop[:, :, 1] < colour_smoothing_threshold)
                                & (clear_crop[:, :, 2] < colour_smoothing_threshold))
        clear_crop[black_pixels] = roi[black_pixels]
        result[bbox[1]:bbox[3], bbox[0]:bbox[2]] = clear_crop

    return result, df


def slide_titles_generator(background, array, width, height, REGULAR_FONTS, BOLD_FONTS, markov_text,
                           max_color_value=255, slide_title_bold_prob=0.38, slide_title_upper_prob=0.41,
                           change_slide_title_color_prob=0.04, max_slide_title_lines=2, min_slide_title_height=0.03,
                           max_slide_title_height=0.839, margins_coeff=4, right_gap_coeff=0.9, down_gap_coeff=0.1,
                           max_words=15, max_attempts=4, cut_last_line_coeff=4, red_line_prob=0.007,
                           patch_coeffs=(8, 15)):
    """Adding a slide title to the background."""
    result = np.copy(background)
    slide_titles = array[array[:, -1] == 0]
    random_color = get_random_color(max_value=max_color_value)

    bold_or_not = random.random()
    # randomly select a random bold or regular font
    if bold_or_not < slide_title_bold_prob:
        random_font = random.sample(BOLD_FONTS, 1)[0]
        fonts_dir = config.BOLD_FONTS_DIR
    else:
        random_font = random.sample(REGULAR_FONTS, 1)[0]
        fonts_dir = config.REGULAR_FONTS_DIR

    random_upper = random.random()
    # whether to add a red line
    random_red_line = random.random() < red_line_prob

    for b, bbox in enumerate(slide_titles):
        # monitor the number of attempts to generate suitable text to stop if there are too many attempts
        attempt_counts = 0
        right_attempt = False
        w = bbox[2] - bbox[0]
        h = (bbox[3] - bbox[1]) / height
        max_lines_count = max(min(max_slide_title_lines, math.floor(h/min_slide_title_height)), 1)
        min_lines_count = max(min(1, math.ceil(h/max_slide_title_height)), 1)
        # until all rows in the bbox are generated correctly
        while not right_attempt:
            config.LOG.info(f'{b} slide_title start')
            random_lines_count = random.randint(min_lines_count, max_lines_count)
            lines_height = int(h / random_lines_count * height)
            text = random.sample(markov_text, random_lines_count)

            random_long_line_idx = random.randint(0, len(text) - 1)
            # make the text in a random line long so that it reaches the right edge of the bbox
            for _ in range(len(text)):
                text[random_long_line_idx] += '. ' + random.sample(text, 1)[0]

            # to speed up the work, do not make the lines too long
            text = [' '.join(t.split()[:max_words]) for t in text]
            # if there are more than one row
            if len(text) > 1:
                # if the artificially extended string is not the last one, shorten the last one
                if random_long_line_idx != len(text) - 1:
                    text[-1] = text[-1][:math.floor(len(text[-1]) / cut_last_line_coeff)]
                # add red line
                if random_red_line:
                    text[0] = '    ' + text[0]

            # randomly convert all text or the first characters in strings to uppercase
            if random_upper < slide_title_upper_prob:
                text = [s.upper() for s in text]
            else:
                for j, sentence in enumerate(text):
                    # the first term is always capitalized
                    if j != 0:
                        # convert first characters in random strings to lower case
                        if random.random() > slide_title_upper_prob:
                            text[j] = sentence[0].lower() + sentence[1:]

            # randomly generate line spacing
            random_lower_margin = random.randint(1, max(round(lines_height / margins_coeff), 1))

            generator = GeneratorFromStrings(
                        strings=text,
                        background_type=3,  # the background is chosen randomly from the folder with backgrounds
                        fonts=[os.path.join(fonts_dir, random_font)],  # font set
                        size=lines_height,  # bbox height
                        width=w,  # bbox width
                        alignment=0,  # left alignment
                        text_color=random_color,  # text color
                        margins=(0, 0, 0, 0),  # margins top, left, bottom, right
                        output_mask=True,  # returns the text mask as the second value
                        fit=True,  # no space between text and box borders
                        image_dir=config.TEMP_DIR)

            bad_right_gap = False
            for i in range(random_lines_count):
                new_x1 = bbox[0]
                new_y1 = bbox[1] + lines_height * i
                new_x2 = bbox[0] + w
                new_y2 = new_y1 + lines_height
                background_crop = background[new_y1:new_y2, new_x1:new_x2]
                background_crop = Image.fromarray(cv2.cvtColor(background_crop, cv2.COLOR_BGR2RGB))
                background_crop.save(os.path.join(generator.image_dir, 'text_line_background.png'))

                # add red line
                if random_red_line and i == 0:
                    generator.fit = False
                else:
                    generator.fit = True

                change_slide_title_color = random.random()

                # randomly change the color of the line (slide titles can have multi-colored lines)
                if change_slide_title_color <= change_slide_title_color_prob:
                    generator.text_color = get_random_color(max_value=max_color_value)

                # set the gap from the edge of the bbox to the text
                if i != random_lines_count - 1:
                    generator.margins = (0, 0, random_lower_margin, 0)

                (img, mask), sentence = generator.next()
                open_cv_image = np.array(img)
                open_cv_image = open_cv_image[:, :, ::-1].copy()
                mask = np.array(mask)[:, :, ::-1].copy()

                # randomly blur some text from the end of the line so that it is not cut off by a clear line
                random_patch_coeff = random.randint(patch_coeffs[0], patch_coeffs[1])
                patch_width = min(patch_coeffs[0], max(int((w - right_gap_coeff * w) / random_patch_coeff), 1))
                patch_crop = cv2.cvtColor(np.array(background_crop), cv2.COLOR_RGB2BGR)[:, w - patch_width:w]
                open_cv_image[:, w - patch_width:w] = patch_crop

                # make sure that text in artificially expanded line reaches right border of bbox or is close to it
                if i == random_long_line_idx:
                    check_right_gap = np.max(last_nonzero(mask, 1, invalid_val=-1))
                    # if this did not happen, then we consider this attempt to generate unsuccessful
                    if check_right_gap < right_gap_coeff * w:
                        bad_right_gap = True
                        config.LOG.info('bad right gap')

                # make sure that the text in the bottom line reaches the bottom border of the bbox or is close to it
                check_lower_gap = (last_nonzero(mask, 1, invalid_val=-1)[:, 0][::-1] != -1).argmax(axis=0)
                if (check_lower_gap < down_gap_coeff * h * height) and (i == random_lines_count - 1) \
                        and not bad_right_gap:
                    config.LOG.info(f'{b} slide_title ready')
                    right_attempt = True

                if (check_lower_gap >= down_gap_coeff * h * height) and (i == random_lines_count - 1):
                    config.LOG.info('bad down gap')

                result[new_y1:new_y2, new_x1:new_x2] = open_cv_image

                # if the current attempt was considered unsuccessful, increase the counter of attempts
                if not right_attempt:
                    attempt_counts += 1

                # if there were too many attempts, return 0 instead of text height
                if attempt_counts >= max_attempts * random_lines_count:
                    return result, 0

    return result, h/random_lines_count


def content_titles_generator(background, array,  width, height, REGULAR_FONTS, BOLD_FONTS, markov_text,
                             slide_title_lines_height, content_title_bold_prob=0.36, content_title_upper_prob=0.38,
                             max_color_value=255, max_content_title_lines=3, min_content_title_height=0.022,
                             max_content_title_height=0.228, margins_coeff=4, right_gap_coeff=0.9, down_gap_coeff=0.16,
                             max_words=23, max_attempts=4, cut_last_line_coeff=4, red_line_prob=0.007,
                             patch_coeffs=(8, 15)):
    """Adding content titles to the background."""
    result = np.copy(background)
    content_titles = array[array[:, -1] == 1]
    # content titles are generated in a simple dark font
    random_color = get_simple_random_color()

    bold_or_not = random.random()
    # randomly select a random bold or regular font
    if bold_or_not < content_title_bold_prob:
        random_font = random.sample(config.SIMPLE_BOLD_FONTS, 1)[0]
        fonts_dir = config.SIMPLE_BOLD_FONTS_DIR
    else:
        random_font = random.sample(config.SIMPLE_REGULAR_FONTS, 1)[0]
        fonts_dir = config.SIMPLE_REGULAR_FONTS_DIR

    all_heights = content_titles[:, 3] - content_titles[:, 1]
    # if one of the content titles on the slide is much larger in height, then we do not take such a slide
    if all_heights.max() > all_heights.min() * max_content_title_lines:
        config.LOG.error('Some content_title too big!')
        return result, 0

    # choose how many lines the smallest content header will consist of
    r = random.randint(1, max_content_title_lines)
    # set the line height of the smallest content header
    min_height = all_heights[np.where(all_heights == all_heights.min())[0][0]] / height / r
    # make sure that the height of the content title is not greater than the height of the slide title
    max_possible_height = min(min_height, slide_title_lines_height)
    # set the number of lines in each content title
    lines_counts = np.ceil(all_heights / height / max_possible_height).astype(int)
    # if the lines turned out to be more than the limit value, we automatically increase the font
    lines_counts[np.where(lines_counts > max_content_title_lines)[0]] = max_content_title_lines

    # randomly generate line spacing
    random_lower_margin = random.randint(1, max(round(min_height * height / margins_coeff), 1))
    random_upper = random.random()

    # whether to add a red line to all content titles on a slide
    random_red_line = random.random() < red_line_prob

    content_titles_heights = set()
    for b, bbox in enumerate(content_titles):
        # monitor the number of attempts to generate suitable text to stop if there are too many attempts
        attempt_counts = 0
        right_attempt = False
        w = bbox[2] - bbox[0]
        h = (bbox[3] - bbox[1]) / height
        # until all rows in the bbox are generated correctly
        while not right_attempt:
            config.LOG.info(f'{b} content_title start')
            random_lines_count = lines_counts[b]
            lines_height = int(h / random_lines_count * height)
            text = random.sample(markov_text, random_lines_count)
            # sort the text by length so that the blank line, if anything, is always at the bottom
            text = sorted(text, key=lambda x: len(x), reverse=True)
            # make the text in the first line long so that it reaches the right edge of the bbox
            text[0] = '. '.join(text)
            # to speed up the work, do not make the lines too long
            text = [' '.join(t.split()[:max_words]) for t in text]
            # if there is more than one line, we shorten the last line
            if len(text) > 1:
                text[-1] = text[-1][:math.floor(len(text[-1]) / cut_last_line_coeff)]
                # add red line
                if random_red_line:
                    text[0] = '    ' + text[0]
            # add a colon at the end
            text[-1] += ':'

            # randomly convert all text or the first characters in strings to uppercase
            if random_upper < content_title_upper_prob:
                text = [s.upper() for s in text]
            else:
                for j, sentence in enumerate(text):
                    # the first term is always capitalized
                    if j != 0:
                        # convert first characters in random strings to lower case
                        if random.random() > content_title_upper_prob:
                            text[j] = sentence[0].lower() + sentence[1:]

            generator = GeneratorFromStrings(
                        strings=text,
                        count=-1,  # number of samples, default=1000
                        background_type=3,  # the background is chosen randomly from the folder with backgrounds
                        fonts=[os.path.join(fonts_dir, random_font)],  # font set
                        size=lines_height,  # bbox height
                        width=w,  # bbox width
                        alignment=0,  # left alignment
                        text_color=random_color,  # text color
                        margins=(0, 0, 0, 0),  # margins top, left, bottom, right
                        output_mask=True,  # returns the text mask as the second value
                        fit=True,  # no space between text and box borders
                        image_dir=config.TEMP_DIR)

            bad_right_gap = False
            for i in range(random_lines_count):
                new_x1 = bbox[0]
                new_y1 = bbox[1] + lines_height * i
                new_x2 = bbox[0] + w
                new_y2 = new_y1 + lines_height
                background_crop = background[new_y1:new_y2, new_x1:new_x2]
                background_crop = Image.fromarray(cv2.cvtColor(background_crop, cv2.COLOR_BGR2RGB))
                background_crop.save(os.path.join(generator.image_dir, 'text_line_background.png'))

                # add red line
                if random_red_line and i == 0:
                    generator.fit = False
                else:
                    generator.fit = True

                # set the gap from the edge of the bbox to the text
                if i != random_lines_count - 1:
                    generator.margins = (0, 0, random_lower_margin, 0)

                (img, mask), sentence = generator.next()
                open_cv_image = np.array(img)
                open_cv_image = open_cv_image[:, :, ::-1].copy()
                mask = np.array(mask)[:, :, ::-1].copy()

                # randomly blur some text from the end of the line so that it is not cut off by a clear line
                random_patch_coeff = random.randint(patch_coeffs[0], patch_coeffs[1])
                patch_width = min(patch_coeffs[0], max(int((w - right_gap_coeff * w) / random_patch_coeff), 1))
                patch_crop = cv2.cvtColor(np.array(background_crop), cv2.COLOR_RGB2BGR)[:, w - patch_width:w]
                open_cv_image[:, w - patch_width:w] = patch_crop

                # make sure that the text in the first line reaches the right border of the bbox or is close to it
                if i == 0:
                    check_right_gap = np.max(last_nonzero(mask, 1, invalid_val=-1))
                    # if this did not happen, then we consider this attempt to generate unsuccessful
                    if check_right_gap < right_gap_coeff * w:
                        bad_right_gap = True
                        config.LOG.info('bad right gap')
                    else:
                        content_titles_heights.add(lines_height / height)

                # make sure that the text in the bottom line reaches the bottom border of the bbox or is close to it
                check_lower_gap = (last_nonzero(mask, 1, invalid_val=-1)[:, 0][::-1] != -1).argmax(axis=0)
                if (check_lower_gap < down_gap_coeff * h * height) and (i == random_lines_count - 1) \
                        and not bad_right_gap:
                    config.LOG.info(f'{b} content_title ready')
                    right_attempt = True
                    content_titles_heights.add(lines_height / height)

                if (check_lower_gap >= down_gap_coeff * h * height) and (i == random_lines_count - 1):
                    config.LOG.info('bad down gap')

                result[new_y1:new_y2, new_x1:new_x2] = open_cv_image

                # if the current attempt was considered unsuccessful, increase the counter of attempts
                if not right_attempt:
                    attempt_counts += 1

                # if there were too many attempts, return 0 instead of text height
                if attempt_counts >= max_attempts * random_lines_count:
                    return result, 0

    return result, min(content_titles_heights)


def bullets_generator(background, array,  width, height, REGULAR_FONTS, BOLD_FONTS, markov_text,
                      content_title_lines_height, bullet_bold_prob=0.3, bullet_upper_prob=0.06, max_color_value=255,
                      max_bullet_lines=12, min_bullet_height=0.022, max_bullet_height=0.254,
                      margins_coeff=4, right_gap_coeff=0.9, down_gap_coeff=0.16, max_words=42, max_attempts=4,
                      cut_last_line_coeff=4, red_line_prob=0.021, patch_coeffs=(8, 15)):
    """Adding shootouts to the background."""
    result = np.copy(background)
    bullets = array[array[:, -1] == 3]
    random_color = get_random_color(max_value=max_color_value)

    # randomly select a random bold or regular font
    bold_or_not = random.random()
    if bold_or_not < bullet_bold_prob:
        random_font = random.sample(BOLD_FONTS, 1)[0]
        fonts_dir = config.BOLD_FONTS_DIR
    else:
        random_font = random.sample(REGULAR_FONTS, 1)[0]
        fonts_dir = config.REGULAR_FONTS_DIR

    all_heights = bullets[:, 3] - bullets[:, 1]
    min_height = all_heights[np.where(all_heights == all_heights.min())[0][0]] / height
    # we make sure that the bullets are not higher than the content titles
    max_possible_height = min(min_height, content_title_lines_height)
    # calculate how many lines will fit in the smallest height bullet
    max_lines_count = math.ceil(max_possible_height / min_bullet_height)
    # randomly set the font size for all bullets on the slide, based on the size of the smallest bullet in height
    random_bullet_height = max_possible_height / random.randint(1, max_lines_count)
    # set the number of lines in each bullet
    lines_counts = np.around(all_heights / height / random_bullet_height).astype(int)
    # if the lines turned out to be more than the limit value, we automatically increase the font
    lines_counts[np.where(lines_counts > max_bullet_lines)[0]] = max_bullet_lines

    # randomly generate line spacing
    random_lower_margin = random.randint(1, max(round(random_bullet_height * height / margins_coeff), 1))
    random_upper = random.random()
    # whether to add a red line to all bullets on the slide
    random_red_line = random.random() < red_line_prob

    bullets_heights = set()
    for b, bbox in enumerate(bullets):
        # monitor the number of attempts to generate suitable text to stop if there are too many attempts
        attempt_counts = 0
        right_attempt = False
        w = bbox[2] - bbox[0]
        h = (bbox[3] - bbox[1]) / height
        # until all rows in the bbox are generated correctly
        while not right_attempt:
            config.LOG.info(f'{b} bullet start')
            random_lines_count = lines_counts[b]
            lines_height = int(h / random_lines_count * height)
            text = random.sample(markov_text, random_lines_count)

            random_long_line_idx = random.randint(0, len(text) - 1)
            # make the text in a random line long so that it reaches the right edge of the bbox
            for _ in range(len(text)):
                text[random_long_line_idx] += '. ' + random.sample(text, 1)[0]

            # to speed up the work, do not make the lines too long
            text = [' '.join(t.split()[:max_words]) for t in text]
            # if there are more than one row
            if len(text) > 1:
                # if the artificially extended string is not the last one, shorten the last one
                if random_long_line_idx != len(text) - 1:
                    text[-1] = text[-1][:math.floor(len(text[-1]) / cut_last_line_coeff)]
                # add red line
                if random_red_line:
                    text[0] = '    ' + text[0]

            # randomly convert all text or the first characters in strings to uppercase
            if random_upper < bullet_upper_prob:
                text = [s.upper() for s in text]
            else:
                for j, sentence in enumerate(text):
                    # the first term is always capitalized
                    if j != 0:
                        # convert first characters in random strings to lower case
                        if random.random() > bullet_upper_prob:
                            text[j] = sentence[0].lower() + sentence[1:]

            generator = GeneratorFromStrings(
                        strings=text,
                        count=-1,  # number of samples, default=1000
                        background_type=3,  # the background is chosen randomly from the folder with backgrounds
                        fonts=[os.path.join(fonts_dir, random_font)],  # font set
                        size=lines_height,  # bbox height
                        width=w,  # bbox width
                        alignment=0,  # left alignment
                        text_color=random_color,  # text color
                        margins=(0, 0, 0, 0),  # margins top, left, bottom, right
                        output_mask=True,  # returns the text mask as the second value
                        fit=True,  # no space between text and box borders
                        image_dir=config.TEMP_DIR)
            bad_right_gap = False
            for i in range(random_lines_count):
                new_x1 = bbox[0]
                new_y1 = bbox[1] + lines_height * i
                new_x2 = bbox[0] + w
                new_y2 = new_y1 + lines_height
                background_crop = background[new_y1:new_y2, new_x1:new_x2]
                background_crop = Image.fromarray(cv2.cvtColor(background_crop, cv2.COLOR_BGR2RGB))
                background_crop.save(os.path.join(generator.image_dir, 'text_line_background.png'))

                # add red line
                if random_red_line and i == 0:
                    generator.fit = False
                else:
                    generator.fit = True

                # set the gap from the edge of the bbox to the text
                if i != random_lines_count - 1:
                    generator.margins = (0, 0, random_lower_margin, 0)

                (img, mask), sentence = generator.next()
                open_cv_image = np.array(img)
                open_cv_image = open_cv_image[:, :, ::-1].copy()
                mask = np.array(mask)[:, :, ::-1].copy()

                # randomly blur some text from the end of the line so that it is not cut off by a clear line
                random_patch_coeff = random.randint(patch_coeffs[0], patch_coeffs[1])
                patch_width = min(patch_coeffs[0], max(int((w - right_gap_coeff * w) / random_patch_coeff), 1))
                patch_crop = cv2.cvtColor(np.array(background_crop), cv2.COLOR_RGB2BGR)[:, w - patch_width:w]
                open_cv_image[:, w - patch_width:w] = patch_crop

                # make sure that text in artificially expanded line reaches right border of bbox or is close to it
                if i == random_long_line_idx:
                    check_right_gap = np.max(last_nonzero(mask, 1, invalid_val=-1))
                    # if this did not happen, then we consider this attempt to generate unsuccessful
                    if check_right_gap < right_gap_coeff * w:
                        bad_right_gap = True
                        config.LOG.info('bad right gap')
                    else:
                        bullets_heights.add(lines_height / height)

                # make sure that the text in the bottom line reaches the bottom border of the bbox or is close to it
                check_lower_gap = (last_nonzero(mask, 1, invalid_val=-1)[:, 0][::-1] != -1).argmax(axis=0)
                if (check_lower_gap < down_gap_coeff * h * height) and (i == random_lines_count - 1) \
                        and not bad_right_gap:
                    config.LOG.info(f'{b} bullet ready')
                    right_attempt = True
                    bullets_heights.add(lines_height / height)

                if (check_lower_gap >= down_gap_coeff * h * height) and (i == random_lines_count - 1):
                    config.LOG.info('bad down gap')

                result[new_y1:new_y2, new_x1:new_x2] = open_cv_image

                # if the current attempt was considered unsuccessful, increase the counter of attempts
                if not right_attempt:
                    attempt_counts += 1
                # if there were too many attempts, return 0 instead of text height
                if attempt_counts >= max_attempts * random_lines_count:
                    return result, 0

    return result, min(bullets_heights)


def notes_generator(background, array,  width, height, REGULAR_FONTS, BOLD_FONTS, markov_text, bullet_lines_height,
                    note_bold_prob=0.04, note_upper_prob=0.06, max_color_value=255,
                    max_note_lines=3, min_note_height=0.014, max_note_height=0.166,
                    slides_numbers_width_threshold=0.021, star_t_prob=0.1,
                    margins_coeff=4, right_gap_coeff=0.9, down_gap_coeff=0.16, max_words=16, max_attempts=4,
                    cut_last_line_coeff=4, patch_coeffs=(8, 15)):
    """Adding notes to the background."""
    result = np.copy(background)
    notes = array[array[:, -1] == 9]
    random_color = get_random_color(max_value=max_color_value)

    # randomly select a random bold or regular font
    bold_or_not = random.random()
    if bold_or_not < note_bold_prob:
        random_font = random.sample(BOLD_FONTS, 1)[0]
        fonts_dir = config.BOLD_FONTS_DIR
    else:
        random_font = random.sample(REGULAR_FONTS, 1)[0]
        fonts_dir = config.REGULAR_FONTS_DIR

    all_heights = notes[:, 3] - notes[:, 1]
    all_widths = notes[:, 2] - notes[:, 0]

    min_height = all_heights[np.where(all_heights == all_heights.min())[0][0]] / height
    # we make sure that the notes are not higher than the bullets
    max_possible_height = min(min_height, bullet_lines_height)
    # calculate how many lines will fit in the smallest note
    max_lines_count = math.ceil(max_possible_height / min_note_height)
    # randomly set the font size for all notes on a slide, based on the size of the smallest note in height
    random_note_height = max_possible_height / random.randint(1, max_lines_count)
    # set the number of lines in each note
    lines_counts = np.around(all_heights / height / random_note_height).astype(int)
    # if the lines turned out to be more than the limit value, we automatically increase the font
    lines_counts[np.where(lines_counts > max_note_lines)[0]] = max_note_lines

    # notes with too narrow boxes are marked as slide numbers
    is_slide_number = all_widths / width < slides_numbers_width_threshold
    lines_counts[is_slide_number] = 1

    # randomly generate line spacing
    random_lower_margin = random.randint(1, max(round(random_note_height * height / margins_coeff), 1))
    random_upper = random.random()

    notes_heights = set()
    for b, bbox in enumerate(notes):
        # monitor the number of attempts to generate suitable text to stop if there are too many attempts
        attempt_counts = 0
        right_attempt = False
        w = bbox[2] - bbox[0]
        h = (bbox[3] - bbox[1]) / height
        # until all rows in the bbox are generated correctly
        while not right_attempt:
            config.LOG.info(f'{b} note start')
            random_lines_count = lines_counts[b]
            number = is_slide_number[b]
            lines_height = int(h / random_lines_count * height)
            text = random.sample(markov_text, random_lines_count)

            # if the footnote is a page number
            if number:
                # replace string with random pseudo slide number
                text = str(random.randint(1, 10))
            else:
                random_long_line_idx = random.randint(0, len(text) - 1)
                # make the text in a random line long so that it reaches the right edge of the bbox
                for _ in range(len(text)):
                    text[random_long_line_idx] += '. ' + random.sample(text, 1)[0]
                # to speed up the work, do not make the lines too long
                text = [' '.join(t.split()[:max_words]) for t in text]
                # if there are more than one row
                if len(text) > 1:
                    # if the artificially extended string is not the last one, shorten the last one
                    if random_long_line_idx != len(text) - 1:
                        text[-1] = text[-1][:math.floor(len(text[-1]) / cut_last_line_coeff)]

                # randomly convert all text or the first characters in strings to uppercase
                if random_upper < note_upper_prob:
                    text = [s.upper() for s in text]
                else:
                    for j, sentence in enumerate(text):
                        # the first term is always capitalized
                        if j != 0:
                            # convert first characters in random strings to lower case
                            if random.random() > note_upper_prob:
                                text[j] = sentence[0].lower() + sentence[1:]

                # randomly add a "*" character at the beginning of the line
                for j, sentence in enumerate(text):
                    if random.random() < star_t_prob:
                        text[j] = '* ' + sentence

            generator = GeneratorFromStrings(
                        strings=text,
                        count=-1,  # number of samples, default=1000
                        background_type=3,  # the background is chosen randomly from the folder with backgrounds
                        fonts=[os.path.join(fonts_dir, random_font)],  # font set
                        size=lines_height,  # bbox height
                        width=w,  # bbox width
                        alignment=0,  # left alignment
                        text_color=random_color,  # text color
                        margins=(0, 0, 0, 0),  # margins top, left, bottom, right
                        output_mask=True,  # returns the text mask as the second value
                        fit=True,  # no space between text and box borders
                        image_dir=config.TEMP_DIR)
            bad_right_gap = False
            for i in range(random_lines_count):
                new_x1 = bbox[0]
                new_y1 = bbox[1] + lines_height * i
                new_x2 = bbox[0] + w
                new_y2 = new_y1 + lines_height
                background_crop = background[new_y1:new_y2, new_x1:new_x2]
                background_crop = Image.fromarray(cv2.cvtColor(background_crop, cv2.COLOR_BGR2RGB))
                background_crop.save(os.path.join(generator.image_dir, 'text_line_background.png'))

                # set the gap from the edge of the bbox to the text
                if i != random_lines_count - 1:
                    generator.margins = (0, 0, random_lower_margin, 0)

                (img, mask), sentence = generator.next()
                open_cv_image = np.array(img)
                open_cv_image = open_cv_image[:, :, ::-1].copy()
                mask = np.array(mask)[:, :, ::-1].copy()

                # randomly blur some text from the end of the line so that it is not cut off by a clear line
                random_patch_coeff = random.randint(patch_coeffs[0], patch_coeffs[1])
                patch_width = min(patch_coeffs[0], max(int((w - right_gap_coeff * w) / random_patch_coeff), 1))
                patch_crop = cv2.cvtColor(np.array(background_crop), cv2.COLOR_RGB2BGR)[:, w - patch_width:w]
                open_cv_image[:, w - patch_width:w] = patch_crop

                # make sure that text in artificially expanded line reaches right border of bbox or is close to it
                if not number and i == random_long_line_idx:
                    check_right_gap = np.max(last_nonzero(mask, 1, invalid_val=-1))
                    # if this did not happen, then we consider this attempt to generate unsuccessful
                    if check_right_gap < right_gap_coeff * w:
                        bad_right_gap = True
                        config.LOG.info('bad right gap')
                    else:
                        notes_heights.add(lines_height / height)

                # make sure that the text in the bottom line reaches the bottom border of the bbox or is close to it
                check_lower_gap = (last_nonzero(mask, 1, invalid_val=-1)[:, 0][::-1] != -1).argmax(axis=0)
                if (check_lower_gap < down_gap_coeff * h * height) and (i == random_lines_count - 1) \
                        and not bad_right_gap:
                    config.LOG.info(f'{b} note ready')
                    right_attempt = True
                    notes_heights.add(lines_height / height)

                if (check_lower_gap >= down_gap_coeff * h * height) and (i == random_lines_count - 1):
                    config.LOG.info('bad down gap')

                result[new_y1:new_y2, new_x1:new_x2] = open_cv_image

                # if the current attempt was considered unsuccessful, increase the counter of attempts
                if not right_attempt:
                    attempt_counts += 1

                # if there were too many attempts, return 0 instead of text height
                if attempt_counts >= max_attempts * random_lines_count:
                    return result, 0

    return result, min(notes_heights)


def draw_visualization(background, array):
    """Drawing bboxes on the background."""
    visualization = np.copy(background)
    for bbox in array:
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        label = bbox[-1]
        color = config.LABELS_TO_COLORS[config.ALL_CLASSES[label]]
        visualization = cv2.rectangle(visualization, (x1, y1), (x2, y2), color, 2)
        cv2.putText(visualization, config.ALL_CLASSES[label], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    return visualization


def gen_save(imagename, background, visualization, lines):
    """Saving the resulting slide and its copy with rendered bboxes to the appropriate folders."""
    out_name = imagename.rsplit('/')[-1][:-4] + '_gen_' + secrets.token_hex(nbytes=16) + '.jpg'
    background = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    background.save(os.path.join(config.OUTPUT_GENERATION_DIR, out_name), quality=95)
    visualization = Image.fromarray(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB))
    visualization.save(os.path.join(config.OUTPUT_VISUALIZATION_DIR, out_name), quality=95)

    with open(os.path.join(config.OUTPUT_GENERATION_DIR, out_name[:-3] + 'txt'), "w") as f:
        for line in lines:
            # skipping persons
            if not line[:2] == '11':
                f.write(line)
