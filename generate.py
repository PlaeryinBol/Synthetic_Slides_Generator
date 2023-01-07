import os
import random

import cv2
import numpy as np
import pandas as pd

import config
import utils

markov_text = utils.generate_random_text(config.TEXT_DONOR, config.TEXT_SAMPLES)

# loading visual classes dataframes
tables_df = pd.read_csv(config.TABLES_DF, sep='\t')
clear_images_df = pd.read_csv(config.CLEAR_IMAGES_DF, sep='\t')
text_images_df = pd.read_csv(config.TEXT_IMAGES_DF, sep='\t')
diagrams_df = pd.read_csv(config.DIAGRAMS_DF, sep='\t')
schemes_df = pd.read_csv(config.SCHEMES_DF, sep='\t')
logos_df = pd.read_csv(config.LOGOS_DF, sep='\t')
icons_df = pd.read_csv(config.ICONS_DF, sep='\t')

# if necessary, we can set all already used visuals in dataframes as unused
if config.RESET_DFS:
    tables_df['is_used'] = False
    clear_images_df['is_used'] = False
    text_images_df['is_used'] = False
    diagrams_df['is_used'] = False
    schemes_df['is_used'] = False
    logos_df['is_used'] = False
    icons_df['is_used'] = False

dataset_samples = utils.get_pictures(config.DATASET_DIR)
while len(os.listdir(config.OUTPUT_VISUALIZATION_DIR)) < config.SLIDES_COUNT:
    imagename = random.choice(dataset_samples)
    config.LOG.info(imagename)
    try:
        image = cv2.imread(imagename)
        width = image.shape[1]
        height = image.shape[0]
        # read markup from dataset
        txt_path = imagename[:-4] + ".txt"
        with open(txt_path, "r") as f:
            lines = f.readlines()
            array = []
            lines_to_del = set()
            for idx, line in enumerate(lines):
                line = line.replace('\n', '').split(' ')
                label = int(line[0])

                x1, y1, x2, y2 = utils.yolo2tlrb(float(line[1]), float(line[2]), float(line[3]),
                                                 float(line[4]), width, height)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = max(0, x2)
                y2 = max(0, y2)

                color = config.LABELS_TO_COLORS[config.ALL_CLASSES[label]]
                image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, config.ALL_CLASSES[label], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

                # do not take bboxes with zero height/width and bboxes that go beyond the borders
                if (x2 - x1 > 0) and (y2 - y1 > 0):
                    # in the case of media classes, we can replace it with some random other
                    if label in config.MEDIA_CLASSES:
                        # but before that we check that table, scheme or diagram does not turn out to be too small
                        if ((x2 - x1) / width > config.MIN_MEDIA_WIDTH_COEFF) \
                                and ((y2 - y1) / height > config.MIN_MEDIA_HEIGHT_COEFF):
                            label = random.sample(config.MEDIA_CLASSES, 1)[0]
                        else:
                            label = 5

                        lines[idx] = str(label) + lines[idx][1:]
                    array.append([min(x1, width), min(y1, height), min(x2, width), min(y2, height), label])
                else:
                    lines_to_del.add(idx)

        # do not write incorrect bboxes in yolo-textfile
        lines = [el for i, el in enumerate(lines) if i not in lines_to_del]
        array = np.array(array)
        # if there is something on the slide besides the background
        if array.size != 0:
            classes = array[:, -1]

            # delete persons as unnecessary
            person_to_del = np.where(array[:, -1] == 11)[0]
            array = np.delete(array, person_to_del, 0)

            # too small slides are enlarged by 2 times
            if width <= 600:
                image = utils.image_resize(image, width=width*2)
                width = image.shape[1]
                height = image.shape[0]
                array[:, :-1] *= 2

            type_of_background = random.random()
            background = np.ones_like(image)
            # choose a white, color or picture slide background
            if type_of_background < config.COLOUR_BACKGROUND_PROB:
                background[:, :, 0] *= random.randint(0, 255)
                background[:, :, 1] *= random.randint(0, 255)
                background[:, :, 2] *= random.randint(0, 255)
            elif type_of_background < config.COLOUR_BACKGROUND_PROB + config.PICTURE_BACKGROUND_PROB:
                coords = np.array([[0, 0, width, height, 5]])
                background, new_clear_images_df = utils.add_media(background, coords, clear_images_df,
                                                                  5, config.CLEAR_IMAGES_FOLDER)
                if background is None:
                    config.LOG.warning(f'bad {os.path.basename(imagename)} slide, background image not found!')
                    continue
                if new_clear_images_df is None:
                    config.LOG.warning('empty dataframe, background image not found!')
                    clear_images_df['is_used'] = False
                    continue
                else:
                    clear_images_df = new_clear_images_df
            else:
                background *= 255

            # tables
            if (len(classes[classes == 4]) > 0):
                background, new_tables_df = utils.add_media(background, array, tables_df, 4, config.TABLES_FOLDER)
                # if no crop was found with a suitable height / width ratio for the donor bbox
                if background is None:
                    config.LOG.warning(f'bad {os.path.basename(imagename)} slide, table not found!')
                    continue
                # if in the dataframe there are only either already used or too small crops for the bbox-donor
                if new_tables_df is None:
                    config.LOG.warning('empty dataframe, table not found!')
                    tables_df['is_used'] = False
                    continue
                else:
                    tables_df = new_tables_df

            # images
            if (len(classes[classes == 5]) > 0):
                # decide whether to insert text or non text pictures on the background
                if random.random() > config.TEXT_IMAGES_PROBS:
                    background, new_clear_images_df = utils.add_media(background, array, clear_images_df,
                                                                      5, config.CLEAR_IMAGES_FOLDER)
                    # if no crop was found with a suitable height / width ratio for the donor bbox
                    if background is None:
                        config.LOG.warning(f'bad {os.path.basename(imagename)} slide, image not found!')
                        continue
                    # if in the dataframe there are only either already used or too small crops for the bbox-donor
                    if new_clear_images_df is None:
                        config.LOG.warning('empty dataframe, image not found!')
                        clear_images_df['is_used'] = False
                        continue
                    else:
                        clear_images_df = new_clear_images_df
                else:
                    background, new_text_images_df = utils.add_media(background, array, text_images_df,
                                                                     5, config.TEXT_IMAGES_FOLDER)
                    # if no crop was found with a suitable height / width ratio for the donor bbox
                    if background is None:
                        config.LOG.warning(f'bad {os.path.basename(imagename)} slide, text image not found!')
                        continue
                    # if in the dataframe there are only either already used or too small crops for the bbox-donor
                    if new_text_images_df is None:
                        config.LOG.warning('empty dataframe, text image not found!')
                        text_images_df['is_used'] = False
                        continue
                    else:
                        text_images_df = new_text_images_df

            # diagrams
            if (len(classes[classes == 6]) > 0):
                background, new_diagrams_df = utils.add_media(background, array, diagrams_df,
                                                              6, config.DIAGRAMS_FOLDER)
                # if no crop was found with a suitable height / width ratio for the donor bbox
                if background is None:
                    config.LOG.warning(f'bad {os.path.basename(imagename)} slide, diagram not found!')
                    continue
                # if in the dataframe there are only either already used or too small crops for the bbox-donor
                if new_diagrams_df is None:
                    config.LOG.warning('empty dataframe, diagram not found!')
                    diagrams_df['is_used'] = False
                    continue
                else:
                    diagrams_df = new_diagrams_df

            # schemes
            if (len(classes[classes == 7]) > 0):
                background, new_schemes_df = utils.add_media(background, array, schemes_df,
                                                             7, config.SCHEMES_FOLDER)
                # if no crop was found with a suitable height / width ratio for the donor bbox
                if background is None:
                    config.LOG.warning(f'bad {os.path.basename(imagename)} slide, scheme not found!')
                    continue
                # if in the dataframe there are only either already used or too small crops for the bbox-donor
                if new_schemes_df is None:
                    config.LOG.warning('empty dataframe, scheme not found!')
                    schemes_df['is_used'] = False
                    continue
                else:
                    schemes_df = new_schemes_df

            # logos
            if (len(classes[classes == 8]) > 0):
                background, new_logos_df = utils.add_logo_or_icon(background, array, logos_df,
                                                                  8, config.LOGOS_FOLDER)
                # if no crop was found with a suitable height / width ratio for the donor bbox
                if background is None:
                    config.LOG.warning(f'bad {os.path.basename(imagename)} slide, logo not found!')
                    continue
                # if in the dataframe there are only either already used or too small crops for the bbox-donor
                if new_logos_df is None:
                    config.LOG.warning('empty dataframe, logo not found!')
                    logos_df['is_used'] = False
                    continue
                else:
                    logos_df = new_logos_df

            # icons
            if (len(classes[classes == 10]) > 0):
                background, new_icons_df = utils.add_logo_or_icon(background, array, icons_df,
                                                                  10, config.ICONS_FOLDER)
                # if no crop was found with a suitable height / width ratio for the donor bbox
                if background is None:
                    config.LOG.warning(f'bad {os.path.basename(imagename)} slide, icon not found!')
                    continue
                # if in the dataframe there are only either already used or too small crops for the bbox-donor
                if new_icons_df is None:
                    config.LOG.warning('empty dataframe, icon not found!')
                    icons_df['is_used'] = False
                    continue
                else:
                    icons_df = new_icons_df

            good_gen = np.copy(background)

            # text classes
            if any(np.isin(config.TEXT_CLASSES, classes)):
                bad_try = False
                try_count = 0
                # trying to generate adequate text several attempts
                while try_count < config.MAX_TRY:
                    background = good_gen

                    # slide_title
                    if (len(classes[classes == 0]) > 0):
                        background, slide_title_lines_height = utils.slide_titles_generator(background, array,
                                                                                            width, height,
                                                                                            config.REGULAR_FONTS,
                                                                                            config.BOLD_FONTS,
                                                                                            markov_text)
                    else:
                        slide_title_lines_height = config.MAX_CONTENT_TITLE_HEIGHT
                    # if the attempt to generate elements of this class is unsuccessful,
                    # then the general attempt to generate the text of the slide is also unsuccessful
                    if not slide_title_lines_height:
                        config.LOG.warning('Max slide title attempts!')
                        bad_try = True

                    # content_titles
                    if (len(classes[classes == 1]) > 0) and not bad_try:
                        background, content_title_lines_height = utils.content_titles_generator(background, array,
                                                                                                width, height,
                                                                                                config.SIMPLE_REGULAR_FONTS,
                                                                                                config.SIMPLE_BOLD_FONTS,
                                                                                                markov_text,
                                                                                                slide_title_lines_height)
                    else:
                        content_title_lines_height = min(slide_title_lines_height, config.MAX_BULLET_HEIGHT)
                    # if the attempt to generate elements of this class is unsuccessful,
                    # then the general attempt to generate the text of the slide is also unsuccessful
                    if not content_title_lines_height:
                        config.LOG.warning('Max content title attempts!')
                        bad_try = True

                    # bullets
                    if (len(classes[classes == 3]) > 0) and not bad_try:
                        background, bullet_lines_height = utils.bullets_generator(background, array,
                                                                                  width, height,
                                                                                  config.REGULAR_FONTS,
                                                                                  config.BOLD_FONTS,
                                                                                  markov_text,
                                                                                  content_title_lines_height,
                                                                                  max_bullet_lines=6)
                    else:
                        bullet_lines_height = min(content_title_lines_height, config.MAX_NOTE_HEIGHT)
                    # if the attempt to generate elements of this class is unsuccessful,
                    # then the general attempt to generate the text of the slide is also unsuccessful
                    if not bullet_lines_height:
                        config.LOG.warning('Max bullet attempts!')
                        bad_try = True

                    # notes
                    if (len(classes[classes == 9]) > 0) and not bad_try:
                        background, note_lines_height = utils.notes_generator(background, array,
                                                                              width, height,
                                                                              config.REGULAR_FONTS,
                                                                              config.BOLD_FONTS,
                                                                              markov_text, bullet_lines_height)
                        # if the attempt to generate elements of this class is unsuccessful,
                        # then the general attempt to generate the text of the slide is also unsuccessful
                        if not note_lines_height:
                            config.LOG.warning('Max note attempts!')
                            bad_try = True

                    # if the attempt is successful, stop
                    if (try_count < config.MAX_TRY) and not bad_try:
                        config.LOG.info('Well done text!')
                        break

                    try_count += 1
                    bad_try = False

                # if there are too many attempts, go to the next donor slide
                if try_count == config.MAX_TRY:
                    config.LOG.warning(f'bad {os.path.basename(imagename)} slide')
                    continue
                else:
                    visualization = utils.draw_visualization(background, array)
                    utils.gen_save(imagename, background, visualization, lines)
                    config.LOG.info('good text slide')

            else:
                visualization = utils.draw_visualization(background, array)
                utils.gen_save(imagename, background, visualization, lines)
                config.LOG.info('good not text slide')

        config.LOG.info('-'*27)
    # if we manually interrupt generation, then we save dataframes so as not to lose information about visuals used
    except KeyboardInterrupt:
        tables_df.to_csv(config.TABLES_DF, sep='\t', index=False)
        clear_images_df.to_csv(os.path.join(config.ROOT_DIR, 'clear_images.tsv'), sep='\t', index=False)
        text_images_df.to_csv(os.path.join(config.ROOT_DIR, 'text_images.tsv'), sep='\t', index=False)
        diagrams_df.to_csv(os.path.join(config.ROOT_DIR, 'diagrams_crops.tsv'), sep='\t', index=False)
        schemes_df.to_csv(os.path.join(config.ROOT_DIR, 'schemes_crops.tsv'), sep='\t', index=False)
        logos_df.to_csv(os.path.join(config.ROOT_DIR, 'logos_crops.tsv'), sep='\t', index=False)
        icons_df.to_csv(os.path.join(config.ROOT_DIR, 'icons_crops.tsv'), sep='\t', index=False)
        config.LOG.info('dataframes saved')
        break
    # except Exception as e:
        # config.LOG.error(e)
    #     continue

    # break

# at the end of the generation, resave the dataframes
tables_df.to_csv(config.TABLES_DF, sep='\t', index=False)
clear_images_df.to_csv(os.path.join(config.ROOT_DIR, 'clear_images.tsv'), sep='\t', index=False)
text_images_df.to_csv(os.path.join(config.ROOT_DIR, 'text_images.tsv'), sep='\t', index=False)
diagrams_df.to_csv(os.path.join(config.ROOT_DIR, 'diagrams_crops.tsv'), sep='\t', index=False)
schemes_df.to_csv(os.path.join(config.ROOT_DIR, 'schemes_crops.tsv'), sep='\t', index=False)
logos_df.to_csv(os.path.join(config.ROOT_DIR, 'logos_crops.tsv'), sep='\t', index=False)
icons_df.to_csv(os.path.join(config.ROOT_DIR, 'icons_crops.tsv'), sep='\t', index=False)
config.LOG.info('dataframes saved')
