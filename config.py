import logging
import os

SLIDES_COUNT = 60
TEXT_SAMPLES = 1000
RESET_DFS = True
ROOT_DIR = './data'

TABLES_DF = os.path.join(ROOT_DIR, 'dfs/tables_crops.tsv')
TABLES_FOLDER = os.path.join(ROOT_DIR, 'media/tables')
CLEAR_IMAGES_DF = os.path.join(ROOT_DIR, 'dfs/clear_images.tsv')
CLEAR_IMAGES_FOLDER = os.path.join(ROOT_DIR, 'media/clear_images')
TEXT_IMAGES_DF = os.path.join(ROOT_DIR, 'dfs/text_images.tsv')
TEXT_IMAGES_FOLDER = os.path.join(ROOT_DIR, 'media/text_images')
DIAGRAMS_DF = os.path.join(ROOT_DIR, 'dfs/diagrams_crops.tsv')
DIAGRAMS_FOLDER = os.path.join(ROOT_DIR, 'media/diagrams')
SCHEMES_DF = os.path.join(ROOT_DIR, 'dfs/schemes_crops.tsv')
SCHEMES_FOLDER = os.path.join(ROOT_DIR, 'media/schemes')
LOGOS_DF = os.path.join(ROOT_DIR, 'dfs/logos_crops.tsv')
LOGOS_FOLDER = os.path.join(ROOT_DIR, 'media/logos')
ICONS_DF = os.path.join(ROOT_DIR, 'dfs/icons_crops.tsv')
ICONS_FOLDER = os.path.join(ROOT_DIR, 'media/icons')

DATASET_DIR = os.path.join(ROOT_DIR, 'dataset_yolo_format')  # folder with dataset in yolo format
TEXT_DONOR = os.path.join(ROOT_DIR, 'text_donor.txt')  # text document for pseudotext sampling
REGULAR_FONTS_DIR = os.path.join(ROOT_DIR, 'fonts/regular_fonts')  # normal font folder
REGULAR_FONTS = os.listdir(REGULAR_FONTS_DIR)  # list of normal fonts
BOLD_FONTS_DIR = os.path.join(ROOT_DIR, 'fonts/bold_fonts')  # bold font folder
BOLD_FONTS = os.listdir(BOLD_FONTS_DIR)  # list of bold fonts
SIMPLE_REGULAR_FONTS_DIR = os.path.join(ROOT_DIR, 'fonts/simple_regular_fonts')  # folder with simple regular fonts
SIMPLE_REGULAR_FONTS = os.listdir(SIMPLE_REGULAR_FONTS_DIR)  # list of simple fonts in normal style
SIMPLE_BOLD_FONTS_DIR = os.path.join(ROOT_DIR, 'fonts/simple_bold_fonts')  # folder with simple bold fonts
SIMPLE_BOLD_FONTS = os.listdir(SIMPLE_BOLD_FONTS_DIR)  # list of simple bold fonts

OUTPUT_GENERATION_DIR = os.path.join(ROOT_DIR, 'output_generation')  # folder to save slides
if not os.path.exists(OUTPUT_GENERATION_DIR):
    os.mkdir(OUTPUT_GENERATION_DIR)
OUTPUT_VISUALIZATION_DIR = os.path.join(ROOT_DIR, 'output_visualization')  # folder for slides with bbox markup
if not os.path.exists(OUTPUT_VISUALIZATION_DIR):
    os.mkdir(OUTPUT_VISUALIZATION_DIR)
TEMP_DIR = os.path.join(ROOT_DIR, 'temp')  # folder to save the background of the element's generating text
if not os.path.exists(TEMP_DIR):
    os.mkdir(TEMP_DIR)

ALL_CLASSES = {
                0: 'slide_title',
                1: 'content_title',
                2: 'content_block',
                3: 'bullet',
                4: 'table',
                5: 'image',
                6: 'diagram',
                7: 'scheme',
                8: 'logo',
                9: 'note',
                10: 'icon',
                11: 'person'
               }

LABELS_TO_COLORS = {
                    "slide_title": (219, 112, 147),
                    "content_title": (173, 255, 47),
                    "content_block": (0, 0, 0),
                    "bullet": (255, 140, 0),
                    "table": (0, 0, 128),
                    "image": (255, 0, 255),
                    "diagram": (72, 61, 139),
                    "scheme": (47, 79, 79),
                    "logo": (95, 158, 160),
                    "note": (106, 90, 205),
                    "icon": (128, 0, 128),
                    "person": (130, 212, 21)
                    }

MAX_TRY = 4  # number of attempts to generate text on a slide
TEXT_CLASSES = [0, 1, 3, 9]  # list of indexes of text classes
MEDIA_CLASSES = [4, 5, 6, 7]  # list of indexes of visual classes
TEXT_IMAGES_PROBS = 0.14  # the probability of choosing a picture with text instead of a picture without text
COLOUR_BACKGROUND_PROB = 0.14  # color background substitution probability
PICTURE_BACKGROUND_PROB = 0.18  # picture background substitution probability
MIN_MEDIA_WIDTH_COEFF = 0.027  # minimum possible media class width (from slide width)
MIN_MEDIA_HEIGHT_COEFF = 0.035  # minimum possible media class height (from slide height)

# text classes line height parameters (from slide height)
MIN_SLIDE_TITLE_HEIGHT, MAX_SLIDE_TITLE_HEIGHT = 0.03, 0.839
MIN_CONTENT_TITLE_HEIGHT, MAX_CONTENT_TITLE_HEIGHT = 0.022, 0.228
MIN_BULLET_HEIGHT, MAX_BULLET_HEIGHT = 0.022, 0.254
MIN_NOTE_HEIGHT, MAX_NOTE_HEIGHT = 0.014, 0.166

logging.basicConfig(filename="app.log", level=logging.INFO, filemode="w")
LOG = logging.getLogger()
