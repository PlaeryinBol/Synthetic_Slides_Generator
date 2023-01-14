import os

ROOT_DIR = './'
TEXT_DONOR = os.path.join(ROOT_DIR, 'text_donor.txt')  # text document for pseudotext sampling
TEMP_DIR = os.path.join(ROOT_DIR, 'temp')  # folder to save the intermediate slide and its screenshot
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output_generation')  # folder for saving output slides and markup for them

# creating the above folders
if not os.path.exists(TEMP_DIR):
    os.mkdir(TEMP_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# fonts that cause rendering bugs when substituted
BAD_FONTS = {'Wingdings', 'Wingdings 2', 'Wingdings 3', 'MS Reference Specialty', 'Symbol', 'HoloLens MDL2 Assets',
             'Segoe MDL2 Assets', 'MT Extra', 'Marlett', 'Microsoft Himalaya', 'Webdings', 'Bookshelf Symbol 7'}

BACKGROUNDS_DIR = './data/backgrounds'  # folder with pictures for backgrounds

PPTX_COUNT = 1000  # number of generated samples
MIN_BLOCK_COUNT = 4  # minimum possible number of content_block per slide
MAX_BLOCK_COUNT = 8  # minimum possible number of content_block per slide
SLIDE_WIDTH = 1200  # slide width
SLIDE_HEIGHT = 900  # slide height
TEXT_SHAPE_WIDTH = 100  # the width of each text shape (essentially does not affect anything)
TEXT_SHAPE_HEIGHT = 50  # the height of each text shape (essentially does not affect anything)

COLOUR_BACKGROUND_PROB = 0.14  # the probability of a colored background on a slide
PICTURE_BACKGROUND_PROB = 0.18  # the probability of picture background on the slide

MAX_MARGIN_AFTER_LINES = 4  # maximum possible line spacing
MIN_WORDS = 23  # the minimum possible number of words per line
MAX_WORDS = 42  # maximum number of words per line
MIN_STRING_SIZE = 8  # minimum possible font height
MAX_STRING_SIZE = 16  # maximum possible font height
HORIZONTAL_CLASS_GAP_COEFF = 8  # max possible deviation of left side of content_title from left side of its bullets

CONTENT_TITLE_BOLD_PROB = 0.36  # the probability of bolding of the content_title
CONTENT_TITLE_UPPER_PROB = 0.38  # probability of uppercase content_title
MAX_CONTENT_TITLE_LINES = 3  # the minimum possible number of content_title lines
CONTENT_TITLE_COLON_PROB = 0.46  # probability of ending a content title with a colon

BULLET_BOLD_PROB = 0.3  # probability of a bold bullet
BULLET_UPPER_PROB = 0.06  # probability of an upper register bullet
MAX_BULLET_LINES = 1  # maximum possible number of bullet lines
RED_LINE_PROB = 0  # probability of a red line in a bullet

MAX_BULLET_COUNT = 1  # maximum possible number of bullets in the content_block

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
