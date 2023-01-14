# Creation simple synthetic slides in pptx format.

<p align="center">
    <img src="examples.png", width="700px">
</p>

### Getting started

Install dependencies by  
```
pip install -r requirements.txt
```

The contents of the *./data* folder should be presented as follows:  
```
├── text_donor.txt
├── temp
└── backgrounds
    └── set of jpg slide backgrounds
```

#### Сlasses used in generation
* `content_title` - the textual description of the related content in a `content block` is typically at the top of that block, often ending in a colon. Content can be anything - text classes, `image`, `table`, `diagram`, etc. This class can only be located inside some kind of `content block`
* `content_block` - text or media elements combined by the author in one area of the slide. `content block` can often be nested: an obvious example is a nested list, in which one of the elements is itself a list \(and therefore a `content block`\)
* `bullet` - a block of text that is logically and visually separated from other slide elements. The slide creators themselves visually separate the bullets from each other (as a rule, using a wider line spacing than usual). `bullet` can be both inside the `content block` and outside it

As a result of generation, folder *output_generation* will contain the resulting synthetic pptx-slides, its jpg-screenshots and its text files with yolo-markup.

### Data description

When generating, the following approach is used: bbox coordinates are taken from the markup of real slides, a synthetic slide is created by placing other random content in these bboxes.  

1. **Backgrounds**
    * The backgrounds on the slides were randomly selected from three options: a white background, a random colored background, and a random picture (without text).
2. **Layouts**
    The following 3 types of templates were used by me:
    * a column of many content_blocks "`content_title` + one-line `bullet`".  
    * two "`content_title` + `bullet`" content_blocks, one in the top half of the slide, the other in the bottom.
    * a column of several content_blocks "`content_title` + several `bullet`".
3. **Text**
    * As a donor for the generation of random text I used my favorite books, combined into one text document.  
    For text substitution, [markovgen](https://pypi.org/project/markovgen/) library was used: a set of text samples was generated from a text donor, then duplicate samples (and quote characters as unnecessary) were removed from them.
    The text was rendered using [TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator) library, in particular the *GeneratorFromStrings* module.
4. **Fonts**
    * Installed fonts.

### Event probabilities analysis for item substitution

The substitution probabilities are mostly the same as in the *jpg-generator*, depending on the template, the number of words per line, font size, deviation of the left edge of the content_title from its bullets, the number of lines and elements in the content_block changed.

### Additional remarks

When generating the text, it can go beyond the boundaries of the slide and be cut off when screenshotting - this is not essential, since the boundaries of the text boxes will still be adequate to the text.  
An interesting bug was also noticed - if during the generation process you open a folder in Explorer that contains any pptx file, then the generating presentation will not be able to close, which will lead to an error - be careful.
