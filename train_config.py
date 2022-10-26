"""
Expected directory format:

VideoMatte Train/Valid:
    ├──fgr/
      ├── 0001/
        ├── 00000.jpg
        ├── 00001.jpg
    ├── pha/
      ├── 0001/
        ├── 00000.jpg
        ├── 00001.jpg
        
ImageMatte Train/Valid:
    ├── fgr/
      ├── sample1.jpg
      ├── sample2.jpg
    ├── pha/
      ├── sample1.jpg
      ├── sample2.jpg

Background Image Train/Valid
    ├── sample1.png
    ├── sample2.png

Background Video Train/Valid
    ├── 0000/
      ├── 0000.jpg/
      ├── 0001.jpg/

"""


DATA_PATHS = {

    'videomatte': {
        'train': '/content/VideoMatte240K_JPEG_SD/train',
        'valid': '/content/VideoMatte240K_JPEG_SD/test',
    },
    'imagematte': {
        'train': '../matting-data/ImageMatte/train',
        'valid': '../matting-data/ImageMatte/valid',
    },
    'background_images': {
        'train': '/content/Backgrounds',
        'valid': '/content/Backgrounds',
    },
    'background_videos': {
        'train': '/content/Untitled Folder',
        'valid': '/content/Untitled Folder',
    },
    
    
    'coco_panoptic': {
        'imgdir': '/content/Untitled Folder',
        'anndir': '/content/Untitled Folder',
        'annfile': '/content/Untitled Folder',
    },
    'spd': {
        'imgdir': '/content/Untitled Folder',
        'segdir': '/content/Untitled Folder',
    },
    'youtubevis': {
        'videodir': '/content/Untitled Folder',
        'annfile': '/content/Untitled Folder',
    }
    
}
