from setuptools import setup, find_packages

setup(
    name='TextualVideoToSpeech',
    version='1.0',
    description='Flask app for video-to-text-to-speech conversion',
    author='Shravani | Vanishka | Vignesh',
    packages=find_packages(),
    install_requires=[
        'flask',
        'werkzeug',
        'opencv-python',
        'opencv-contrib-python',
        'Pillow',
        'pytesseract',
        'gtts',
        'numpy',
        'imutils'
        'langdetect',
        'easyocr',        
        'unidecode',
    ],
    include_package_data=True
)
