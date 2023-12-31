from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Segment Cars '
LONG_DESCRIPTION = 'A package that allows to segment the exterior car image'

# Setting up
setup(
    name="carSegement",
    version=VERSION,
    author="Tawfiq",
    author_email="<tawfiqaadnane@gmail.com>",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=['sys','opencv-python', 'numpy','segment_anything', 'ultralytics'],
    keywords=['python', 'image', 'car', 'segement'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)

