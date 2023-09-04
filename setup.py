from pathlib import Path

import setuptools

# The directory containing this file
HERE = Path(__file__).parent.resolve()

# The text of the README file
NAME = 'revelionn'
VERSION = '1.0.2'
AUTHOR = 'CAIS Lab'
SHORT_DESCRIPTION = 'Retrospective Extraction of Visual and Logical Insights for Ontology-based interpretation of ' \
                    'Neural Networks'
README = Path(HERE, 'README.md').read_text(encoding='utf-8')
URL = 'https://github.com/cais-lab/revelionn'
REQUIRES_PYTHON = '>=3.9'
LICENSE = 'BSD 3-Clause'

setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email='agafonov.a@spcras.ru',
    description=SHORT_DESCRIPTION,
    long_description=README,
    long_description_content_type='text/markdown',
    url=URL,
    python_requires=REQUIRES_PYTHON,
    license=LICENSE,
    packages=['revelionn', 'revelionn.mapping_nets', 'revelionn.utils'],
    package_dir={'revelionn': 'revelionn'},
    include_package_data=True,
    install_requires=[
        'pandas>=1.5.2',
        'chardet>=5.1.0',
        'matplotlib>=3.7.1',
        'networkx>=3.1',
        'numpy>=1.24.2',
        'nxontology>=0.5.0',
        'opencv-python>=4.7.0.72',
        'pytest>=7.3.1',
        'pytest-cov>=4.0.0',
        'scikit-learn>=0.24.2',
        'scipy>=1.10.1',
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'xaitk-saliency>=0.7.0'
    ],
    keywords=['explainable AI', 'XAI', 'interpretation', 'black-box', 'convolutional neural network', 'ontology',
              'concept extraction', 'visual explanation', 'logical explanation'],
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
