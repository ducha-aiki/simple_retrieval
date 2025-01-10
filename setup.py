from setuptools import setup, find_packages

setup(
    name='simple_retrieval',
    version='0.1.0',
    author='Dmytro Mishkin',
    author_email='ducha.aiki@gmail.com  ',
    description='A simple image retrieval package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ducha-aiki/simple_retrieval',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        # List your package dependencies here
    ],
)