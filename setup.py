import os
from setuptools import setup

README = open(os.path.join(os.path.dirname(__file__), 'README.md')).read()

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='django-south-rebase',
    version='0.1',
    packages=['south_rebase'],
    include_package_data=True,
    license='GPL License',
    description='Simple app for rebasing south migrations',
    long_description=README,
    url='https://github.com/roverorna/django-south-rebase',
    author='Tomasz Kontusz',
    author_email='tomasz.kontusz@gmail.com',
    classifiers=[
        'Environment :: Web Environment',
        'Framework :: Django',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GPL License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
    ],
)
