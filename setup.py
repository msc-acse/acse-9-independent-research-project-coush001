# Author (GitHub alias): coush001
from distutils.core import setup

setup(
  name = 'SeismicReduction',         # How you named your package folder (MyLib)
  packages = ['SeismicReduction'],   # Chose the same as "name"
  version = 'v1.0',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here:
  description = 'Perform unsupervised machine learning on seismic data.',   # Give a short description about your library
  # long_description=long_description,
  # long_description_content_type='text/markdown',  # This is important!
  author = 'coush001',                   # Type in your name
  author_email = 'coush001@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/coush001',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/msc-acse/acse-9-independent-research-project-coush001/archive/v1.0.tar.gz',    # I explain this later on
  keywords = ['Seismic Data', 'Unsupervised Learning', 'Machine Learning', 'VAE'],   # Keywords that define your package best
  install_requires=['livelossplot',
                    'matplotlib',
                    'numpy',
                    'scikit-image',
                    'scikit-learn',
                    'scipy',
                    'segypy',
                    'torch',
                    'torchvision',
                    'umap',
                    'umap-learn',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)