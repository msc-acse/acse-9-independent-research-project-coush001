from distutils.core import setup
setup(
  name = 'SeismicReduction',         # How you named your package folder (MyLib)
  packages = ['SeismicReduction'],   # Chose the same as "name"
  version = '0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here:
  description = 'Perform unsupervised machine learning on seismic data.',   # Give a short description about your library
  author = 'Hugo Coussens',                   # Type in your name
  author_email = 'hcoscos1@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/coush001',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/msc-acse/acse-9-independent-research-project-coush001/archive/0.1.tar.gz',    # I explain this later on
  keywords = ['seismicdata', 'unsupervisedlearning', 'machinelearning'],   # Keywords that define your package best
  install_requires=['bokeh',
                    'datashader',
                    'datashape',
                    'livelossplot',
                    'matplotlib',
                    'numpy',
                    'numpydoc',
                    'pytest',
                    'scikit-image',
                    'scikit-learn',
                    'scipy',
                    'seaborn',
                    'segypy',
                    'torch',
                    'torchvision',
                    'umap',
                    'umap-learn',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)