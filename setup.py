from distutils.core import setup
setup(
  name = 'quickml',
  packages = ['quickml'],
  version = '0.1',
  license='MIT',
  description = 'Machine Learning with high level abstraction!',
  author = 'Joel Barmettler',
  author_email = 'michaelnavean@gmail.com',
  url = 'https://https://github.com/onuohasilver/quickml',
  keywords = ['machine learning', 'quick', 'data', 'analyse', 'predict', 'fast', 'baseline'],
  install_requires=[
          'category_encoders'
      ],
  classifiers=[  # Optional
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 3 - Alpha',

    # Indicate who your project is intended for
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',

    # Pick your license as you wish
    'License :: OSI Approved :: MIT License',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)