from setuptools import setup

setup(name='idora',
      version='0.01dev',
      description='Explore, find feature relevance scores and model your data',
      url='http://github.com/storborg/funniest',
      author='Max R',
      author_email='dontwanna@say.nl',
      license='Pblic',
#       packages=['numpy',etc],
      packages=setuptools.find_packages(),
      classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ),
      entry_ponits = { 'console_scipts': [ 'hello=Have_a_good_day_Hinda:say_hello'],
                     },
      
      )
