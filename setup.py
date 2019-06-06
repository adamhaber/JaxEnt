from setuptools import setup, find_packages

long_description = '''
For more information, see
`the Github project page <https://github.com/adamhaber/jaxent>`_.
'''

setup(name='jaxent',
      version='0.1.0',
      description='A JAX-based python package for maximum entropy modelling of binary data',
      long_description=long_description,
      url='https://github.com/adamhaber/jaxent',
      license='MIT',
      author='Adam Haber',
      author_email='adamhaber@gmail.com',
      packages=find_packages(),
      install_requires=[
          'jax',
          'jaxlib',
      ])