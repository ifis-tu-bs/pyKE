from setuptools import setup

setup(
    name='pyKE',
    version='0.3.0',
    packages=['pyke', 'pyke.models', 'pyke.openke', 'tests'],
    package_dir={
        'pyke': 'pyke',
    },
    package_data={
        'pyke': [
            'cpp_library/*.cpp',
            'cpp_library/*.h',
            'cpp_library/make.sh',
        ],
    },
    url='https://github.com/ifis-tu-bs/pyKE',
    license='MIT',
    author='Johannes Ruthmann',
    author_email='',
    description='Open-source library for knowledge representation learning',
    install_requires=[
        'numpy==1.14.5',
        'tensorflow==1.8.0',
        'pandas==0.23.3',
    ],
)
