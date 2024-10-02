from setuptools import setup, find_packages

setup(
    name =                      'known',
    version =                   '0.0.23',
    url =                       'https://github.com/auto-notify-ps/known',
    author =                    'Nelson.S',
    author_email =              'mail.nelsonsharma@gmail.com',
    description =               'a collection of reusable python code',
    packages =                  find_packages(include=['known', 'known.*']),
    classifiers=                ['License :: OSI Approved :: MIT License'],
    #package_dir =               { '' : ''},
    install_requires =          [],
    include_package_data =      False,
    #python_requires =           ">=3.8",
)   