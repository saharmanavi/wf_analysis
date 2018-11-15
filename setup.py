import setuptools

setuptools.setup(
    name="ophys",
    version="0.1.0",
    url="https://github.com/saharmanavi/wf_analysis",

    author="Sahar Manavi",
    author_email="saharm@alleninstitute.org",

    description="analysis for chronic widefield imaging during change detection task",
    long_description=open('README.md').read(),

    packages=setuptools.find_packages(),

    # install_requires=[
    #     'Click',
    # ],

    # entry_points='''
    #     [console_scripts]
    #     summary_csv=braintv_pilot.summary:load_and_save
    # ''',

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        # 'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.4',
        # 'Programming Language :: Python :: 3.5',
    ],
)
