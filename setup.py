from setuptools import setup

package_name = 'cti_image'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='eltsu',
    maintainer_email='43186596+EltonCN@users.noreply.github.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bodypix = cti_image.bodypix:main',
            'image_viewer = cti_image.image_viewer:main',
        ],
    },
)
