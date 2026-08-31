from glob import glob
from setuptools import find_packages, setup


package_name = "yolo_detector"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Qzz139",
    maintainer_email="182426073+Qzz139@users.noreply.github.com",
    description="Ultralytics YOLO detector node for ROS 2 camera streams.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "yolo_detector_node = yolo_detector.detector_node:main",
        ],
    },
)
