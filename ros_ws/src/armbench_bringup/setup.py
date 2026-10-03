from glob import glob

from setuptools import find_packages, setup

package_name = "armbench_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Aibo Ni",
    maintainer_email="aibo.ni@alumnos.upm.es",
    description="Launch files, controller configuration and simulation self-check for armbench.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "sim_check = armbench_bringup.sim_check:main",
            "torque_probe = armbench_bringup.torque_probe:main",
            "torque_compare = armbench_bringup.torque_compare:main",
            "energy_meter = armbench_bringup.energy_meter:main",
        ],
    },
)
