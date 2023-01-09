from setuptools import setup, find_packages


def get_requirements():
    with open("requirements.txt", "r") as f:
        line = f.readline()
        requirement_list = [requirement.replace("\n", "") for requirement in line]
        if "-e ." in requirement_list:
            requirement_list.remove("-e .")
            
            
        return requirement_list


PROJECT_NAME = "Predict Bank's Credit Risk"
PROJECT_VERSION = "0.0.1"
PROJECT_DESCRIPTION = "This is a machine learning model to predict Bank's Credit Risk"
PROJECT_AUTHOR = "ikshvaku"
PROJECT_DEPENDENCIES = get_requirements()
PROJECT_PACKAGES = find_packages()

setup(
    name=PROJECT_NAME,
    version= PROJECT_VERSION,
    description=PROJECT_DESCRIPTION,
    author=PROJECT_AUTHOR,
    install_requires= PROJECT_DEPENDENCIES,
    packages=PROJECT_PACKAGES
)