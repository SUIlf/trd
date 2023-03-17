#!/usr/bin/env python
#-*- coding:utf-8 -*-
#############################################
# File Name: setup.py
# Author:
# Mail:
# Created Time:
#############################################
from setuptools import setup, find_packages
setup(
    name = "trd",
    version = "0.1.1",
    keywords = ("pip", "tensor","tensor ring decomposition", "", ""),
    description = "tensor ring decomposition toolbox",
    long_description = "tensor ring decomposition toolbox",
    license = "BSD 3-Clause License",
    url = "https://github.com/SUIlf/TRD",
    author = "",
    author_email = "",
    packages = find_packages(),
    include_package_data = True,
    platforms = "any",
    install_requires = ['numpy']
)