"""
Created on 2015-7-20

module: Util

@author: XiaLong
@contact: xl.1988.life@gmail.com
"""

# !/usr/bin/python
# -*- coding:utf-8 -*-

import sys


def strLatter(strInput, sep):
    listItem = strInput.split(sep)
    return listItem[-1]


def change(inputFile):
    inputFile = open(inputFile)
    outputFile = open("change.txt", "w")
    for line in inputFile.readlines():
        listItem = line.split("13:")
        outputLine = listItem[0] + " 13:" + listItem[1]
        outputFile.write(outputLine)
    inputFile.close()
    outputFile.close()


if __name__ == '__main__':
    change(sys.argv[1])
