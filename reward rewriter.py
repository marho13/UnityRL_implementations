import numpy as np
import os
import re
import matplotlib.pyplot as plt

folder = "C:/Users/Martin/Desktop/Results/"
files = os.listdir(folder)
dicty = {"rewards.txt": "Road-Detection", "rewardCarla.txt":"Carla", "resultsCheckpoint.txt":"Checkpoint"}
resulty = []
for f in files:
    # if f[-3:] == "txt":
    # print(f)
    resulty.append(f)

def readFile(folder, filey):
    print(filey)
    reader = open(folder + filey, mode="r").read()
    if filey != "rewards.txt":
        splitText, maxy, miny = splitFile(reader, regexFunctionEpisode)
    else:
        splitText, maxy, miny = splitFile(reader, regexRewards)
    return splitText, maxy, miny


def regexFunctionEpisode(inputStr):
    search = re.search("episode: (\d+), gave a reward of (-?)(\d+)\.(\d+)", inputStr)
    return search

def regexRewards(inputStr):
    search = re.search("(-?)(\d+)\.(\d+)", inputStr)
    return search

def splitFile(fileInfo, refunc):
    output = []
    splitNewLine = fileInfo.split("\n")
    maxy = 0
    miny = 0
    for line in splitNewLine:
        reString = refunc(line)
        if reString:
            if reString.span()[1] > 25:
                outputText = regexRewards(line[:reString.span()[1]])
                result = float(outputText.group())
                output.append(result)

                if result < miny: miny=result
                if result > maxy: maxy=result


            else:
                result = float(reString.group())
                output.append(result)

                if result < miny: miny=result
                if result > maxy: maxy=result

    return output, maxy, miny

matplotLib = []
plotting = []
nameList = []
maxList = []
minList = []
ax = plt.subplot()
for file in resulty:
    output, maxy, miny = readFile(folder, file)
    maxList.append(maxy)
    minList.append(miny)
    matplotLib.append(output)
    nameList.append(dicty[file])

for x in range(len(matplotLib)):
    plotting.append([])
    for y in range(len(matplotLib[x])):
        plotting[-1].append(matplotLib[x][y]-minList[x]-30.0)

for rew in range(len(matplotLib)):
    ax.plot(plotting[rew], label=nameList[rew])
#
plt.ylabel("Reward")
plt.xlabel("Backpropagations")
plt.legend()
plt.show()

print(len(plotting[0]), len(plotting[1]), len(plotting[2]))