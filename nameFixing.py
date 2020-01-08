import os
dirName = "onRoad"
files = os.listdir(dirName)
print(len(files), files)
for a in range(1, len(files)+1):
    os.rename("{}/{}".format(dirName, files[a-1]), "{}/cary{}.png".format(dirName, a))