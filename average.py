import numpy
import scipy.io

average_ad = numpy.zeros((81, 97, 83))
average_young = numpy.zeros((81, 97, 83))
average_mid = numpy.zeros((81, 97, 83))
average_old = numpy.zeros((81, 97, 83))

images = scipy.io.loadmat("d:/RecentlyUsed/Graduation Project/adData.mat")
images = images['adData']

len_AD = len(images[0])
for i in range(len_AD):
    average_ad += images[0, i][3][0][0][4]


images = scipy.io.loadmat("d:/RecentlyUsed/Graduation Project/youngData.mat")
images = images['youngData']

len_young = len(images[0])
for i in range(len_young):
    average_young += images[0, i][3][0][0][4]


images = scipy.io.loadmat("d:/RecentlyUsed/Graduation Project/midData.mat")
images = images['midData']

len_mid = len(images[0])
for i in range(len_mid):
    average_mid += images[0, i][3][0][0][4]


images = scipy.io.loadmat("d:/RecentlyUsed/Graduation Project/oldData.mat")
images = images['oldData']

len_old = len(images[0])
for i in range(len_old):
    average_old += images[0, i][3][0][0][4]

average_ad = average_ad / len_AD
average_young = average_young / len_young
average_mid = average_mid / len_mid
average_old = average_old / len_old

average_ad = average_ad.reshape(81,97*83)
average_young = average_young.reshape(81,97*83)
average_mid = average_mid.reshape(81,97*83)
average_old = average_old.reshape(81,97*83)


"""存储平均数"""
numpy.savetxt("average_ad.txt", average_ad, fmt=['%s'] * average_ad.shape[1], newline='\r\n')
numpy.savetxt("average_young.txt", average_young, fmt=['%s'] * average_young.shape[1], newline='\r\n')
numpy.savetxt("average_mid.txt", average_mid, fmt=['%s'] * average_mid.shape[1], newline='\r\n')
numpy.savetxt("average_old.txt", average_old, fmt=['%s'] * average_old.shape[1], newline='\r\n')
