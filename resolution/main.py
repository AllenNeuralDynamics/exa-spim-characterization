from numpy import all, asarray, array, where, exp
from pandas import DataFrame
import pandas as pd
from skimage.filters import gaussian as gaussian_filter
from skimage.feature import peak_local_max
from scipy.optimize import curve_fit
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numpy

def compute(im, options):
    beads, maxima, centers, smoothed = getCenters(im, options)
    data=[]
    for i in range(len(beads)):
        data.append(getPSF(beads[i], maxima[i], centers[i], options))
    # data = [getPSF(x, options) for x in beads]
    PSF = pd.concat([i for i in data])
    PSF = PSF.reset_index().drop(['index'],axis=1)
    return PSF, smoothed

def inside(shape, center, window):
    """
    Returns boolean if a center and its window is fully contained
    within the shape of the image on all three axes
    """
    return all([(center[i]-window[i] >= 0) & (center[i]+window[i] <= shape[i]) for i in range(0,3)])

def volume(im, center, window):
    if inside(im.shape, center, window):
        volume = im[(center[0]-window[0])+1:(center[0]+window[0]), (center[1]-window[1])+1:(center[1]+window[1]), (center[2]-window[2])+1:(center[2]+window[2])]
        volume = volume.astype('float64')
        baseline = volume[[0,-1],[0,-1],[0,-1]].mean()
        volume = volume - baseline
        volume = volume/volume.max()
        return volume

def findBeads(im, window, thresh, max_beads):
    smoothed = gaussian_filter(im, 1, output=None, mode='nearest', cval=0, multichannel=None)
    maxval = numpy.max(im[:])
    centers = peak_local_max(im, min_distance=3, threshold_lower=thresh[0], threshold_upper=thresh[1], num_peaks=max_beads, exclude_border=True)
    return centers, smoothed.max(axis=0)

def keepBeads(im, window, centers, options):
    centersM = asarray([[x[0]/options['pxPerUmAx'], x[1]/options['pxPerUmLat'], x[2]/options['pxPerUmLat']] for x in centers])
    centerDists = [nearest(x,centersM) for x in centersM]
    keep = where([x>3 for x in centerDists])
    centers = centers[keep[0],:]
    keep = where([inside(im.shape, x, window) for x in centers])
    return centers[keep[0],:]

def getCenters(im, options):
    window = [options['windowUm'][0]*options['pxPerUmAx'], options['windowUm'][1]*options['pxPerUmLat'], options['windowUm'][2]*options['pxPerUmLat']]
    window = [round(x) for x in window]
    centers, smoothed = findBeads(im, window, options['thresh'], options['max_beads'])
    centers = keepBeads(im, window, centers, options)
    beads = [volume(im, x, window) for x in centers]
    maxima = [im[x[0], x[1], x[2]] for x in centers]
    return beads, maxima, centers, smoothed

def getPSF(bead, maxima, centers, options):
    xProfile, yProfile, zProfile = getProfiles(bead)
    try:
        xFit = fit(xProfile,options['pxPerUmLat'])
        yFit = fit(yProfile,options['pxPerUmLat'])
        zFit = fit(zProfile,options['pxPerUmAx'])
        # data = DataFrame([xFit[4], yFit[4], zFit[4], xFit, yFit, zFit], index = ['FWHMx', 'FWHMy', 'FWHMz', 'xFit', 'yFit', 'zFit']).T
        data = DataFrame([xFit[4], yFit[4], zFit[4], xFit, yFit, zFit, maxima, centers[2], centers[1], centers[0], bead], index = ['FWHMx', 'FWHMy', 'FWHMz', 'xFit', 'yFit', 'zFit', 'Max', 'x_center', 'y_center', 'z_center', 'beads']).T
        return data
    except:
        print('bead could not be fit')

def getProfiles(bead):
    xProfile = numpy.max(numpy.max(bead, axis=1), axis=0)
    yProfile = numpy.max(numpy.max(bead, axis=2), axis=0)
    zProfile = numpy.max(numpy.max(bead, axis=2), axis=1)
    # xProfile = bead[int(bead.shape[0]/2), :, int(bead.shape[2]/2)]
    # yProfile = bead[int(bead.shape[0]/2), int(bead.shape[1]/2), :]
    # zProfile = bead[:, int(bead.shape[1]/2), int(bead.shape[2]/2)]
    return xProfile, yProfile, zProfile

def fit(yRaw,scale):
    y = yRaw - (yRaw[0]+yRaw[-1])/2
    x = (array(range(y.shape[0])) - y.shape[0]/2)
    xFit = numpy.arange(-y.shape[0]/2, y.shape[0]/2, 0.1)
    popt, pcov = curve_fit(gauss, x, y, p0 = [1, 0, 1, 0])
    FWHM = 2.355*popt[2]/scale
    yFit = gauss(xFit, *popt)
    perr = numpy.sqrt(numpy.diag(pcov))
    if perr[2] > FWHM/3.0:
        print('bad bead fit')
    else:
        return x, y, xFit, yFit, FWHM

def plotPSF(x,y,xFit,yFit,FWHM,scale,Max):
    plt.figure(figsize=(8,5));
    plt.plot(xFit.astype(float)/scale,yFit/yFit.max(), lw=3);
    plt.plot(x.astype(float)/scale,y/y.max(),'ok');
    plt.xlim([-x.shape[0]/2/scale, x.shape[0]/2/scale])
    plt.ylim([0, 1.1])
    plt.xlabel('Distance (um)')
    plt.ylabel('Norm. intensity')
    plt.rcParams.update({'font.size': 14})
    plt.annotate('FWHM: %.2f um' % FWHM, xy=(-x.shape[0]/2/scale*0.9, 1), size=14)
    plt.annotate('Max intensity: %.2f' % Max, xy=(-x.shape[0]/2/scale*0.9, 0.9), size=14)


def plotAvg(i):
    plt.figure(figsize=(5,5));
    plt.imshow(average[i], vmin=0, vmax=.9);
    if i==average.shape[0]/2:
        plt.plot(average.shape[1]/2, average.shape[2]/2, 'r.', ms=10);
    plt.xlim([0, average.shape[1]])
    plt.ylim([average.shape[2], 0])
    plt.axis('off');

def plotAvg(i):
    plt.figure(figsize=(5,5));
    plt.imshow(average[i], vmin=0, vmax=.9);
    if i==average.shape[0]/2:
        plt.plot(average.shape[1]/2, average.shape[2]/2, 'r.', ms=10);
    plt.xlim([0, average.shape[1]])
    plt.ylim([average.shape[2], 0])
    plt.axis('off');

def dist(x,y):
    return ((x - y)**2)[1:].sum()**(.5)

def nearest(x,centers):
    z = [dist(x,y) for y in centers if not (x == y).all()]
    return abs(array(z)).min(axis=0)

def gauss(x, a, mu, sigma, b):
    return a*exp(-(x-mu)**2/(2*sigma**2))+b