import numpy as np
import os, glob
#from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt
from pylab import figure, show
#04/27/15 Corrected efficiency below 60 keV. Below 10 keV correction is not correct. Ignore data below 10 keV.

#-----------------------------------------------------------------------------------

def ReadData(name):                                             # read value of livetime, total # lines, return channels, counts
    f = open(name, 'r')
    nlines = 0                                                  # nlines = total number of lines in file
    nDlines = 0                                                  
    while f.readline():
        nlines += 1
        #print(f.read())                                        # check to see it's reading exactly what's in the .mca file
    #print('total # lines in file =', nlines)

    f.seek(0)                                                   # set readline pointer back to 0

    for i in range(nlines):
        line = f.readline().split()
        if line[0] == 'LIVE_TIME':                              # finds and assigns livetime from the 3-element livetime list
            livetime = float(line[2])
        if line[0] == '<<DATA>>':                               # could use match case instead of multiple ifs
            Dlocation = i                                       # Dlocation = data starts after this line
            nDlines = nlines - 1 - Dlocation - 1
            break                                               # break stops readline where it found the '<<DATA>>'
    #print(livetime, nDlines, nlines, Dlocation)        # THIS WORKS        # check against .mca

    D = []                                                      # raw data list
    for i in range(nDlines):
        Ds = f.readline().split()                               # starts reading lines again at the beginning of the data
        if Ds[0] == '<<END>>':
            endpoint = i                                        # end of data ('<<END>>')
            break
        D.append(int(Ds[0]))                                    # turns bin data into list
    nDlines = len(D)                                            # nDlines = number of channels with data (1024)
    f.close()
    return(livetime, nDlines, D)                                # THIS WORKS

#-----------------------------------------------------------------------------------

# Function to open, normalize, calibrate, correct for detector efficiency,  X-Ray & Background Data
def GetxbgFile(xbgname, livetime1):
    for i in range(xbgnDlines):                                 # divide data by livetime, now in counts/s (normalized)
        xbgD[i] = xbgD[i] / livetime1                           # nDlines should always equal 1024
        if xbgD[i] <= 0:
            xbgD[i] = 0.0001

    aCal = 0.16594626                                           # CHECK GAIN
    bCal = -0.4991754892        
    #print('Energy Calibration = ' +  str(aCal) + '*Channel + ' + str(bCal))

    E = []                                                      # list for calibrated energy spectrum
    for i in range(xbgnDlines):
        energy = aCal*i + bCal
        E.append(energy)
    #print('E:',E)

    fig = figure(facecolor = 'w')                               # plot normalized and calibrated x-ray spectrum 
    ax = fig.add_subplot(111, frame_on = True, facecolor = 'darkseagreen')
    #ax.step(E, xbgD, where = 'pre', color = 'k')
    ax.semilogy(E, xbgD, linestyle = '-', color = 'black')
    ax.set_xlim(0, 300)
    ax.set_ylim(0.01, 10)
    ax.set_xlabel('Energy (keV)', color = 'black')
    ax.set_ylabel('Counts/s', color = 'black')
    ax.set_title(xbgname,'Calibrated X-Ray Spectrum')
    show()
    
    print('Accounting for Efficiencies...')                     # correct for detector efficiencies           # if NO, will throw error on Spectral Temp calc                       
    parameters, covariance = curve_fit(LSpoly3, EffEnergy, EffAbs) # call Least-Squares 3rd-Order Polynomial Fit for Scintillator Efficiency function
    fit_a = parameters[0]
    fit_b = parameters[1]
    fit_c = parameters[2]
    fit_d = parameters[3]
    #print('a, b, c, d = ', fit_a, fit_b, fit_c, fit_d)

    fit = []                                                    # y = a + bx + cx^2 + dx^3
    Effxray = []                                                # this fits a 3rd-order polynomial to the efficiency curve (could try higher-order or spline fit)
    diff = []
    for i in range(len(EffEnergy)):
        fits = fit_a + fit_b*EffEnergy[i] + fit_c*EffEnergy[i]*EffEnergy[i] + fit_d*EffEnergy[i]*EffEnergy[i]*EffEnergy[i]
        fit.append(fits)
    for i in range(len(E)):                                     # E is the detector efficiency-corrected energies for each dataset
        if E[i] < 50 and E[i] > 10:                             # for 10 < E < 50 MeV, efficiency is 100%
            Effxrays = 1.0
            Effxray.append(Effxrays)
        else:
            Effxrays = fit_a + fit_b*E[i] + fit_c*E[i]*E[i] + fit_d*E[i]*E[i]*E[i]
            Effxray.append(Effxrays)
    #print('Effxray:',Effxray)

    fig = figure(facecolor = 'lightpink')                   # efficiency plot
    ax = fig.add_subplot(111, frame_on = True, facecolor = 'lightpink')
    ax.plot(EffEnergy, EffAbs, linestyle = '-', color = 'black', marker = 'o', markersize = 8, label = 'Scintillator Efficiency')
    ax.plot(EffEnergy, fit, linestyle = '-', color = 'darkseagreen', marker = 'o', markersize = 8, label = 'Ideal LSPoly3rd Order Fit')
    ax.plot(E, Effxray, linestyle = '-', color = 'white', marker = 'o', markersize = 5, label = 'Effective X-Ray Spectrum')     # is this right? in sensitive region (>50 keV)
    ax.set_xlim(0, 410)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Energy (keV)', color = 'black')
    ax.set_ylabel('Total Absorption', color = 'black')
    ax.set_title('Efficiency')
    ax.legend()
    #show()

    CorrectedxbgD = []
    nE = []                                                 # number of counts * energy
    for i in range(len(E)):
        CorrectedxbgD.append(xbgD[i]/Effxray[i])            # correction applied (normalized data divided by fitted 3rd-order polynomial)
        nE.append(CorrectedxbgD[i]*E[i])
    #print(E[i], CorrectedxbgD[i], nE[i])        

    fig = figure(facecolor = 'darkseagreen')                        # efficiency plot
    ax = fig.add_subplot(111, frame_on = True, facecolor = 'white')
    ax.plot(E, xbgD, linestyle = '-', color = 'black', label = 'Original')
    ax.plot(E, CorrectedxbgD, linestyle = '-', color = 'lightcoral', label = 'Corrected')
    ax.set_xlim(0, max(E))
    ax.set_ylim(0.01, max(CorrectedxbgD))
    ax.set_xlabel('Energy (keV)', color = 'black')
    ax.set_ylabel('Counts/s', color = 'black')
    ax.set_title(xbgname)
    ax.legend()
    show()

    writename = "Corrected" + xbgname                       # create output file
    wr = open(writename, 'w')
    for i in range(len(E)):
        wr.write("%f %f %f %f\n"%(E[i], xbgD[i], CorrectedxbgD[i], nE[i])) # %f is replaced with the arguments
    wr.write("\n")

    f, (ax1, ax2, ax3) = plt.subplots(3, sharex = True, sharey = False, facecolor = 'darkseagreen') # nE vs. E
    ax = fig.add_subplot(111, frame_on = True, facecolor = 'white')
    ax1.plot(E, xbgD, linestyle = '-', color = 'black', label = 'Original')
    ax1.set_xlim(10, max(E))
    ax1.set_ylim(0.01, max(xbgD))
    ax2.plot(E, CorrectedxbgD, linestyle = '-', color = 'darkseagreen', label = 'Corrected')
    ax2.set_xlim(10, max(E))
    ax2.set_ylim(0.01, max(CorrectedxbgD))
    ax3.plot(E, xbgD, linestyle = '-', color = 'black', label = 'Original')
    ax3.plot(E, CorrectedxbgD, linestyle = '-', color = 'darkseagreen', label = 'Corrected')
    ax3.set_xlim(10, max(E))
    ax3.set_ylim(0.01, max(CorrectedxbgD))
    ax3.set_xlabel('Energy (keV)', color = 'black')
    ax1.set_ylabel('Counts/s', color = 'black')
    ax2.set_ylabel('Counts/s', color = 'black')
    ax3.set_ylabel('Counts/s', color = 'black')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax1.set_title(xbgname)
    #show()

    print('Calculating Spectral Temperature...')
    beginE = float(input("Enter beginning Energy (keV): "))
    endE = float(input("Enter ending Energy (keV): "))
    for i in range(len(E)):
        if beginE < E[i]:
            beginlocation = i
            break
    for i in range(len(E)):
        if endE < E[i]:
            endlocation = i
            break

    TempxD = []
    TempE = []
    SumCounts = 0.
    for i in range(beginlocation, endlocation):
        #print(E[i], xbgD[i], CorrectedxbgD[i], np.log(CorrectedxbgD[i]))
        TempxD.append(np.log(CorrectedxbgD[i]))
        TempE.append(E[i])
        SumCounts = SumCounts + CorrectedxbgD[i]

    result = linregress(TempE, TempxD)                          # use a linear fit to get the slope for the spectral temperature
    specT = abs(1.0 / result.slope)
    print('a =', result.intercept, 'b =', result.slope, 'Ts =', specT, 'error =', result.intercept_stderr)

    stdDev = np.sqrt(SumCounts/livetimes)
    #print('Ts = ', specT)
    wr.write("Ts ")
    wr.write("%f \n"%(specT))
    wr.write("\n")
    print('Sum of Counts in Range', SumCounts, '+/-', stdDev)
    wr.write("Sum in Range ")
    wr.write("%f \n"%(SumCounts))
    wr.write("\n")
    return()

#-----------------------------------------------------------------------------------

def LSpoly3(x, a, b, c, d):                                     # least-squares 3rd-order polynomial fit
    y = a + b*x + c*x*x + d*x*x*x                               # y = a + bx + cx^2 + dx^3
    return y

#-----------------------------------------------------------------------------------

EffEnergy = []
EffAbs = []
EffEnergy = [51.2, 54.7, 58.5, 62.5, 66.8, 71.5, 76.4, 81.7, 87.3, 93.3, 99.8, 107, 114, 122, 130, 139, 149, 159, 170, 182, 194, 208, 222, 237, 254, 271, 290, 310, 332, 354, 379, 405]
EffAbs = [0.994, 0.989, 0.98, 0.963, 0.938, 0.903, 0.859, 0.807, 0.747, 0.684, 0.619, 0.556, 0.495, 0.439, 0.387, 0.34, 0.299, 0.263, 0.231, 0.203, 0.18, 0.159, 0.142, 0.127, 0.114, 0.103, 0.0937, 0.0856, 0.0787, 0.0726, 0.0674, 0.0628]
    # SCINTILLATOR EFFICIENCY: probability that a photon will be completely absorbed.
    # total attenuation 1000micron with 4mil Be window CdTe from Amptek website

fig = figure(facecolor = 'palevioletred')                   # plot efficiency
ax = fig.add_subplot(111, frame_on = True, facecolor = 'palevioletred')
ax.plot(EffEnergy, EffAbs, linestyle = '-', color = 'black', marker = 'o', markersize = 8)
ax.set_xlim(0, 410)
ax.set_ylim(0, 1)
ax.set_xlabel('Energy (keV)', color = 'black')
ax.set_ylabel('Total Absorption', color = 'black')          # is this an accurate description?
ax.set_title('Scintillator Efficiency')
#show()

xbgnames = []                                                   # empty list, to fill with .mca file names
for filename in glob.glob('*.mca'):
    with open(os.path.join(os.getcwd(), filename), 'r') as f:   # open in read-only mode
        livetimes, xbgnDlines, xbgD = ReadData(f.name)
        xbgnames.append(filename)
        GetxbgFile(filename, livetimes)
print(xbgnames)