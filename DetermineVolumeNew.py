""" DetermineVolume.py — Python 3 version

    This program determines the volume of the resonance zone.

    Converted from Python 2 to Python 3 by ChatGPT (2025-11-04)

    
    This program will determine the volume of the resonance zone.
    The user supplies and external text file called "AxialField.txt"
    that has two columns: z-position and Bz with two additional header
    lines defining units.  This file is read to get the axial field and   
    the radial field due to solenoids is determined by simple integration
    of Maxwell's equation delB=0.  The radial field is set by defining
    a radius and a field at that radius.  The model uses a scaled 
    electrostatic sextupole to determine the vector field due to the 
    sextupole inside the source.  These are summed in order to get the total
    field at any point within the source.  The volume calculation is done
    by defining a mesh of points which are checked for being inside the
    resonance zone.  If they are inside, the volume surrounding that mesh
    point is added to total volume.  The volume calculation is improved by
    successively increasing the number of mesh points checked.
    			
    			Damon Todd
    			Thu Jun 19 12:01:58 PDT 2008 
"""

import numpy as np
from math import sqrt, cos, sin, pi, pow

def RunProgram():
    writefile = 0  # Change to 1 to write output file of points inside resonance zone

    # First form all of the arrays necessary to do calculations
    print()
    z, Bz, Bz2, dBz_dz, dBz_dz2, bars, mult = GetNecessaryArrays()
    dzmesh = z[1] - z[0]  # set mesh step size

    # Set the resonance surface field
    Bres = float(input("B_res in tesla: "))

    Btot = np.zeros(4, float)  # Array to store total field
    quitflag = '0'              # variable to determine when to quit
    chang = firsttime = 1
    print()

    while quitflag != 'q':
        # initialize variables to find extrema of point inside resonance zone
        zptmin = 10.0
        zptmax = -10.0
        rptmax = -10.0

        # Set ranges for searching mesh
        chang = 8
        while chang != 0:
            if firsttime != 1:
                print('options... 1:rmax 2:zmin 3:zmax 4:nz 5:nr ')
                chang = int(input('change which (or 0: done changing...run again, or 9: quit)?-> '))
            if firsttime == 1 or chang == 1:
                rmax = float(input("max radius to test for resonance [cm]: ")) / 100.0
            if firsttime == 1 or chang == 2:
                zmin = float(input("minimum z position to search for resonance [cm]: ")) / 100.0
            if firsttime == 1 or chang == 3:
                zmax = float(input("maximum z position to search for resonance [cm]: ")) / 100.0
            if firsttime == 1 or chang == 4:
                nz = int(input("number of steps in z: "))
            if firsttime == 1 or chang == 5:
                nx = ny = int(input("number of steps in x: "))
            if chang == 9:
                return
            if firsttime == 1:
                firsttime = chang = 0

        dz = (zmax - zmin) / float(nz)  # z step size
        dx = dy = 2 * rmax / float(nx)  # x,y step size
        xst = yst = dx / 2.0 - rmax     # x starting position [xst,xst+dx,...,xst+nx*dx]
        zst = dz / 2.0 + zmin           # z starting position [zst,zst+dz,...,zst+nz*dz]

        dvol = dx * dy * dz             # volume of each element
        totalVol = 0.0                  # store total volume inside resonance zone

        if writefile == 1:
            ofile = open('pts.m', 'w')  # open file if writing results
        print(f"{nx*ny*nz} points to check\n")  # print number of points checked

        for i in range(nx):             # loop over mesh points
            for j in range(ny):
                for k in range(nz):
                    xpt = xst + i * dx  # set position being checked
                    ypt = yst + j * dy
                    zpt = zst + k * dz
                    rpt = sqrt(xpt * xpt + ypt * ypt)

                    if rpt < rmax:      # check if point is within checking region
                        Bsext = GetSextField(xpt, ypt, mult, bars)  # first get Sext and Sol fields
                        Bsol = GetSolenoidField(xpt, ypt, zpt, rpt, z, dzmesh, Bz, Bz2, dBz_dz, dBz_dz2)
                        Btot[0] = Bsext[0] + Bsol[0]    # sum x field
                        Btot[1] = Bsext[1] + Bsol[1]    # sum y field
                        Btot[2] = Bsol[2]               # z field due to solenoids only
                        Btot[3] = sqrt(Btot[0]**2 + Btot[1]**2 + Btot[2]**2)

                        if Btot[3] <= Bres:     # if |B| <= resonance value then count it
                            totalVol += dvol    # add on volume of cube to total
                            if writefile == 1:
                                ofile.write(f"{xpt} {ypt} {zpt}\n")
                            rptmax = max(rptmax, rpt)
                            zptmin = min(zptmin, zpt)
                            zptmax = max(zptmax, zpt)

        # After computing volume, give information about step size used...
        print("with dx[cm]=%.4f, dy[cm]=%.4f, dz[cm]=%.4f dvol[cm^3]=%.4f" %
              (dx * 100.0, dy * 100.0, dz * 100.0, dvol * 1e6))
        #  ... and computed volume
        print("total Volume[cm^3]=%.2f" % (totalVol * 1e6))
        # Remind user of defined limits...
        print("\ndefined: zmin[cm]=%.2f, zmax[cm]=%.2f, rmax[cm]=%.2f" %
              (zmin * 100.0, zmax * 100.0, rmax * 100.0))
        # ...and let the user know what the limits of the resonance zone are
        print("actual points: zmin[cm]=%.2f, zmax[cm]=%.2f, rmax[cm]=%.2f" %
              (zptmin * 100.0, zptmax * 100.0, rptmax * 100.0))
        print()
        if writefile == 1:
            ofile.close()


def GetNecessaryArrays():
    # First, get the information for Solenoids
    z, Bz = GetAxialField()     # get z, Bz in m, T
    dBz_dz, Bz2 = makeDerivatives(z, Bz)    # make first, second derivative array
    dBz_dz2 = spline(z, dBz_dz)         #2nd deriv of deriv for splint

    # then get information for sextupoles
    bars = np.zeros((2, 6), float)
    for i in range(6):      # Make sextupole bars at 10 m (far from source)
        bars[0][i] = 10.0 * cos(i * 2.0 * pi / 6.0)
        bars[1][i] = 10.0 * sin(i * 2.0 * pi / 6.0)

    rSextKnown = float(input('At what radius in centimeters is the sextupole field known: ')) / 100.0
    BSextKnown = float(input('What is the field in Tesla at this radius: '))
    Btmp = GetSextField(rSextKnown, 0.0, 1.0, bars)
    mult = BSextKnown / Btmp[0]

    return z, Bz, Bz2, dBz_dz, dBz_dz2, bars, mult


def GetSextField(xpt, ypt, mult, bars):     # get sextupole field
    # mult=lambda/(2*pi)/epsilon_naught, lambda=charge/length
    Bsext = np.zeros(2, float)
    for i in range(6):
        sgn = -1 if (i % 2 == 0) else 1
        len_sq = (xpt - bars[0][i]) ** 2 + (ypt - bars[1][i]) ** 2
        Bsext[0] += sgn * mult * (xpt - bars[0][i]) / len_sq
        Bsext[1] += sgn * mult * (ypt - bars[1][i]) / len_sq
    return Bsext


def GetSolenoidField(xpt, ypt, zpt, rpt, z, dz, Bz, Bz2, dBz_dz, dBz_dz2):      # get sol field
    Bsolenoid = np.zeros(3, float)
    Bsolenoid[2] = splint(z, Bz, Bz2, zpt, dz)
    if rpt != 0:
        Brtmp = -rpt / 2.0 * splint(z, dBz_dz, dBz_dz2, zpt, dz)
        Bsolenoid[0] = xpt * Brtmp / rpt
        Bsolenoid[1] = ypt * Brtmp / rpt
    return Bsolenoid


def makeDerivatives(z, Bz):     # make derivatives necessary for expansion
    dBz_dz = np.zeros_like(Bz)  # define first deriv array
    d2Bz_dz2 = spline(z, Bz)    # make second deriv array
    dz = z[1] - z[0]            # define step size
    dz_sm = 0.001 * dz          # step size for calculating derivative
    for i in range(len(z) - 1):
        dBz_dz[i] = (splint(z, Bz, d2Bz_dz2, z[i] + dz_sm, dz) - Bz[i]) / dz_sm
    dBz_dz[-1] = (Bz[-1] - splint(z, Bz, d2Bz_dz2, z[-1] - dz_sm, dz)) / dz_sm
    return dBz_dz, d2Bz_dz2


def GetAxialField():
    with open('AxialField.txt', 'r') as f:
        lines = f.readlines()

    header_lines = [line for line in lines if line.startswith('#')]
    Nheader = len(header_lines)     # count header lines
    nlines = len(lines)             # count total lines

    Bz = np.zeros(nlines - Nheader - 2, float)  # subtract header and two info lines 
    z = np.zeros(nlines - Nheader - 2, float)   # subtract header and two info lines 

    data_lines = lines[Nheader:]
    cseZ = int(data_lines[0][0])
    cseB = int(data_lines[1][0])

    for i, line in enumerate(data_lines[2:]):
        vals = line.strip().split()
        if len(vals) < 2:
            continue
        z[i] = float(vals[0])
        Bz[i] = float(vals[1])

    if cseZ == 2:
        z = z / 100.0       # change to m
    elif cseZ == 3:
        z = z / 1000.0      # change to m
    if cseB == 2:
        Bz = Bz / 10000.0   # change to Tesla
    return z, Bz


def spline(x, y):
    N = len(x)
    D = np.zeros(N, float)
    B = np.zeros(N, float)
    y2 = np.zeros(N, float)
    dx = x[1] - x[0]
    B[0] = y2[0] = 0.0
    D[0] = 1.0
    y2[1] = dx
    D[1] = 4.0 * dx
    B[1] = 6.0 * (y[2] - 2.0 * y[1] + y[0]) / dx
    for j in range(2, N - 1):
        i = j - 1
        k = j + 1
        B[j] = y2[i] / D[i]
        y2[j] = dx
        D[j] = 4.0 * dx - y2[j] * B[j]
        B[j] = 6.0 * (y[j + 1] - 2.0 * y[j] + y[j - 1]) / dx - B[j - 1] * B[j]

    B[N - 1] = 0.0
    y2[N - 1] = 0.0
    y2[N - 2] = B[N - 2] / D[N - 2]
    for i in range(N - 3, 0, -1):
        y2[i] = (B[i] - y2[i] * y2[i + 1]) / D[i]
    y2[0] = 0.0
    return y2


def splint(x, y, y2, X, dx):
    N = len(x)
    if X < x[0] or X > x[N - 1]:
        print("out\n")
        return 3.14159
    i = int((X - x[0]) / dx) + 1
    if i >= N:
        i = N - 1
    l1 = X - x[i - 1]
    l2 = x[i] - X
    return ((y2[i] * pow(l1, 3) + y2[i - 1] * pow(l2, 3)) / (6.0 * dx) +
            (y[i] / dx - y2[i] * dx / 6.0) * l1 +
            (y[i - 1] / dx - y2[i - 1] * dx / 6.0) * l2)


if __name__ == "__main__":
    RunProgram()