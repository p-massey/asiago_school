import sys
import numpy as np
from astropy.io import fits
from astropy.table import Table
import synphot as S
from matplotlib import pyplot as plt

#specname = sys.argv[1]
#bandname = sys.argv[2]
#print(f'specname = {specname} bandname = {bandname}')


def readspec(fspe):       #########  read fits to ascii ##############

    spec = fits.open(fspe)
    head = spec[0].header
    if len(spec[0].data.shape)==2: fl = spec[0].data[0]
    elif len(spec[0].data.shape)>2: fl = spec[0].data[0,0,:]
    else: fl = spec[0].data
    naxis1 = head['naxis1']
    crpix1 = head['crpix1']
    crval1 = head['crval1']
    cdelt1 = head['cd1_1']
    lam = crval1 + (np.arange(1, naxis1 + 1) - crpix1) * cdelt1
    return lam ,fl

def singlespec_calc(specname,bandname1,bandname2,bandname3):
    if '.fits' in specname: lam,fl = readspec(specname)
    else:
        tab = Table.read(specname,format='ascii.no_header')
        lam,fl = tab['col1'],tab['col2']
    
    vega = S.SourceSpectrum.from_vega()
    sp = S.SourceSpectrum(S.Empirical1D,points=lam,lookup_table=fl*S.units.FLAM,
                          keep_neg=True)
    
    if '.dat' in bandname1: 
        bp1 = S.SpectralElement.from_file(bandname1)
    else:
        bp1 = S.SpectralElement.from_filter(bandname1)
        
    if '.dat' in bandname2: 
        bp2 = S.SpectralElement.from_file(bandname2)
    else:
        bp2 = S.SpectralElement.from_filter(bandname2)
    
    if '.dat' in bandname3: 
        bp3 = S.SpectralElement.from_file(bandname3)
    else:
        bp3 = S.SpectralElement.from_filter(bandname3)
    #sp = S.SourceSpectrum(sp,z=0.6)

    obs1 = S.Observation(sp,bp1,force='extrap')
    obs2 = S.Observation(sp,bp2,force='extrap')
    obs3 = S.Observation(sp,bp3,force='extrap')
    
    print(nome_input)
    #print(f"{obs.effstim(flux_unit='vegamag',vegaspec=vega):.2f}")
    print(f"{obs1.effstim(flux_unit='vegamag',vegaspec=vega):.2f}",f"{obs2.effstim(flux_unit='vegamag',vegaspec=vega):.2f}",f"{obs3.effstim(flux_unit='vegamag',vegaspec=vega):.2f}")
    #print(f"{obs2.effstim(flux_unit='abmag',vegaspec=vega):.2f}")
    #print(f"{obs3.effstim(flux_unit='abmag',vegaspec=vega):.2f}")
    
    bp1.plot()
    plt.plot(sp.waveset,sp(sp.waveset)/np.max(sp(sp.waveset)))
    plt.plot(bp2.waveset,bp2(bp2.waveset)/np.max(bp2(bp2.waveset)))
    plt.plot(bp3.waveset,bp3(bp3.waveset)/np.max(bp3(bp3.waveset)))
    plt.show()

#filename = 'lista' # <<<<<<<<<<<<<<<<<<< NOME FILE .DAT
#data = np.genfromtxt(filename)

#f=open('lista',"r")
#lines=f.readlines()
#nomi=[]
#for x in lines:
#    nomi.append(x.split())
#f.close()

#print(nomi)
#print(type(nomi[0][0]))

#nome_input = nomi[7][0]
nome_input = 'AF1024452.a.fits'

singlespec_calc(nome_input,'B.dat','V.dat','I.dat')

#for i in range (len(nomi)):
#	singlespec_calc(nomi[i][0],'gprime.dat','rprime.dat','iprime.dat')


