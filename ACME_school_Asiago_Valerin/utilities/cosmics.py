import sys,os
from astropy.io import fits
import argparse
import ccdproc
from photutils import detect_sources

if __name__ == '__main__':
    parser = argparse.ArgumentParser(\
         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("imglist",help="file name or @filelist or '*.fits' ")
    parser.add_argument("--sigmaclip",dest="sigmaclip",default=4.5,
                        type=float,help='sigmaclip')
    parser.add_argument("--satlim",dest="satlim",default=65000.,
                        type=float,help='saturation limit')
    parser.add_argument("--gain",dest="gain",default=1.0,
          type=float,help='gain')
    parser.add_argument("--ron",dest="ron",default=5.0,
          type=float,help='ron')
    parser.add_argument("--niter",dest="niter",default=3,
          type=float,help='number of iteration')
    args = parser.parse_args()

def cosmics(img,args):
    
    if '.fits' in img: img = img[:-5]
    print('processing image',img)
    hdulist = fits.open(img+'.fits')
    pixels = hdulist[0].data
    detcosm,mask =  ccdproc.cosmicray_lacosmic(pixels, niter=args.niter,
       sigclip = args.sigmaclip,  sigfrac=0.3, objlim = 5.0,
       satlevel=args.satlim,gain=args.gain,readnoise=args.ron,verbose=True)
    detcosm /= args.gain

    fits.writeto(img+'_clean.fits',detcosm.value,hdulist[0].header,
                 overwrite=True)

    print(f'>>> done {img}_clean.fits')


def read_imglist(imglist):

    if imglist[0]=='@':             # read file list iraf format
        with open(imglist[1:]) as ff: ll = ff.readlines()
        _imglist = []
        for r in ll:
           if len(r.strip()) ==0 :continue
           elif r[0] =='#': continue
           _imglist.append(r.split()[0])
    elif '*' in imglist: 
        _imglist = sorted(glob.glob(imglist))
    else: _imglist = imglist.split(',')
    _imglist = [img.replace('.fits','') for img in _imglist]
    
    return _imglist

if __name__=='__main__':
    
    imglist = read_imglist(args.imglist)
    for img in imglist: cosmics(img,args)
