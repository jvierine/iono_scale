# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from PIL import Image
from datetime import datetime
import argparse
import os
import sys
import numpy as np
import gzip

def deffn():
    # Default file name, data file for testing.
    return 'SO166GRAM_20161123_100000.166'

def genPath(kin, day, modtime=None, prefix='/Volumes/Archive', create=False):
    '''genpath(kin, day, modtime, prefix='/Volumes/Archive', create=False)

    function to generate the rather lengthy paths to ionosonde data.

    kin: Ftxt, Fmat, Fpng, Feps, G, A, I or P;
    day: datetime object

    (c)2016 by Thomas Ulich, SGO, Finland.
    '''

    if type(day) is not datetime:
        print('day must be datetime object.')
        return None

    # Make sure, kin is upper case
    kin=kin.upper()
    # Define a standard dummy path and file name
    p=prefix+day.strftime('/%Y/xyz_%Y/xyz_%Y%m/xyz_%Y%m%d')
    fn=day.strftime('xyz_%Y%m%d_%H%M%S.zyx')
    # Go through different types of directory paths
    if kin=='G' or kin=='GRAM':
        pref='SO166GRAM'
        suff='.166'
    elif kin=='I' or kin=='IMG':
        pref='SO166IMG'
        suff='.jpg'
    elif kin=='R' or kin=='ION':
        pref='ION'
        suff=''
    elif kin=='P' or kin=='PARA':
        pref='SO166PARA'
        suff='.mat'
        print('Modification time not implemented yet!')
    # Replace strings as defined above
    p=p.replace('xyz', pref)
    fn=fn.replace('xyz', pref).replace('.zyx', suff)
    # Create directory if it does not exist, exist_ok option ensures
    # that no error is raised when it exists already
    if create:
        os.makedirs(p, exist_ok=True)
    # Return path and file name
    return (p, fn)


def loadHeader(fn, offset=0):
    '''loadHeader(filename, offset=0)

    Loads ionogram files of the SGO *.166 type.
    Returns file headers as dictionary.

    (c)2016 by Thomas Ulich, SGO, Finland.
    '''

    # Open the GRAM file and read the header information
    with open(fn, 'rt', encoding='latin-1') as f:
        # First line contains four numbers:
        # Length of Header minus first row, Length of Data,
        # Number of Newlines before next ionogram, if more than one in file,
        # Sum of numbers 1-3, just for checking.
        #
        # Read first line and strip of newline.
        L=f.readline().rstrip()
        # Convert to integers
        N=[int(x) for x in L.split(" ")]
        # Check if last number is sum of first three
#        if np.sum(N[0:3]) == N[-1]:
 #           print("--- Header ok.")
  #      else:
   #         print("--- Header does not check out. File corrupt.")
        # Find out where we are in the file, i.e. length of first line.
        # This is where the actual header starts.
        Hi=f.tell()
        # Update N[0] to the correct (complete) length of header
        N[0]=N[0]+Hi
        # Create Header dictionary for all variables in the file, recording
        # first line. This replaces Matlab's "struct"
        HDR={'N': N, 'fname': fn}
        # Read until end of header while file position < header length
        while f.tell() < N[0]:
            # Read next line from header
            L = f.readline().rstrip().rstrip(';')
            # Ignore if it's a comment, indicated by "%"
            if L[0] != '%':
                # Split a variable assignment at the "="
                key, val=tuple(L.split(sep='=', maxsplit=1))
                if key == 'time_':
                    val=datetime.strptime(val, '[ %Y %m %d %H %M %S ]')
                # Strip of quotes and brackets from beginning and end.
                elif val[0] == "'":
                    val = val[1:-1]
                elif val[0] == '[':
                    # Some values must be integers, others are float.
                    if key in [ 'rowcol', 'nbits', 'fft_', 'fatal_', 'time_' ]:
                        numtype=int
                    else:
                        numtype=float
                    # Convert strings to numbers
                    val = [numtype(x) for x in val[1:-1].split()]

                # Add key-value pair to data dictionary, strip trailing "_"
                HDR[key.rstrip('_')]=val
    return HDR

def loadData(HDR):
    '''loadData(ionogram_header)

    Reads the data portion of an ionogram. Requires technical information
    obtained by loadHeader, i.e. requires header of ionogram.

    Returns complete header plus ionogram matrix as dictionary.'''

    # Extract variables from header dictionary for easier access
    N=HDR['N']

    # Open the GRAM file for binary reading the data block
    with open(HDR['fname'], 'rb') as f:
        # Go to the beginning of data block (after header)
        f.seek(N[0], 0)
        # Read entire data block as bytes
        d=np.fromfile(f, dtype='uint8', count=N[1])

    if 'gunzip' in HDR['uncompressor']:
        # Data section if gzip'd, let's uncompress it.
        q=gzip.decompress(d)
    else:
        print('Unknown data compression.')

    # Data are read above as q and developed in steps, which are
    # explicit without overwriting for debugging purposes. Thus
    # sequence of data variables is: q, w, e, r, t. t is final.
    #
    # Check correct data type from header
    if HDR['dtype'] == 'uint16':
        dtype='<u2'
    else:
        print('Unknown data type:', HDR['dtype'])
    # Convert into little-endian uint16 (all GRAM are LE)
    w=np.fromstring(q, dtype=dtype)
    # Reshape using dimensions of original matrix from Header.
    # Note that the original matrix was Matlab, so order 'F' is important.
    e=w.reshape(tuple(HDR['rowcol']), order='F')
    # Scale values back to correct range.
    # Make sure to divide first in order to convert to float,
    # otherwise the first multiplication exceeds the limits of
    # uint16 and the result will be wrong.
    r=(110-19) * (e/(2**16-1))
    # Add 19 to all non-zero values
    t=np.where(r>0, r+19, r)
    # Add data matrix to header HDR and return complete header
    HDR['data']=t
    return HDR

def loadGram(fn=None, offset=0):
    '''loadGram(fn=None, offset=0)

    Load a complete ionogram data set.  Shorthand for running
    loadHeader and loadData consecutively.'''

    # For debugging and testing, just load some ionogram as
    # defined above.
    if fn is None:
        fn = deffn()
    h = loadHeader(fn, offset)
    d = loadData(h)
    return d

def plotGram(GRM, ax=None, maxfactor=0, fontsize=11):
    '''genGram(ionogram, ax=plt.gca, maxfactor=0, fontsize=11)

    Generates an ionogram in standard form and returns handles to all
    relevant elements. This can be run, e.g., in a loop over several
    subplots.  By default, it plots in the current axis.

    (c)2016 by Thomas Ulich, SGO, Finland.
    '''
    # If ax not specified, use current axis.
    if ax is None:
        ax = plt.gca()
    # Some housekeeping before we start to plot.
    # Find maximum value in data:
    CMax = np.max(GRM['data'])
    # Generate XLabel and CLims (VLims) accordingly
    if maxfactor==0:
        vmin=GRM['clim'][0]
        vmax=GRM['clim'][1]
        xlabel='Frequency [MHz]'
    else:
        vmin=20
        vmax=maxfactor*CMax
        xlabel='Frequency [MHz]   ---   max = {:2.0f} dB; '.format(CMax)
        xlabel=xlabel+'scale = {:2.0f} dB'.format(maxfactor*CMax)

    # Generate a color map, which sets values below CLim range to white.
    mymap = copy.copy(mpl.cm.get_cmap('jet'))
    mymap.set_under('w')
    # Plot data
    # If we want CLims from header:  vmin=GRM['clim'][0], vmax=GRM['clim'][1],
    ax.imshow(GRM['data'], origin='lower',
              extent=GRM['xlim']+GRM['ylim'], aspect='auto',
              vmin=vmin, vmax=vmax, interpolation='nearest',
              cmap=mymap)
    # Add labels
    xlh=ax.set_xlabel(xlabel, fontsize=fontsize)
    ylh=ax.set_ylabel('Virtual height [km]', fontsize=fontsize)
    th=ax.set_title(GRM['title'], fontsize=fontsize,
                    verticalalignment='bottom',
                    position=(0.5,1.015))
    #return locals()
    # Sort out major and minor ticks
    xmajorLocator  = MultipleLocator(1)
    xminorLocator  = MultipleLocator(0.5)
    ymajorLocator  = MultipleLocator(100)
    yminorLocator  = MultipleLocator(50)
    majorFormatter = FormatStrFormatter('%d')
    # Format Major ticks
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.xaxis.set_major_formatter(majorFormatter)
    ax.yaxis.set_major_formatter(majorFormatter)
    # For the minor ticks, use no labels; default NullFormatter
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    # Adjust length and direction of tick marks
    ax.tick_params(axis='both', which='both', direction='out', labelsize=10)
    ax.tick_params(axis='both', which='minor', length=4)
    ax.tick_params(axis='both', which='major', length=6)

    # Create a dictionary for all values to be returned.
    handles={}
    for var in ['ax', 'xlh', 'ylh', 'th',
                'mymap', 'vmin', 'vmax', 'CMax', 'maxfactor']:
        handles[var]=locals()[var]

    return handles


def webGram(GRM, maxfactor=0):
    '''ionoplot(ionogram, maxfactor=0)

    Plots an ionogram in the standard JPG fashion.
    If I could only remember what maxfactor was for...?

    (c)2016 by Thomas Ulich, SGO, Finland.
    '''

    # Default font size for default ionogram format
    fontsize=11
    # Create figure window
    fh, ax = plt.subplots()
    # Generate ionogram in new axis
    handles=plotGram(GRM, ax, maxfactor, fontsize=fontsize)

    # Fix axes position
    ax.set_position([0.1, 0.1, 0.865, 0.833])

    # Export everything we need to know about the plot
    # Add figure handle to already existing handles
    handles['fh']=fh
    # Return dictionary with relevant handles
    return handles


def webPict(pname, fh, dpi):
    '''webPict(pname, fh, dpi)

    Dump ionogram to disk as jpg image.

    '''
    print(pname, fh, dpi)
    # save ionogram as PNG (matplotlib doesn't do jpg well)
    fh.savefig('ionogram.png', dpi=dpi)
    # convert PNG to JPG and adjust quality
    Image.open('ionogram.png').save(pname, 'JPEG', quality=60)
    # remove temporary ionogram PNG file
    os.remove('ionogram.png')
    return None


def fourGrams():
    # For testing with subplots
    G=loadGram()
    ax=[]
    h=[]
    fh=plt.figure(figsize=(24,18), dpi=80)
    for k in range(4):
        ax.append(plt.subplot(2, 2, k+1))
        h.append(plotGram(G, ax=ax[k], maxfactor=.9))
    fh.savefig('test.png')
    fh.savefig('test.pdf')
    return fh


def parseCommandLine():
    # Define parser for command line arguments
    parser = argparse.ArgumentParser(description='''Plots ionograms
         when supplied with an ionogram timestamp.''',
                   epilog="(c)2016 by Th.Ulich, SGO, Finland.")
    parser.add_argument("-t", "--time", dest="t", type=str,
                        help='''time of ionogram or start of time interval,
                                format: YYYYMMDD_HHMMSS''')
    duration_group = parser.add_mutually_exclusive_group()
    duration_group.add_argument("-e", "--end", dest="e", type=str,
                        help='''end of time interval, format YYYYMMDD_HHMMSS;
                                conflicts with -d''')
    duration_group.add_argument("-d", "--duration", dest="d", type=int,
                        help='''time interval duration in hours;
                                conflicts with -e''')
    parser.add_argument("-s", "--step", dest="s", type=int,
                        help='time step in minutes, default every minute')
    parser.add_argument("-k", "--kind", dest="k", type=str, default='standard',
                        help='one of standard (default), latest, paper')
    parser.add_argument("-c", "--copy", dest="c", action='store_false',
                        help='''copy ionogram data files to directory
                                specified by -o''')
    parser.add_argument("-p", "--prefix", dest="p", type=str,
                        default='/Volumes/Archive',
                        help='''Root of the data archive from which to retrieve
                                ionograms. Default: /Volumes/Archive''')
    parser.add_argument("-o", "--output", dest="o", type=str,
                        help='''output, specify filename (for single ionogram
                                request (default depends on context) or
                                directory for multiple output files
                                (default is current directory)''')
    # Return command line arguments
    return parser.parse_args()


def getTimeStampArgument(ts):
    '''getTimeStampArgument(ts)

    Takes string as given by command line argument, performs some basic
    consistency checks, and returns to a datetime object.'''

    if len(ts)<9:
        sys.exit('ERROR: Incomprehensible timestamp provided: ', ts)
    ts=ts+'0'*(len('YYYYMMDD_HHMMSS')-len(ts))
    return datetime.strptime(ts, '%Y%m%d_%H%M%S')


if __name__ == '__main__':

    # Command Line Arguments
    args = parseCommandLine()

    ## PREFIX
    # Define data root, default is /Volumes/Archive
    # Return with error if prefix does not exist, never create it.
    prefix = args.p
    if not os.path.isdir(prefix):
        sys.exit('ERROR: Prefix directory does not exist: {}'.format(prefix))

    ## TIME STAMP (START TIME)
    if args.t:
        t0 = getTimeStampArgument(args.t)
    else:
        parser.print_help()
        sys.exit('ERROR: No time specified. Stopping.')

    ## TIME STAMP (END TIME)
    if args.e:
        e = getTimeStampArgument(args.e)

    ## DURATION (in hours)
    if args.d:
        e = t0 + timedelta(hours=args.d)
    else:
        # if neither e nor d are specified, end time defaults to start time
        # and only one ionogram will be processed.
        e = t0

    # Finally, make sure end time is larger than start time.
    if e > s:
        s, e = e, s

    ## OUTPUT
    if args.o:
        if s < e:
            outdir = args.o
            os.makedirs(outdir, exist_ok=True)
        else:
            outfile = args.o

    ## LOOP OVER TIMES
    t = t0
    while t <= e:
        pn, fn = genPath('G', t, prefix=prefix, create=False)
        gram = loadGram(pn+'/'+fn)
        hdr = plotGram(gram, .9)
        if args.k == 'standard':
            pn, fn = genPath('I', t, prefix=prefix, create=True)



    ## KIND
    if args.k:
        H = webGram(G, .9)
        if args.k == 'standard':
            webPict(fn, H['fh'], 80)
        elif args.k == 'latest':
            fn='latest.jpg'
            webPict(fn, H['fh'], 80)
        elif args.k == 'paper':
            pn, fn = genPath('I', t, prefix=prefix)
            fn=fn.replace('.jpg', '.pdf')
            #fh.set_size_inches((11.69, 8.27))
            H['fh'].savefig(fn, dpi=80, orientation='landscape', papertype='a4')
            print('Saving plot as {}'.format(fn))
        else:
            parser.print_help()
            sys.exit('ERROR: Unknown option "{}" for kind.'.format(args.k))


#pn, fn = genPath('G', t, prefix=prefix)
#G = loadGram(pn+'/'+fn)






