# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 08:07:01 2013

@author: ispielma

IgorBin: module to load Igor Binary files

Provides

LoadIBW: Function to load igor binary files into numpy arrays

--------- Version History --------- 

IgorBin v1.0: IBS
    initial version, seems to work, limited data types and does not pull
    scaling information.  Because this is legacy support code, I have added
    only those features which are currently required to open the igor binary
    files that are saved in our experiments.  This includes the data array and
    the wave note.

IgorBin v1.1: IBS 
    updated the error handeling to use structures like

        msg = "LoadIBW Error: File " + filename + ", endian byte not readable";            
        raise RuntimeError(msg);
    
    Expanded function text description
"""

import struct
import numpy

def LoadIBW(filename):
    """
    Loads an igor binary file

    filename : the filename including path to open
    
    returns a dictionary:
        {"Note": NoteString, "Data": DataOut}
    
    NoteString is the note from Igor
    Data is a ndarray containing the data
    """
    with open(filename, "rb") as f:
        
        # Endianness
        # endian is zero big-endian byte ordering, otherwise little-endian byte ordering
        try:
            Endian = ord(f.read(1));
        except:
            msg = "LoadIBW Error: File " + filename + ", endian byte not readable";            
            raise RuntimeError(msg);
            return None;
        
        # In the struct.upack()'s that follow this will define the endian order
        EndianChar = '>' if (Endian==0) else '<'
        
        # Version
        # now reset file pointer (the endian information is partally encoded in the two-byte version)
        try:
            f.seek(0);
            VersionTuple = struct.unpack(EndianChar+"h",f.read(2));
            Version = VersionTuple[0];
        except:
            msg = "LoadIBW Error: File " + filename + ", version code not readable";            
            raise RuntimeError(msg);
            return None;
      
        # Full Header
        # reset the pointer (the version will be included in the header code)
       
        try:
            f.seek(0);
            if Version == 1:
                BinHeaderString = f.read(8);
                WaveHeaderString = f.read(110);
                BinHeaderTuple = struct.unpack(EndianChar+"2xl2x",BinHeaderString);
                WaveHeaderTuple = struct.unpack(EndianChar+"h40xl64x",WaveHeaderString);
                NoteSize=0;
                FormulaSize=0;
                Type = WaveHeaderTuple[0];
                Size = WaveHeaderTuple[1];
                Dimensions = Size,0,0,0;
                
            elif Version == 2:
                BinHeaderString = f.read(16);
                WaveHeaderString = f.read(110);
                BinHeaderTuple = struct.unpack(EndianChar+"2xll6x",BinHeaderString);
                WaveHeaderTuple = struct.unpack(EndianChar+"h40xl64x",WaveHeaderString);
                NoteSize=BinHeaderTuple[1];
                FormulaSize = 0;
                Type = WaveHeaderTuple[0];
                Size = WaveHeaderTuple[1];
                Dimensions = Size,0,0,0;
                
            elif Version == 3:
                BinHeaderString = f.read(20);
                WaveHeaderString = f.read(110);
                BinHeaderTuple = struct.unpack(EndianChar+"2xlll",BinHeaderString);
                WaveHeaderTuple = struct.unpack(EndianChar+"h40xl64x",WaveHeaderString);
                NoteSize=BinHeaderTuple[1];
                FormulaSize=BinHeaderTuple[2];
                Type = WaveHeaderTuple[0];
                Size = WaveHeaderTuple[1];
                Dimensions = Size,0,0,0;

            elif Version == 5:
                BinHeaderString = f.read(64);
                WaveHeaderString = f.read(320);
                BinHeaderTuple = struct.unpack(EndianChar+"4x4l44x",BinHeaderString);
                WaveHeaderTuple = struct.unpack(EndianChar+"12xlh50x4l236x",WaveHeaderString);
                NoteSize=BinHeaderTuple[2];
                FormulaSize=BinHeaderTuple[1];
                Type = WaveHeaderTuple[1];
                Size = WaveHeaderTuple[0];
                Dimensions = WaveHeaderTuple[2:6];
            else:
                msg = "LoadIBW Error: File " + filename + ", as an invalid version code" + Version;            
                raise RuntimeError(msg);
                return None;
        except:
            msg = "LoadIBW Error: File " + filename + ", Bin or Wave header invalid";            
            raise RuntimeError(msg);
            return None;
                
        # Load Data
        # reset the pointer (the version will be included in the header code)
        # Complex not supported yet
        if (Type == 0x02):
            TypeString = EndianChar + "f";
        elif (Type == 0x04):
            TypeString = EndianChar + "d";
        elif (Type == 0x08): # Signed Char
            TypeString = EndianChar + "b";
        elif (Type == 0x48): # Unsigned Char
            TypeString = EndianChar + "B";
        elif (Type == 0x10): # Signed Int
            TypeString = EndianChar + "h";
        elif (Type == 0x50): # Unsigned Int
            TypeString = EndianChar + "H";
        elif (Type == 0x20): # Signed Long
            TypeString = EndianChar + "l";
        elif (Type == 0x60): # Unsigned Long
            TypeString = EndianChar + "L";
        else:
            msg = "LoadIBW Error: File " + filename + ", unsupported data type" + Type;            
            raise RuntimeError(msg);
            return None;
           
        dt = numpy.dtype(TypeString);
        try:
            Data = numpy.fromfile(f,dtype=dt,count=Size)
        except:
            msg = "LoadIBW Error: File " + filename + ", data block read failed" + Type;            
            raise RuntimeError(msg);
            return None;
            
        Dims = tuple(y for y in reversed(Dimensions) if y);
        DataOut = numpy.reshape(Data, Dims);      
        
        # Read the wave note if possible
        try:
            if Version == 1:
                NoteString = "";         
            elif Version == 2:
                f.read(16); # padding
            elif Version == 3:
                f.read(16); # padding
            elif Version == 5:
                f.read(FormulaSize); # skip formula if present
            else:
                msg = "LoadIBW Error: File " + filename + ", has an invalid version code" + Version;            
                raise RuntimeError(msg);
                return None;

            if (NoteSize > 0): 
                ByteString = f.read(NoteSize);
                NoteString = ByteString.decode();       
            else: 
                NoteString = "";
        except:
            msg = "LoadIBW Error: File " + filename + ", Note read failed";            
            raise RuntimeError(msg);
            return None;

    return {"Note": NoteString, "Data": DataOut};

#// IgorBin.h -- structures and #defines for dealing with Igor binary data.
#
#
#// All structures written to disk are 2-byte-aligned.
#
##ifdef WIN32
#	typedef void** Handle;
##endif
#
#
#// From IgorMath.h
##define NT_CMPLX 1			// Complex numbers.
##define NT_FP32 2			// 32 bit fp numbers.
##define NT_FP64 4			// 64 bit fp numbers.
##define NT_I8 8				// 8 bit signed integer. Requires Igor Pro 2.0 or later.
##define NT_I16 	0x10		// 16 bit integer numbers. Requires Igor Pro 2.0 or later.
##define NT_I32 	0x20		// 32 bit integer numbers. Requires Igor Pro 2.0 or later.
##define NT_UNSIGNED 0x40	// Makes above signed integers unsigned. Requires Igor Pro 3.0 or later.
#
#
#// From wave.h
##define MAXDIMS 4
#
#
#//	From binary.h
#
#typedef struct BinHeader1 {
#	short version;						// Version number for backwards compatibility.
#	long wfmSize;						// The size of the WaveHeader2 data structure plus the wave data plus 16 bytes of padding.
#	short checksum;						// Checksum over this header and the wave header.
#} BinHeader1;
#
#typedef struct BinHeader2 {
#	short version;						// Version number for backwards compatibility.
#	long wfmSize;						// The size of the WaveHeader2 data structure plus the wave data plus 16 bytes of padding.
#	long noteSize;						// The size of the note text.
#	long pictSize;						// Reserved. Write zero. Ignore on read.
#	short checksum;						// Checksum over this header and the wave header.
#} BinHeader2;
#
#typedef struct BinHeader3 {
#	short version;						// Version number for backwards compatibility.
#	long wfmSize;						// The size of the WaveHeader2 data structure plus the wave data plus 16 bytes of padding.
#	long noteSize;						// The size of the note text.
#	long formulaSize;					// The size of the dependency formula, including the null terminator, if any. Zero if no dependency formula.
#	long pictSize;						// Reserved. Write zero. Ignore on read.
#	short checksum;						// Checksum over this header and the wave header.
#} BinHeader3;
#
#typedef struct BinHeader5 {
#	short version;						// Version number for backwards compatibility.
#	short checksum;						// Checksum over this header and the wave header.
#	long wfmSize;						// The size of the WaveHeader5 data structure plus the wave data.
#	long formulaSize;					// The size of the dependency formula, including the null terminator, if any. Zero if no dependency formula.
#	long noteSize;						// The size of the note text.
#	long dataEUnitsSize;				// The size of optional extended data units.
#	long dimEUnitsSize[MAXDIMS];		// The size of optional extended dimension units.
#	long dimLabelsSize[MAXDIMS];		// The size of optional dimension labels.
#	long sIndicesSize;					// The size of string indicies if this is a text wave.
#	long optionsSize1;					// Reserved. Write zero. Ignore on read.
#	long optionsSize2;					// Reserved. Write zero. Ignore on read.
#} BinHeader5;
#
#
#//	From wave.h
#
##define MAX_WAVE_NAME2 18	// Maximum length of wave name in version 1 and 2 files. Does not include the trailing null.
##define MAX_WAVE_NAME5 31	// Maximum length of wave name in version 5 files. Does not include the trailing null.
##define MAX_UNIT_CHARS 3
#
#//	Header to an array of waveform data.
#
#struct WaveHeader2 {
#2	short type;							// See types (e.g. NT_FP64) above. Zero for text waves.
#4	struct WaveHeader2 **next;			// Used in memory only. Write zero. Ignore on read.
#
#20	char bname[MAX_WAVE_NAME2+2];		// Name of wave plus trailing null.
#2	short whVersion;					// Write 0. Ignore on read.
#2	short srcFldr;						// Used in memory only. Write zero. Ignore on read.
#4	Handle fileName;					// Used in memory only. Write zero. Ignore on read.
#
#4	char dataUnits[MAX_UNIT_CHARS+1];	// Natural data units go here - null if none.
#4	char xUnits[MAX_UNIT_CHARS+1];		// Natural x-axis units go here - null if none.
#
#l	long npnts;							// Number of data points in wave.
#
#	short aModified;					// Used in memory only. Write zero. Ignore on read.
#	double hsA,hsB;						// X value for point p = hsA*p + hsB
#
#	short wModified;					// Used in memory only. Write zero. Ignore on read.
#	short swModified;					// Used in memory only. Write zero. Ignore on read.
#	short fsValid;						// True if full scale values have meaning.
#	double topFullScale,botFullScale;	// The min full scale value for wave.
#		   
#	char useBits;						// Used in memory only. Write zero. Ignore on read.
#	char kindBits;						// Reserved. Write zero. Ignore on read.
#	void **formula;						// Used in memory only. Write zero. Ignore on read.
#	long depID;							// Used in memory only. Write zero. Ignore on read.
#	unsigned long creationDate;			// DateTime of creation. Not used in version 1 files.
#	unsigned char platform;				// 0=unspecified, 1=Macintosh, 2=Windows; Added for Igor Pro 5.5.
#	char wUnused[1];					// Reserved. Write zero. Ignore on read.
#
#	unsigned long modDate;				// DateTime of last modification.
#	Handle waveNoteH;					// Used in memory only. Write zero. Ignore on read.
#
#	float wData[4];						// The start of the array of waveform data.
#};
#typedef struct WaveHeader2 WaveHeader2;
#typedef WaveHeader2 *WavePtr2;
#typedef WavePtr2 *waveHandle2;
#
#
#struct WaveHeader5 {
#4	struct WaveHeader5 **next;			// link to next wave in linked list.
#
#4	unsigned long creationDate;			// DateTime of creation.
#4	unsigned long modDate;				// DateTime of last modification.
#
#l	long npnts;							// Total number of points (multiply dimensions up to first zero).
#h	short type;							// See types (e.g. NT_FP64) above. Zero for text waves.
#2	short dLock;						// Reserved. Write zero. Ignore on read.
#
#6	char whpad1[6];						// Reserved. Write zero. Ignore on read.
#2	short whVersion;					// Write 1. Ignore on read.
#32	char bname[MAX_WAVE_NAME5+1];		// Name of wave plus trailing null.
#4	long whpad2;						// Reserved. Write zero. Ignore on read.
#4	struct DataFolder **dFolder;		// Used in memory only. Write zero. Ignore on read.
#
#	// Dimensioning info. [0] == rows, [1] == cols etc
#4l	long nDim[MAXDIMS];					// Number of of items in a dimension -- 0 means no data.
#32	double sfA[MAXDIMS];				// Index value for element e of dimension d = sfA[d]*e + sfB[d].
#32	double sfB[MAXDIMS];
#
#	// SI units
#4	char dataUnits[MAX_UNIT_CHARS+1];			// Natural data units go here - null if none.
#16	char dimUnits[MAXDIMS][MAX_UNIT_CHARS+1];	// Natural dimension units go here - null if none.
#
#2	short fsValid;						// TRUE if full scale values have meaning.
#2	short whpad3;						// Reserved. Write zero. Ignore on read.
#16	double topFullScale,botFullScale;	// The max and max full scale value for wave.
#
#4	Handle dataEUnits;					// Used in memory only. Write zero. Ignore on read.
#4	Handle dimEUnits[MAXDIMS];			// Used in memory only. Write zero. Ignore on read.
#4	Handle dimLabels[MAXDIMS];			// Used in memory only. Write zero. Ignore on read.
#	
#4	Handle waveNoteH;					// Used in memory only. Write zero. Ignore on read.
#
#1	unsigned char platform;				// 0=unspecified, 1=Macintosh, 2=Windows; Added for Igor Pro 5.5.
#3	unsigned char spare[3];
#
#52	long whUnused[13];					// Reserved. Write zero. Ignore on read.
#
#8	long vRefNum, dirID;				// Used in memory only. Write zero. Ignore on read.
#
#	// The following stuff is considered private to Igor.
#
#2	short aModified;					// Used in memory only. Write zero. Ignore on read.
#2	short wModified;					// Used in memory only. Write zero. Ignore on read.
#2	short swModified;					// Used in memory only. Write zero. Ignore on read.
#	
#1	char useBits;						// Used in memory only. Write zero. Ignore on read.
#1	char kindBits;						// Reserved. Write zero. Ignore on read.
#4	void **formula;						// Used in memory only. Write zero. Ignore on read.
#4	long depID;							// Used in memory only. Write zero. Ignore on read.
#	
#2	short whpad4;						// Reserved. Write zero. Ignore on read.
#2	short srcFldr;						// Used in memory only. Write zero. Ignore on read.
#4	Handle fileName;					// Used in memory only. Write zero. Ignore on read.
#	
#4	long **sIndices;					// Used in memory only. Write zero. Ignore on read.
#
#8	float wData[1];						// The start of the array of data. Must be 64 bit aligned.
#};

