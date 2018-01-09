# -*- coding: utf-8 -*-

"""

Created on Wed Sep  4 08:07:01 2013



@author: ispielma



Generic v1.0: Generic legacy functions to replicate desired Igor behavior



"""



import re;



def NumberByKey(Default, Key, TestString, keySepStr, listSepStr):

    """

    NumberByKey Extracts a number from a string by identifying a key

    Default is the number to return upon failure

    Key is the keyword to search for

    TestString is the string to search

    keySepStr is a token seperating the Key from the number

    listSepStr identifies when the number ends

    """

    

    RegExp = Key + r"\s*" + keySepStr + "\s*(?P<ExtractStr>[^" + listSepStr + r"]+)";

    

    matchObj = re.search( RegExp, TestString, re.I)



    if matchObj:

        MatchString = matchObj.group("ExtractStr");

        Num = float(MatchString);

    else:

        Num = Default;



    return Num;



def StringByKey(Default, Key, TestString, keySepStr, listSepStr):

    """

    StringByKey Extracts a string from a string by identifying a key

    Default is the string to return upon failure

    Key is the keyword to search for

    TestString is the string to search

    keySepStr is a token seperating the Key from the number

    listSepStr identifies when the number ends

    """

    

    RegExp = Key + r"\s*" + keySepStr + "\s*(?P<ExtractStr>[^" + listSepStr + r"]+)";

    

    matchObj = re.search( RegExp, TestString, re.I)



    if matchObj:

        MatchString = matchObj.group("ExtractStr");

    else:

        MatchString = Default;



    return MatchString;



