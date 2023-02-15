import csv
import pandas as pd
infile1="data_forRecoLength_beamlikeEvts19.csv"
infile2="recoMRDlength19.csv"
filein1 = open(str(infile1))
df1=pd.read_csv(filein1)
print(df1.head())
filein2 = open(str(infile2))
df2=pd.read_csv(filein2)
print(df2.head())
eventnumber=df1['eventNumber']
MRDTrackLength=df2['recoTrackLengthInMrd']
MRDTrackLengthnew=[]
for i in eventnumber:
   MRDTrackLengthnew.append(MRDTrackLength[i])
print(df1.shape)
df1.insert(2224, 'recoTrackLengthInMRD', MRDTrackLengthnew, allow_duplicates="True")
df1.to_csv("vars_ErecoMRD19.csv", float_format = '%.3f')

