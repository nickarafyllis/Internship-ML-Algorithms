#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TStopwatch.h"
#include "TF1.h"
#include "TMath.h"
void LocalFindMRDTracks(){
  ofstream csvfile;
  csvfile.open ("recoMRDlength19.csv");
   csvfile<<"eventNumber"<<",";
   csvfile<<"recoTrackLengthInMrd";
   csvfile<<'\n';
   char fname[100];
   sprintf(fname,"mrdtracks/vtxreco-beamlikewithmrdtracks19.root");
   TFile *input=new TFile(fname,"READONLY");
   cout<<"input file: "<<fname<<endl;
   
   TTree *regTree= (TTree*) input->Get("phaseIITriggerTree");
   double recoTrackLengthInMrd;
   Int_t event;
   std::vector<double> *MRDTrackLength=0;
   regTree->SetBranchAddress("eventNumber", &event);
   regTree->SetBranchAddress("MRDTrackLength", &MRDTrackLength);
   for (Long64_t ievt=0; ievt<regTree->GetEntries(); ievt++) {
   regTree->GetEntry(ievt);
   recoTrackLengthInMrd = MRDTrackLength->at(0);
   csvfile<<event<<",";
   csvfile<<recoTrackLengthInMrd;
   csvfile<<'\n';
   }
   //ofstream csvfile1;
   //csvfile1.open ("numMRDtracks.csv");
   //csvfile1<<"numMRDTracks";
   //csvfile1<<'\n';
   //int numMRDTracks;
   //TTree *regTree1= (TTree*) input->Get("phaseIITriggerTree");
   //regTree1->SetBranchAddress("numMRDTracks", &numMRDTracks);
   //for (Long64_t ievt=0; ievt<regTree1->GetEntries(); ievt++) {
   //regTree1->GetEntry(ievt);
   //csvfile1<<numMRDTracks;
   //csvfile1<<'\n';
   //}
   //csvfile1.close();
}
