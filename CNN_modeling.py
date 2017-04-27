import pandas as pd
import numpy as np

import os,  sys, json, csv,random, math
#from scipy.misc import imread
from sklearn.metrics import accuracy_score

from __future__ import print_function
#from six.moves import cPickle as pickle
#from six.moves import range
import time
import datetime as dt
import re

case_withApplication = pd.read_csv('app_cas.csv',sep='|',header = None)

case_withApplication.columns=["application_id","createdDateTime","lastUpdateDateTime","hotCustomer","customerType","market","submarket", "dealerCodeOrAttId","requestedLines","salesChannel","rulesStr","banStatus"
                     ,"distinctApplicationIdCount" ,"distinctBanCount",  "distinctEmailCount"  ,   "distinctPersistentKeyCount", "distinctSSNCount" ,  "distinctZipCount" 
                     ,"nameAddressHotTagged","nameDobHotTagged","persistentKeyHotTagged", "nameHotTagged",   "emailHotTagged","addressHotTagged","ssnHotTagged"
                     ,"zip","SSN","dob","mobileCustomerRefDate","caseDisposition"
                    , "ApplicationId" 
                     , "RecordDateTime"
                     , "TimeZone"
                     , "RecordType"
                     , "SBPLIndicator"
                     , "ApplicationStatus"
                     , "ApplicationStatusReason"
                     , "MarketCode"
                     , "SubMarket"
                     , "DealerCode"
                     , "ApplicationType"
                     , "LinesRequested"
                     , "CreditBureauPreferenceOrder"
                     , "FirstName"
                     , "MiddleInitial"
                     , "LastName"
                     , "Generation"
                     , "AddressType"
                     , "MilitaryType"
                     , "StreetNumber"
                     , "StreetName"
                     , "StreetType"
                     , "StreetDirection"
                     , "Apartment"
                     , "City"
                     , "State"
                     , "ZipCode"
                     , "ZipCode4"
                     , "Country"
                     , "CurrentRequiredCreditClass"
                     , "CurrentRequiredDeposit"
                     , "CurrentApprovedLines"
                     , "NumberofLinesActivated"
                     , "DecisionCreditBureau"
                     , "DecisionLevel"
                     , "CurrentDecision"
                     , "ModelScore"
                     , "ModelSource"
                     , "DecisionReportGroup"
                     , "DecisionReportGroupText"
                     , "DuplicateApplicationNumber"
                     , "MatchOverride"
                     , "LastOverrideCode"
                     , "LastOverrideCASUserID"
                     , "OutOfWalletIndicator"
                     , "OOWInitialStatus"
                     , "OOWResult"
                     , "AffiliateSalesApplication"
                     , "AffiliateDecision"
                     , "NNDBAccessed"
                     , "NNDBMatchFound"
                     , "NNDBSSNMatchIndicator"
                     , "NNDBNameMatchIndicator"
                     , "NNDBAddressMatchIndicator"
                     , "NNDBDriversLicenceMatchIndicator"
                     , "NNDBContactDetailsMatchIndicator"
                     , "PersistentKey"
                     , "UnifiedCreditTransactionID"
                     , "InterconnectTransactionID"
                     , "AffiliateID"
                     , "has_accountNum"
                     , "SalesChannel"
                     , "StoreID"
                     , "CreditClass"
                     , "SourceApplication"
                     ,"BureauCode"
                     , "BureauModel"
                     ,                "HitCode"
                     ,                "ScoreSourceCode"
                     ,                "RelatedAppNumb"
                     ,                "ScoringSource"
                     ,                "ModelScoreValue"
                     ,                "ScoreFirstName"
                     ,                "ScoreLastName"
                     ,                "ScoreStreetNumber"
                     ,                "ScoreStreetName"
                     ,                "ScoreStreetType"
                     ,                "ScoreCity"
                     ,                "ScoreState"
                     ,                "ScoreZipCode"
                     ,                "SocialSecurityMatchPercent"
                     ,                "NameMatchPercent"
                     ,                "AddressMatchPercent"
                     ,                "SocialSecurityMatchPoints"
                     ,                "NameMatchPoints"
                     ,                "AddressMatchPoints"
                     ,                "StreetNumberPoints"
                     ,                "StreetNamePoints"
                     ,                "CityPoints"
                     ,                "StatePoints"
                     ,                "ZipcodePoints"
                     ,                "TotalMatchPoints"
                     ,                "MinimumMatchforAcceptance"
                     ,                "AlertCode"]

case_withApplication = case_withApplication[case_withApplication.caseDisposition.isin (['Fraud','NotFraud','UnknownFraud'])]
case_withApplication = case_withApplication.drop_duplicates()

case_withApplication['createdDate']= case_withApplication.createdDateTime.apply(lambda x: dt.datetime.fromtimestamp(x/1000))
#case_withApplication['applicationDate']= case_withApplication.applicationDateTime.apply(lambda x: dt.datetime.fromtimestamp(x/1000))
case_withApplication['lastUpdateDate']= case_withApplication.lastUpdateDateTime.apply(lambda x:dt.datetime.fromtimestamp(x/1000))

def get_hotTaggedFeature (myDf):
    if myDf.nameAddressHotTagged=="ExactMatch":
        return 1
    elif  myDf.nameDobHotTagged=="ExactMatch":
        return 2
    elif myDf.nameHotTagged=="ExactMatch":
        return 3
    elif myDf.persistentKeyHotTagged=="ExactMatch":
        return 4
    elif myDf.ssnHotTagged=="ExactMatch":
        return 5
    elif myDf.addressHotTagged=="ExactMatch":
        return 6
    elif myDf.emailHotTagged=="ExactMatch":
        return 7
    else:
        return 0

def get_ifHotTagged(myColumn):
    if len(re.findall(r'exactmatch',myColumn,re.IGNORECASE))>0:
        return 1
    else:
        return 0

case_withApplication['hotTagged_feature']=case_withApplication.nameAddressHotTagged.map(str)+case_withApplication.nameDobHotTagged.map(str)+case_withApplication.nameHotTagged.map(str)+case_withApplication.persistentKeyHotTagged.map(str)+case_withApplication.ssnHotTagged.map(str)+case_withApplication.addressHotTagged.map(str)+case_withApplication.emailHotTagged.map(str)

#case_withApplication$valid_zip<-as.factor(ifelse(nchar(as.character(case_withApplication$zip))<5,0,1))
#case_withApplication$domain_valid<-as.factor(ifelse(tolower(case_withApplication$email_domain) %in% domains,1,0))

#case_withApplication$email_valid<-as.factor(ifelse(case_withApplication$domain_valid==0,0,tolower(case_withApplication$email_domain)))
case_withApplication['mobileCustRefYear']= case_withApplication.mobileCustomerRefDate.apply(lambda x: str(x)[:3])
case_withApplication['valid_SSN']= case_withApplication.SSN.apply(lambda x: 1 if len(str(x))==9 else 0)

dummies = pd.get_dummies(case_withApplication['rulesStr']).rename(columns=lambda x: 'rulesStr_' + str(x))
case_withApplication = pd.concat([case_withApplication, dummies], axis=1)

dummies = pd.get_dummies(case_withApplication['banStatus']).rename(columns=lambda x: 'banStatus_' + str(x))
case_withApplication = pd.concat([case_withApplication, dummies], axis=1)
#dummies = pd.get_dummies(case_withApplication['zip']).rename(columns=lambda x: 'zip_' + str(x))
#case_withApplication = pd.concat([case_withApplication, dummies], axis=1)

dummies = pd.get_dummies(case_withApplication['salesChannel']).rename(columns=lambda x: 'salesChannel_' + str(x))
case_withApplication = pd.concat([case_withApplication, dummies], axis=1)
dummies = pd.get_dummies(case_withApplication['mobileCustRefYear']).rename(columns=lambda x: 'mobileCustRefYear_' + str(x))
case_withApplication = pd.concat([case_withApplication, dummies], axis=1)
#dummies = pd.get_dummies(case_withApplication['market']).rename(columns=lambda x: 'market_' + str(x))
#case_withApplication = pd.concat([case_withApplication, dummies], axis=1)
dummies = pd.get_dummies(case_withApplication['CurrentRequiredCreditClass']).rename(columns=lambda x: 'CurrentRequiredCreditClass_' + str(x))
case_withApplication = pd.concat([case_withApplication, dummies], axis=1)

case_withApplication['if_hotTagged']=case_withApplication['hotTagged_feature'].apply(get_ifHotTagged)

case_withApplication=case_withApplication.iloc[np.random.permutation(len(case_withApplication))]

case_withApplication.drop(["application_id",     "createdDateTime"     ,       "lastUpdateDateTime"    ,     "hotCustomer",        "customerType",      
 "market",             "submarket",          "dealerCodeOrAttId",     "salesChannel",      
 "rulesStr",           "banStatus",           "nameAddressHotTagged"   ,          "nameDobHotTagged",  
 "persistentKeyHotTagged"    ,       "nameHotTagged",      "emailHotTagged",     "addressHotTagged",   "ssnHotTagged",      
 "zip",  "SSN",  "dob" ,  "mobileCustomerRefDate"     ,       "caseDisposition",   
 "ApplicationId",      "RecordDateTime",     "TimeZone",           "RecordType",         "SBPLIndicator",     
 "ApplicationStatus",  "ApplicationStatusReason"    ,      "MarketCode",         "SubMarket",          "DealerCode",        
 "ApplicationType",    "LinesRequested",     "CreditBureauPreferenceOrder"   ,   "FirstName",          "MiddleInitial",     
 "LastName",           "Generation",         "AddressType",        "MilitaryType",       "StreetNumber",      
 "StreetName",         "StreetType",         "StreetDirection",    "Apartment",          "City",
 "State", "ZipCode",            "ZipCode4",           "Country",            "CurrentRequiredCreditClass"    ,  
       "NumberofLinesActivated" ,          "DecisionCreditBureau"    ,         "DecisionLevel",     
 "CurrentDecision",   'ScoreFirstName', 'ScoreLastName',
       'ScoreStreetNumber', 'ScoreStreetName', 'ScoreStreetType',
       'ScoreCity', 'ScoreState', 'ScoreZipCode', "ModelScore",         "ModelSource",        "DecisionReportGroup","DecisionReportGroupText",    
 "DuplicateApplicationNumber"  ,     "MatchOverride",      "LastOverrideCode",   "LastOverrideCASUserID"     ,       "OutOfWalletIndicator"   ,         
 "OOWInitialStatus",   "OOWResult",          "AffiliateSalesApplication"   ,     "AffiliateDecision",  "NNDBAccessed",      
 "NNDBMatchFound",     "NNDBSSNMatchIndicator"      ,      "NNDBNameMatchIndicator"     ,      "NNDBAddressMatchIndicator"    ,    "NNDBDriversLicenceMatchIndicator",
 "NNDBContactDetailsMatchIndicator", "PersistentKey",      "UnifiedCreditTransactionID"   ,    "InterconnectTransactionID"  ,      "AffiliateID",       
 "has_accountNum",     "SalesChannel",       "StoreID",            "CreditClass",        "SourceApplication", 
 "BureauCode",         "BureauModel",        "HitCode",            "ScoreSourceCode",    "RelatedAppNumb",    
 "ScoringSource",      "ModelScoreValue",   "SocialSecurityMatchPoints"    ,    "NameMatchPoints",   
 "AddressMatchPoints", "StreetNumberPoints", "StreetNamePoints",   "CityPoints",         "StatePoints",       
 "ZipcodePoints",      "MinimumMatchforAcceptance"   ,     "AlertCode",   "hotTagged_feature",     "mobileCustRefYear",  "valid_SSN","zip"], inplace=True, axis=1)

case_withApplication=case_withApplication.iloc[np.random.permutation(len(case_withApplication))]
case_withApplication_tr=case_withApplication[(case_withApplication.lastUpdateDate<='2017-04-05') & (case_withApplication.lastUpdateDate>='2017-03-15') ]
#case_withApplication_ts=dsp_weekly[~((dsp_weekly.year<=2016)&(dsp_weekly.WeekofMonth<45))]
case_withApplication_ts = case_withApplication[(case_withApplication.createdDate >='2017-04-06') &(case_withApplication.createdDate <='2017-04-08')]

#case_withApplication_tr1 = case_withApplication_tr[['ApplicationInfo','if_hotTagged']]
#case_withApplication_ts1 = case_withApplication_ts[['ApplicationInfo','if_hotTagged']]

case_withApplication_tr.drop(['lastUpdateDate','createdDate'],axis=1,inplace=True)
case_withApplication_ts.drop(['lastUpdateDate','createdDate'],axis=1,inplace=True)



split_size = int(case_withApplication_tr.shape[0]*0.9)

train_x, val_x = case_withApplication_tr.iloc[:split_size,:-1], case_withApplication_tr.iloc[split_size:,:-1]
train_y, val_y = case_withApplication_tr.iloc[:split_size,-1], case_withApplication_tr.iloc[split_size:,-1]

train_x = np.array(train_x)
val_x = np.array(val_x)
#val_x = np.array(dsp_weekly_ts.iloc[:,:-1])
train_y = np.array(train_y)
val_y=np.array(val_y)

#val_y = np.array(dsp_weekly_ts.iloc[:,-1])
train = case_withApplication_tr.iloc[:split_size,:]
val = case_withApplication_tr.iloc[split_size:,:]
test = case_withApplication_ts
test_x=np.array(case_withApplication_ts.iloc[:,:-1])
test_y=np.array(case_withApplication_ts.iloc[:,-1])

import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.
        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return [shared_x, T.cast(shared_y, "int32")]

training_data = shared([train_x,train_y])
validation_data = shared([val_x,val_y])
test_data = shared([test_x,test_y])

mini_batch_size=20
poolsize=[10,1]
filter_shape = [20,1,(train_x.shape[1]-np.round(train_x.shape[1]/poolsize[0])*poolsize[0]+1),1]
firstConvLayer_shape = (train_x.shape[1]-filter_shape[2]+1)/poolsize[0]
poolsize1=[5,1]
filter_shape1 = [20,filter_shape[0],(firstConvLayer_shape-np.round(firstConvLayer_shape/poolsize1[0])*poolsize1[0]+1),1]

ConnectLayerOut = 200
net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, train_x.shape[1],1), 
                      filter_shape=(filter_shape[0],filter_shape[1],filter_shape[2],filter_shape[3]), 
                      poolsize=(poolsize[0],1),activation_fn = ReLU),
    ConvPoolLayer(image_shape = (mini_batch_size,filter_shape[0],firstConvLayer_shape,1),
                      filter_shape=(filter_shape1[0],filter_shape1[1],filter_shape1[2],1),
                      poolsize=(poolsize1[0],poolsize1[1]),
                      activation_fn = ReLU),

        FullyConnectedLayer(n_in=filter_shape1[0]*(np.round(firstConvLayer_shape/poolsize1[0]))*1, n_out=ConnectLayerOut,p_dropout=0.1),
        SoftmaxLayer(n_in=ConnectLayerOut, n_out=2,p_dropout=0.1)], mini_batch_size,)
net.SGD(training_data, 100, mini_batch_size, 0.03, 
            validation_data, test_data,lmbda=0.1) 

