import scipy.io
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.fftpack import fft

source = scipy.io.loadmat(r'C:\Users\Dixie\Documents\NTUST\Special Topic II\Fisher Criterion\Fisher Criterion\ADHC.mat')
ad = source["AD"] #3 people, 30x45000
hc = source["HC"] #4 people, 30x45000
#print(ad.shape) #(3,1)
#print(ad[0].shape[0]) #[1][2]
#print(ad[0][0].shape) #(30,45000) Data size of 1 person

#Speftral Power Data Frame of AD and HC
ad_sp_df = [] #3 people data frame
hc_sp_df = [] #4 people data frame

for i in range (0,(ad.shape[0])):
    for j in range (0,30):
        ad_data = ad[i][0][j][0:1500]
        ad_psd = abs(fft(ad_data))**2
        #Delta, Theta, Alpha, Beta_low, Beta_high, Gamma
        ad_sp = ((sum(ad_psd[(1*3):((4*3)-1)])), (sum(ad_psd[(4*3):((8*3)-1)])), (sum(ad_psd[(8*3):((13*3)-1)])), 
                (sum(ad_psd[(13*3):((20*3)-1)])), (sum(ad_psd[(20*3):((30*3)-1)])), (sum(ad_psd[(30*3):((45*3)-1)])))
        ad_sp_df.append(ad_sp)
#print(pd.DataFrame(ad_sp_df))

for i in range (0,(hc.shape[0])):
    for j in range (0,30):
        hc_data = hc[i][0][j][0:1500]
        hc_psd = abs(fft(hc_data))**2
        #Delta, Theta, Alpha, Beta_low, Beta_high, Gamma
        hc_sp = ((sum(hc_psd[(1*3):((4*3)-1)])), (sum(hc_psd[(4*3):((8*3)-1)])), (sum(hc_psd[(8*3):((13*3)-1)])), 
                (sum(hc_psd[(13*3):((20*3)-1)])), (sum(hc_psd[(20*3):((30*3)-1)])), (sum(hc_psd[(30*3):((45*3)-1)])))
        hc_sp_df.append(hc_sp)
#print(pd.DataFrame(hc_sp_df))

#                       | 1 | 2 | 3 | 4 | 5 |...| 27 | 28 | 29 | 30 |
#Delta layer            |   |   |   |   |   |   |    |    |    |    |
#Theta layer            |   |   |   |   |   |   |    |    |    |    |
#Alpha layer            |   |   |   |   |   |   |    |    |    |    |
#Beta low layer         |   |   |   |   |   |   |    |    |    |    |
#Beta high layer        |   |   |   |   |   |   |    |    |    |    |
#Gamma layer            |   |   |   |   |   |   |    |    |    |    |
#180 array with 30 delta, 30 Theta, 30 Alpha, 30 Beta low, 30 Beta high, 30 Gamma
ad1_df = np.concatenate(((pd.DataFrame(ad_sp_df).iloc[0:30,:]).T).to_numpy()) 
ad2_df = np.concatenate(((pd.DataFrame(ad_sp_df).iloc[30:60,:]).T).to_numpy())
ad3_df = np.concatenate(((pd.DataFrame(ad_sp_df).iloc[60:90,:]).T).to_numpy())
hc1_df = np.concatenate(((pd.DataFrame(hc_sp_df).iloc[0:30,:]).T).to_numpy()) 
hc2_df = np.concatenate(((pd.DataFrame(hc_sp_df).iloc[30:60,:]).T).to_numpy())
hc3_df = np.concatenate(((pd.DataFrame(hc_sp_df).iloc[60:90,:]).T).to_numpy())
hc4_df = np.concatenate(((pd.DataFrame(hc_sp_df).iloc[90:120,:]).T).to_numpy())

#                           | AD_1 | AD_2 | AD_3 | AD_4 | AD_5 | HC_1 | HC_2 | HC_3 | HC_4 |
#0-29    :Delta layer       |      |      |      |      |      |      |      |      |      |
#30-59   :Theta layer       |      |      |      |      |      |      |      |      |      |
#60-89   :Alpha layer       |      |      |      |      |      |      |      |      |      |
#90-119  :Beta low layer    |      |      |      |      |      |      |      |      |      |
#120-149 :Beta high layer   |      |      |      |      |      |      |      |      |      |
#150-179 :Gamma layer       |      |      |      |      |      |      |      |      |      |
xij = pd.DataFrame({'AD_1': ad1_df,'AD_2':ad2_df, 'AD_3':ad3_df,'HC_1':hc1_df, 'HC_2':hc2_df, 'HC_3':hc3_df, 'HC_4':hc4_df})
ad_mean_value = xij[['AD_1','AD_2','AD_3']].mean(axis=1)
hc_mean_value = xij[['HC_1','HC_2','HC_3','HC_4']].mean(axis=1)
#print(xij)
#print(ad_mean_value)
#print(hc_mean_value)

#Sample Covariance Matrix
xij_new = pd.DataFrame({'AD_1': ad1_df-ad_mean_value,'AD_2':ad2_df-ad_mean_value, 'AD_3':ad3_df-ad_mean_value,
                        'HC_1':hc1_df-hc_mean_value, 'HC_2':hc2_df-hc_mean_value, 'HC_3':hc3_df-hc_mean_value, 'HC_4':hc4_df-hc_mean_value})
ad_inner = pd.DataFrame(xij_new[['AD_1','AD_2','AD_3']]) #-ad_mean_value
hc_inner = pd.DataFrame(xij_new[['HC_1','HC_2','HC_3','HC_4']]) #-hc_mean_value
#print((ad_inner))
#print(hc_inner)

ad_inner_df = (ad_inner.to_numpy())
ad_inner_df_t = ((ad_inner).T)
hc_inner_df = (hc_inner.to_numpy())
hc_inner_df_t = ((hc_inner).T)
#print(ad_inner_df)
ad_si = (1/3)*(np.dot(ad_inner_df,ad_inner_df_t))
hc_si = (1/4)*(np.dot(hc_inner_df,hc_inner_df_t))
#print(pd.DataFrame(ad_si))
#print(pd.DataFrame(hc_si))


#Within Class Scatter
p_ad = (len(ad))/(len(ad)+len(hc)) 
p_hc = (len(hc))/(len(ad)+len(hc))
#print(p_ad)
#print(p_hc)
ad_sw = p_ad*ad_si
hc_sw = p_hc*hc_si
sw = pd.DataFrame(ad_sw + hc_sw)
#print(ad_sw)


#Between Class Scatter
mean = (xij.mean(axis=1))
#print(mean)

ad_mean_inner = (pd.DataFrame(ad_mean_value-mean)).to_numpy()
ad_mean_inner_t = ad_mean_inner.T
hc_mean_inner = (pd.DataFrame(hc_mean_value-mean)).to_numpy()
hc_mean_inner_t = hc_mean_inner.T
ad_mean_sb = p_ad*(np.dot(ad_mean_inner,ad_mean_inner_t))
hc_mean_sb = p_hc*(np.dot(hc_mean_inner,hc_mean_inner_t))
sb = ad_mean_sb + hc_mean_sb
print(ad_mean_inner)
print(ad_mean_inner_t)

#Kth Feature
k = (sb/sw).to_numpy()
#print(pd.DataFrame(k))
k_value = []
for i in range (180):
        score = k[i][i]
        k_value.append(score)
k_score = sorted(k_value, reverse=True)
k_score_index = np.argsort(k_value)[::-1]
print(k_score)
print((k_score_index))
