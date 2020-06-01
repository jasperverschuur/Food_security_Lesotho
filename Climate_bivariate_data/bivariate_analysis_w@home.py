from pylab import *
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import pandas as pd

from scipy.stats import multivariate_normal as mvn
import scipy.stats as stats
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import kendalltau as kdtau
from scipy.stats import spearmanr as spearman
from scipy import stats
import statsmodels.api as sm
import seaborn as sns
plt.rcParams["font.family"] = "Times New Roman"
#sns.set(style="white", color_codes=True)

values_Lesotho = []
values_SA = []
values_LesothoN = []
values_SAN = []

for i in range(0,118):
    f = nc.MFDataset('WAH_PRCP_ACT_1987-2014_JFM_L.nc')
    value_les = f.variables['PRCP'][:]*90
    value_les = np.squeeze(value_les[:,i,0,0])
    values_Lesotho= np.concatenate([values_Lesotho, value_les])

    f.close

    f = nc.MFDataset('WAH_PRCP_ACT_1987-2014_JFM_SA.nc')
    value_SA = f.variables['PRCP'][:]*90
    value_SA = np.squeeze(value_SA[:,i,0,0])
    values_SA = np.concatenate([values_SA, value_SA])
    f.close

for i in range(0,118):
    f = nc.MFDataset('WAH_PRCP_natural_1987-2017_L.nc')
    value_lesN = f.variables['PRCP'][:]*90
    value_lesN = np.squeeze(value_lesN[0:28,i,0,0])
    values_LesothoN= np.concatenate([values_LesothoN, value_lesN])

    f.close

    f = nc.MFDataset('WAH_PRCP_natural_1987-2017_SA.nc')
    value_SAN = f.variables['PRCP'][:]*90
    value_SAN = np.squeeze(value_SAN[0:28,i,0,0])
    values_SAN = np.concatenate([values_SAN, value_SAN])
    f.close

print(values_Lesotho.shape)
print(values_SA.shape)
print(values_LesothoN.shape)
print(values_SAN.shape)


### Calculate correlation coefficient

covariate =  np.corrcoef(values_Lesotho,values_SA)
print("Correlation Actual",covariate)

covariate1 = np.corrcoef(values_LesothoN,values_SAN)
print("Correlation Natural",covariate1)

mean_Lesotho = np.mean(values_Lesotho)
mean_SA = np.mean(values_SA)

mean_LesothoN = np.mean(values_LesothoN)
mean_SAN = np.mean(values_SAN)

mean_arr = np.array([mean_Lesotho, mean_SA])
print(mean_arr)



## Kolmogorov-Smirnov test to check if values are normally distributed

norm1 = (values_Lesotho - values_Lesotho.mean()) / values_Lesotho.std()
norm2 = (values_SA - values_SA.mean()) / values_SA.std()
norm3 = (values_LesothoN - values_LesothoN.mean()) / values_LesothoN.std()
norm4 = (values_SAN - values_SAN.mean()) / values_SAN.std()


print("Lesotho actual ks:",stats.kstest(norm1,'norm'))
print("SA actual ks:",stats.kstest(norm2,'norm'))



print("Lesotho natural ks:",stats.kstest(norm3,'norm'))
print("SA natural ks:",stats.kstest(norm4,'norm'))

print(sm.stats.lilliefors(values_Lesotho))

#### fit empirical CDF data


#Uniform marginals by applying the empirical CDF to sample values
Xecdf=ECDF(values_Lesotho,side='right') #default side is rightsided [a,b)
Yecdf=ECDF(values_SA,side='right')
U=Xecdf(values_Lesotho)
V=Yecdf(values_SA)

Xecdf1=ECDF(values_LesothoN,side='right') #default side is rightsided [a,b)
Yecdf1=ECDF(values_SAN,side='right')
U1=Xecdf(values_LesothoN)
V1=Yecdf(values_SAN)

def multidim_cumsum(a):
    out = a[...,::-1].cumsum(-1)[...,::-1]
    for i in range(2,a.ndim+1):
        np.cumsum(out, axis=-i, out=out)
    return out

def ecdf(data):
    """ Compute ECDF """
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return(x,y)



xval, yval = ecdf(values_Lesotho)
xval1, yval1 = ecdf(values_SA)

xval2, yval2 = ecdf(values_LesothoN)
xval3, yval3 = ecdf(values_SAN)

xx_val, yy_val = np.meshgrid(yval, yval1)
xx_val1, yy_val1 = np.meshgrid(yval2, yval3)


levels = [0.01, 0.10, 0.50, 0.90,0.99]
z = ((xx_val* yy_val))
z1 = ((xx_val1* yy_val1))
fig,ax = plt.subplots()
h = plt.contour(xval,xval1,z,levels,colors='r')
g = plt.contour(xval2,xval3,z1,levels,colors='b')
plt.plot(values_Lesotho,values_SA,'ro',markersize='1.0', alpha = 0.7)
plt.plot(values_LesothoN,values_SAN,'bo',markersize='1.0', alpha = 0.7)
plt.clabel(h,inline=1, fontsize=7,fmt = '%1.2f',colors = 'r')
plt.clabel(g,inline=1, fontsize=7,fmt = '%1.2f',colors = 'b')
plt.plot(277.5, 175.5, 'ko')
plt.plot([277.5,277.5], [141, 193.5], 'k')
plt.plot([224.5, 310.5],[175, 175],'k')
plt.text(200,450,'$\\rho$ = 0.60',color = 'r', fontsize = '11')
plt.text(200,430,'$\\rho$ = 0.58',color = 'b', fontsize = '11')
plt.xlim(200,700)
plt.ylim(100,500)
ax.tick_params(direction='in', length=4, width=1, grid_alpha=0.5)
plt.xlabel('JFM total precipitation Lesotho (mm)', fontsize = '11')
plt.ylabel('JFM total precipitation South Africa (mm)',fontsize = '11')

plt.tight_layout()


########


standard = np.linspace(0,3.0,31)



data_act = pd.DataFrame({'val-L':values_Lesotho,'val-SA': values_SA})
data_nat = pd.DataFrame({'val-L':values_LesothoN,'val-SA': values_SAN})
data_joint_act_exceedance = pd.DataFrame(index=np.linspace(0,3,31))
data_joint_nat_exceedance = pd.DataFrame(index=np.linspace(0,3,31))


n = len(data_act)
for j in range(2000):
    data_boot_act = data_act.iloc[np.random.randint(n, size=n)]
    data_boot_nat = data_nat.iloc[np.random.randint(n, size=n)]
    prob_co_act = []
    prob_co_nat = []
    for i in range(len(standard)):
        val_L_event = (277.5 - np.mean(values_Lesotho))/np.std(values_Lesotho)
        val_L_low = (224.5 - np.mean(values_Lesotho))/np.std(values_Lesotho)
        val_L_up = (310.5 - np.mean(values_Lesotho))/np.std(values_Lesotho)
        val_SA_event = (175.5 - np.mean(values_SA))/np.std(values_SA)
        val_SA_low = (141 - np.mean(values_SA))/np.std(values_SA)
        val_SA_up = (193.5 - np.mean(values_SA))/np.std(values_SA)

        val_les_co = np.mean(values_Lesotho) - standard[i]*np.std(values_Lesotho)
        val_sa_co = np.mean(values_SA) - standard[i]*np.std(values_SA)
        data_act_subset = data_boot_act[(data_boot_act['val-L']<val_les_co) & (data_boot_act['val-SA'] < val_sa_co)]
        data_nat_subset = data_boot_nat[(data_boot_nat['val-L']<val_les_co) & (data_boot_nat['val-SA'] < val_sa_co)]

        prob_co_act.append(len(data_act_subset)/len(values_Lesotho))
        prob_co_nat.append(len(data_nat_subset)/len(values_Lesotho))
    data_joint_act_exceedance[j] = prob_co_act
    data_joint_nat_exceedance[j] = prob_co_nat

print(data_joint_act_exceedance)
print(data_joint_act_exceedance.mean(axis=1))
print(data_joint_act_exceedance.quantile(0.95,axis=1))

rho_act_exceedance = pd.DataFrame(index=np.linspace(0,1,11))
rho_nat_exceedance = pd.DataFrame(index=np.linspace(0,1,11))
standard1 = np.linspace(0,1.0,11)

n = len(data_act)
for j in range(2000):
    data_boot_act = data_act.iloc[np.random.randint(n, size=n)]
    data_boot_nat = data_nat.iloc[np.random.randint(n, size=n)]
    corr_sub_act = []
    corr_sub_nat = []
    for i in range(len(standard1)):
        val_les_co1 = np.mean(values_Lesotho) - standard1[i]*np.std(values_Lesotho)
        val_sa_co1 = np.mean(values_SA) - standard1[i]*np.std(values_SA)
        val_les_co1N = np.mean(values_LesothoN) - standard1[i]*np.std(values_LesothoN)
        val_sa_co1N = np.mean(values_SAN) - standard1[i]*np.std(values_SAN)

        data_act_subset1 = data_boot_act[(data_boot_act['val-L']<val_les_co1) & (data_boot_act['val-SA'] < val_sa_co1)]
        data_nat_subset1 = data_boot_nat[(data_boot_nat['val-L']<val_les_co1N) & (data_boot_nat['val-SA'] < val_sa_co1N)]


        corr_sub_act.append(np.corrcoef(data_act_subset1['val-L'],data_act_subset1['val-SA'])[0,1])
        corr_sub_nat.append(np.corrcoef(data_nat_subset1['val-L'],data_nat_subset1['val-SA'])[0,1])
    rho_act_exceedance[j] = corr_sub_act
    rho_nat_exceedance[j] = corr_sub_nat


fig = plt.figure(figsize=(7,5))
plt.plot(np.linspace(0,3.0,31),data_joint_act_exceedance.mean(axis=1),'ro-',np.linspace(0,3.0,31),data_joint_nat_exceedance.mean(axis=1),'bo-',linewidth = 3.0)
plt.fill_between(np.linspace(0,3.0,31),data_joint_act_exceedance.quantile(0.95,axis=1).iloc[0:],data_joint_act_exceedance.quantile(0.05,axis=1).iloc[0:],color = 'r',alpha = 0.3)
plt.fill_between(np.linspace(0,3.0,31),data_joint_nat_exceedance.quantile(0.95,axis=1).iloc[0:],data_joint_nat_exceedance.quantile(0.05,axis=1).iloc[0:],color = 'b',alpha = 0.3)

#plt.plot(standard,prob_co_act,'ro-',standard,prob_co_nat,'bo-', linewidth = 3.0)
plt.axvline(-np.mean([val_L_event,val_SA_event]),color = 'Grey', linestyle = '-', linewidth = 1.0)
plt.axvline(-np.max([val_L_up,val_SA_up]),color = 'Grey', linestyle = ':',linewidth = 0.75)
plt.axvline(-np.min([val_L_low,val_SA_low]),color = 'Grey', linestyle = ':',linewidth = 0.75)
plt.axvspan(-np.max([val_L_up,val_SA_up]), -np.min([val_L_low,val_SA_low]),alpha=0.3, color='Grey')
plt.xticks([0, 0.5,1.0, 1.5, 2.0, 2.5, 3.0],('0$\sigma$','-0.5$\sigma$','-1$\sigma$','-1.5$\sigma$','-2$\sigma$','-2.5$\sigma$','-3$\sigma$'))
plt.yticks([0, 0.1,0.2, 0.3,0.4])

plt.ylim(0,0.4)
plt.xlim(0,3.0)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xlabel('Joint exceedance JFM $P_{tot}$',fontsize = '15')
plt.ylabel('Probability density (%)', fontsize = '15')
#plt.tight_layout()
plt.savefig('prob_density.pdf')

fig = plt.figure(figsize=(3,2))
plt.tick_params(axis='both', which='major', labelsize=12)
plt.plot(rho_act_exceedance.mean(axis = 1),'ro-',rho_nat_exceedance.mean(axis = 1),'bo-')
plt.fill_between(standard1,rho_act_exceedance.quantile(0.95,axis=1).iloc[0:],rho_act_exceedance.quantile(0.05,axis=1).iloc[0:],color = 'r',alpha = 0.3)
plt.fill_between(standard1,rho_nat_exceedance.quantile(0.95,axis=1).iloc[0:],rho_nat_exceedance.quantile(0.05,axis=1).iloc[0:],color = 'b',alpha = 0.3)
plt.xticks([0,0.5,1.0],('0$\sigma$','-0.5$\sigma$','-1$\sigma$'))
plt.ylabel(r'$\rho$',fontsize = '13')
plt.xlabel('Joint exceedance JFM $P_{tot}$',fontsize = '13')
plt.ylim(0.2,0.8)
plt.xlim(0,1.0)
plt.tight_layout()
plt.savefig('corr_sub.pdf',transparent = True)

######
x = values_Lesotho
y = values_SA

x1 = values_LesothoN
y1 = values_SAN
# Define the borders
deltaX = (max(x) - min(x))/10
deltaY = (max(y) - min(y))/10
xmin = min(x) - deltaX
xmax = max(x) + deltaX
ymin = min(y) - deltaY
ymax = max(y) + deltaY
print(xmin, xmax, ymin, ymax)
# Create meshgrid
xx, yy = np.mgrid[min(x)-50:max(x)+50:500j, min(y)-50:max(y)+50:500j]

xx1, yy1 = np.mgrid[min(x1)-50:max(x1)+50:500j, min(y1)-50:max(y1)+50:500j]

positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = stats.gaussian_kde(values)
#print(kernel_new)
print(kernel)
f = np.reshape(kernel(positions).T, xx.shape)
f_new = multidim_cumsum(f)

positions1 = np.vstack([xx1.ravel(), yy1.ravel()])
values1 = np.vstack([x1, y1])
kernel1 = stats.gaussian_kde(values1)
f1 = np.reshape(kernel1(positions1).T, xx1.shape)

levels =  [0.10, 0.50, 0.90]
fig = plt.figure(figsize=(3.0,2.0))
ax = fig.gca()
ax.plot(values_Lesotho,values_SA,'ko',markersize='0.2', alpha = 0.6)
ax.plot(values_LesothoN,values_SAN,'ko',markersize='0.2', alpha = 0.6)

print(np.max(f))
cset = ax.contour(xx, yy, (f/np.max(f)), levels,colors= 'r', linewidths=(1.5))#colors='r')
cset1 = ax.contour(xx1, yy1, (f1/np.max(f1)), levels,colors='b', linewidths=(1.5))

ax.set_xlabel('JFM $P_{tot}$ L (mm)',fontsize = '13')
ax.set_ylabel('JFM $P_{tot}$ SA (mm)',fontsize = '13')
plt.plot(277.5, 175.5, 'ko')
plt.plot([277.5,277.5], [141, 193.5], 'k')
plt.plot([224.5, 310.5],[175, 175],'k')
values_ticks = [-3.0,-1.5,0,1.5,3.0]
values_ticks1 = [i * np.std(values_Lesotho) for i in values_ticks]
values_ticks2 = [i * np.std(values_SA) for i in values_ticks]

ticks_number = np.mean(values_Lesotho)+values_ticks1
ticks_number1 = np.mean(values_SA)+values_ticks2
plt.tick_params(axis='both', which='major', labelsize=12)
print(ticks_number)
plt.xticks(ticks_number,('-3$\sigma$','-1.5$\sigma$','0$\sigma$','1.5$\sigma$','3$\sigma$'))
plt.yticks(ticks_number1,('-3$\sigma$','-1.5$\sigma$','0$\sigma$','1.5$\sigma$','3$\sigma$'))
ax.set_xlim(ticks_number[0], ticks_number[4])
ax.set_ylim(ticks_number1[0], ticks_number1[4])
plt.tight_layout()
plt.savefig('correlation_empirical.pdf', orientation='portrait', format='pdf', transparent = True)
######


##########



fig = plt.figure(figsize=(6,5))
plt.plot(values_Lesotho,values_SA,'ro',markersize='1.0', alpha = 0.3)
plt.plot(values_LesothoN,values_SAN,'bo',markersize='1.0', alpha = 0.3)
ax = sns.kdeplot(values_Lesotho,values_SA,cmap="Reds", shade=False, shade_lowest=False,vmin = 0,vmax = 0.001)
ax = sns.kdeplot(values_LesothoN,values_SAN,cmap="Blues", shade=False, shade_lowest=False,vmin = 0,vmax = 0.001)
plt.plot(277.5, 175.5, 'ko')
plt.plot([277.5,277.5], [141, 193.5], 'k')
plt.plot([224.5, 310.5],[175, 175],'k')
plt.xlim(100,700)
plt.ylim(0,600)

x, y = np.mgrid[0:700:1, 0:580:1]
pos = np.dstack((x, y))
pos1 = np.stack(([values_Lesotho,values_SA]),axis = 0)
pos2 = np.stack(([values_LesothoN,values_SAN]),axis = 0)
print(np.cov(pos1))
print(np.cov(pos2))
rv = mvn([mean_Lesotho,mean_SA],np.cov(pos1))
rv1 = mvn([mean_LesothoN,mean_SAN],np.cov(pos2))

x1,y1 = np.mgrid[50:350:0.5, 100:200:0.5]
posnew = np.dstack((x1,y1))


levels = [0.01,0.05, 0.10, 0.50, 0.90,0.95,0.99]
levels1 = [0.88,0.90, 0.92, 0.94, 0.96,0.97, 0.98, 0.99,0.991, 0.992, 0.993, 0.994,0.995, 0.996, 0.997, 0.998, 0.999]


kendal = kdtau(values_Lesotho,values_SA)
spearman = spearman(values_Lesotho,values_SA)

####
data_normalized = pd.DataFrame({'val-L':values_Lesotho-np.mean(values_Lesotho),'val-SA': values_SA-np.mean(values_SA)})
data_normalized_N = pd.DataFrame({'val-L':values_LesothoN-np.mean(values_LesothoN),'val-SA': values_SAN-np.mean(values_SAN)})

data_small = data_normalized[data_normalized['val-L']<0 & (data_normalized['val-SA'] < 0)]
data_small1 = data_normalized_N[data_normalized_N['val-L']<0 & (data_normalized_N['val-SA'] < 0)]
print(np.corrcoef(data_small['val-L'],data_small['val-SA']))
print(np.corrcoef(data_small1['val-L'],data_small1['val-SA']))


fig, ax = plt.subplots()
ax.plot(values_Lesotho,values_SA,'ro',markersize='2.0', alpha = 0.7)
ax.plot(values_LesothoN,values_SAN,'bo', markersize='2.0', alpha = 0.7)
CS = ax.contour(x, y, (1-rv.cdf(pos)),levels, colors = 'r')
ax.clabel(CS,inline=1, fontsize=7,fmt = '%1.2f',colors = 'r')

CS1 = ax.contour(x, y, (1-rv1.cdf(pos)),levels, colors = 'b')
ax.clabel(CS1,inline=1, fontsize=7,fmt = '%1.2f',colors = 'b')


ax.plot(274, 172, 'ko')
ax.plot([274,274], [148, 193], 'k')
ax.plot([224, 307],[172, 172],'k')
ax.text(600,550,'$\\rho$ = 0.73',color = 'r', fontsize = '12')
ax.text(600,530,'$\\rho$ = 0.73',color = 'b', fontsize = '12')
#plt.minorticks_on()
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')

plt.legend(['ACT', 'NAT'], markerscale=3., loc = 'lower right', frameon=False)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,
    labelsize = '12',
    labelbottom=True) # labels along the bottom edge are off

plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    right=False,      # ticks along the bottom edge are off
    left=True,
    labelsize = '12', # ticks along the top edge are off
    labelleft=True)

plt.xlim(100,700)
plt.ylim(0,600)
plt.xlabel('JFM total precipitation Lesotho (mm)', fontsize = '12')
plt.ylabel('JFM total precipitation South Africa (mm)',fontsize = '12')

plt.tight_layout()
plt.savefig('correlation.pdf', orientation='portrait', format='pdf')


#####----- Zoom in ######


fig, ax = plt.subplots()
ax.plot(values_Lesotho,values_SA,'ro',markersize='2.0')
ax.plot(values_LesothoN,values_SAN,'bo', markersize='2.0')
CS = ax.contour(x1, y1, (1-rv.cdf(posnew)),levels1, colors = 'r')
ax.clabel(CS,inline=1, fontsize=6,fmt = '%1.3f',colors = 'r')

CS1 = ax.contour(x1, y1, (1-rv1.cdf(posnew)),levels1, colors = 'b')
ax.clabel(CS1,inline = 1, fontsize=6,fmt = '%1.3f',colors = 'b')


ax.plot(277.5, 175.5, 'ko')
ax.plot([277.5,277.5], [141, 193.5], 'k')
ax.plot([224.5, 310.5],[175, 175],'k')
ax.text(600,550,'$\\rho$ = 0.60',color = 'r', fontsize = '12')
ax.text(600,530,'$\\rho$ = 0.58',color = 'b', fontsize = '12')
plt.minorticks_on()
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,
    labelsize = '9',
    labelbottom=True) # labels along the bottom edge are off

plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    right=False,      # ticks along the bottom edge are off
    left=True,
    labelsize = '9', # ticks along the top edge are off
    labelleft=True)

plt.xlim(200,350)
plt.ylim(120,200)
plt.xlabel('JFM total precipitation Lesotho (mm)', fontsize = '10')
plt.ylabel('JFM total precipitation South Africa (mm)',fontsize = '10')


#plt.show()
