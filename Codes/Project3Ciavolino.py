import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RadioButtons, Slider
from numpy import inf
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.colors as color
                                                                                                    #------------ PARAMETERS --------------
v = 3.0*(10**10)
mu_a_0 = 0.10
mu_s = 10
dt = np.arange(10**-10,0.8*10**-8,10**-9)
D=1/(3*mu_s)
M = []
w = []
det=[]
vox=[]
N = 8  
P = 10**6 
                                                                                                     #------- COORDINATES CREATION --------
for k in range(N):
    for j in range(N):
        for i in range(N):
            if k == 0:
                det.append((j,i))
            vox.append((j,i,k))
                                                                                                     #-------- SENSITIVITY MATRIX ---------
def Contr(x_sd,y_sd,z_sd,x_p,y_p,z_p,dt):
    ro = np.sqrt(abs(x_sd-x_p)**2+abs(y_sd-y_p)**2+abs(z_sd-z_p)**2)
    return float((1)/(2*np.pi*D*ro))*np.exp(ro**2/(-D*v*dt)) 

for x_sd,y_sd in det:
    for t in dt:
        for x_p,y_p,z_p in vox:
            w.append(Contr(x_sd,y_sd,0,x_p,y_p,z_p,t))            # 1D sensitivity matrix
W = np.array(w).reshape(len(dt)*len(det),len(vox))                # 2D sensitivity matrix

W[W==inf]=10
#W=W/5                                                                                                     #-------------  MEASURE -------------
A = [0]*512
A[110]=0.1
M = W.dot(np.array(A))
M_noise = (np.random.poisson(P*(1+M))-np.random.poisson(P))/np.random.poisson(P)
             
                                                                                        #-------------- SVD -----------------
u,d,v = np.linalg.svd(W)
sing_val = np.diag(d)  
inv_sing_val = np.diag(d**(-1))
right_inv_sing_vect =v.transpose()
left_inv_sing_vect = u.transpose()
                                                                                                      #--------- REGULARIZATIONS ----------
                                                                                                      #-------------- TSVD ----------------
tr_inv_sing_val = np.empty_like(inv_sing_val) 
tr_inv_sing_val[:]= inv_sing_val
tr_inv_sing_val[160:]=0   
                                                                                                      #------------- ARRIDGE --------------
l = 0.001*max(d)
def Arridge(W,M,l):                      
   W_arr = np.empty_like(W) 
   W_arr[:]= W
   W_arr = np.concatenate((W_arr,l*np.eye(len(A))))
   M_arr = np.empty_like(M) 
   M_arr[:]= M
   M_arr = np.concatenate((M_arr,np.zeros(len(A))))
   return (W_arr,M_arr)
                                                                                                      #--------- RECONSTRUCTION  ----------
A_ideal = right_inv_sing_vect.dot(inv_sing_val).dot(left_inv_sing_vect).dot(M)
A_rec_TSVD = right_inv_sing_vect.dot(tr_inv_sing_val).dot(left_inv_sing_vect).dot(M_noise)
A_rec_ARR = np.linalg.lstsq(Arridge(W,M_noise,l)[0],Arridge(W,M_noise,l)[1])
                                                                                                      #------------ 3D PLOTS  -------------
f6 = plt.figure(6)
ax3D = f6.add_subplot(1, 1, 1,projection='3d')
d=ax3D.scatter([x[0] for x in vox],[x[1] for x in vox],[x[2] for x in vox], s=100,c=A_rec_ARR[0],marker='o',cmap='YlOrRd',alpha=0.7,norm = color.Normalize(vmin=0.,vmax=0.1))
cbar = plt.colorbar(d,ax=ax3D)                                                       # 3D scatter plot
cbar.set_label('Absorption perturbation [$cm^{-1}$]')
ax3D.set_title('3D Reconstruction')

f7 = plt.figure(7)
x,y,z = np.indices((N,N,N))
Vox = (x > N) & (y > N) & (z > N)
Vox[6,5,1]=True
ax3Did = f7.add_subplot(2, 2, 1,projection='3d')
ax3Did.voxels(Vox, facecolors='r', edgecolor='gray')                                 # 3D voxel plot 
ax3Did.set_title('Ideal 3D Reconstruction')
                                                                                                      #-------------- PLOTS  --------------

ax = f7.add_subplot(212)
ax.plot(A_rec_ARR[0])
ax.set_title('Compressed Reconstruction')
axA = plt.axes([0.56, 0.005, 0.4, 0.020])
slideA = Slider(axA,'A pos',128,192,valinit=0,valstep=1)

f5, axes1 = plt.subplots(2, 4)
A_rec = []
for i in range(8):
    A_r = axes1[int(i%2),int(i/2)].imshow(A_rec_ARR[0].reshape(8,8,8)[i-1,:,:],norm = color.Normalize(vmin=0.,vmax=0.1))
    A_rec.append(A_r)
    axes1[int(i%2),int(i/2)].set_title('Z='+str(i+1)+'cm')
    axes1[int(i%2),int(i/2)].tick_params(top=False ,bottom=False,left=False,right=False, labelbottom=False,labelleft=False )
    axes1[0,0].set_ylabel('Y  [cm]')
    axes1[0,0].set_xlabel('X  [cm]')
f5.suptitle('Tomographies')

plt.figure(12)
ax12=plt.axes()
ax12.plot(M_noise)
ax12.set_title('$M_{noise}$')

#plt.figure(1)
#plt.plot(A_ideal)
#plt.title('Compressed ideal A vector')
#plt.plot(sing_val)
#plt.yscale('log')
#plt.figure(112)
#plt.plot(W)
#plt.title('$W_{ideal}$') 

                                                                                                       #----------- SLIDERS  --------------
def updateIdeal(a):
    A = [0]*512
    A[int(slideA.val)]=0.1
    Vox[:,:,:]=False
    Vox[int(slideA.val%8),int(slideA.val%64/8),int(slideA.val/64)]=True
    ax3Did.cla()
    ax3Did.voxels(Vox, facecolors='r', edgecolor='gray')                                
    M = W.dot(np.array(A))
    M_noise = (np.random.poisson(P*(1+M))-np.random.poisson(P))/np.random.poisson(P)
    A_ideal = right_inv_sing_vect.dot(inv_sing_val).dot(left_inv_sing_vect).dot(M)
    #for j in range(8):
    #     A_id[j].set_data(A_ideal.reshape(8,8,8)[j,:,:])
    #f4.canvas.draw()
    A_rec_ARR = np.linalg.lstsq(Arridge(W,M_noise,l)[0],Arridge(W,M_noise,l)[1])
    A_rec_TSVD = right_inv_sing_vect.dot(tr_inv_sing_val).dot(left_inv_sing_vect).dot(M_noise)
    ax.cla()
    ax.plot(A_rec_ARR[0])
    ax.set_title('Compressed Reconstruction')
    ax12.cla()
    ax12.plot(M_noise)
    ax12.set_title('$M_{noise}$')
    for q in range(8):
         A_rec[q].set_data(A_rec_ARR[0].reshape(8,8,8)[q,:,:])
    f5.canvas.draw()
    f7.canvas.draw()
    ax3D.cla()
    ax3D.scatter([x[0] for x in vox],[x[1] for x in vox],[x[2] for x in vox], s=100,c=A_rec_ARR[0],cmap='YlOrRd',marker='o',alpha=0.7,norm = color.Normalize(vmin=0.,vmax=0.1))
    ax3D.set_title('3D Reconstruction')
    f6.canvas.draw()
slideA.on_changed(updateIdeal)

                                                                                                       #------ singular vectors plot -------


f9 = plt.figure(9)
axv = f9.add_subplot(111)
lft_sing_vect = u[:,0].reshape(len(det),len(dt))
im=axv.imshow(lft_sing_vect[:,0].reshape(8,8),norm = color.Normalize(vmin=-0.012,vmax=0.012),cmap='seismic')
cbar = plt.colorbar(im)
plt.title('Order 0')

ax0 = plt.axes([0.05, 0.03, 0.4, 0.020])
slide0 = Slider(ax0,'ORDER',0,40,valinit=0,valstep=1)
ax1 = plt.axes([0.56, 0.03, 0.4, 0.020])
slide1 = Slider(ax1,'TIME',0,7,valinit=0,valstep=1)

def update(a):
    lft_sing_vect = u[:,int(slide0.val)].reshape(len(det),len(dt))
    im.set_data(lft_sing_vect[:,int(slide1.val)].reshape(8,8))
    axv.set_title('Order '+str(int(slide0.val)))
    f9.canvas.draw()
slide0.on_changed(update)
slide1.on_changed(update)
plt.show()
                                                                                                       #-------- ideal tomographies --------

f4, axes0 = plt.subplots(2, 4)
A_id = []
for i in range(8):
    A_i = axes0[int(i%2),int(i/2)].imshow(A_ideal.reshape(8,8,8)[i-1,:,:],norm = color.Normalize(vmin=0,vmax=0.1))
    A_id.append(A_i)
    axes0[int(i%2),int(i/2)].set_title('Z='+str(i+1)+'cm')
    axes0[int(i%2),int(i/2)].tick_params(top=False ,bottom=False,left=False,right=False, labelbottom=False,labelleft=False )
    axes0[0,0].set_ylabel('Y  [cm]')
    axes0[0,0].set_xlabel('X  [cm]')
f4.suptitle('Ideal Tomographies')

