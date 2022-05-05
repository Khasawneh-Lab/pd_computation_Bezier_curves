
"""
Author: Melih Can Yesilli
Date: 5/2/22
contact: yesillim@msu.edu

Description:This library includes the codes to compute approximated persistence diagrams using Bezier curve approach
explained in S. Tsuji and K. Aihara, “A fast method of computing persistent homology of time series data,” 
in ICASSP 2019- 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), IEEE, may 2019.

"""
import numpy as np
from scipy.special import comb
from scipy.linalg import lu_factor, lu_solve
from math import pi
from sympy import Point, Line, Line2D,Point2D,symbols,Poly
from itertools import combinations
from scipy.optimize import shgo
from ripser import ripser
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def Bezier_Curve(group,t,dim):
    """
    
    Parameters
    ----------
    group : 2D np.array
        The matrix that includes the points after dividing the point cloud into groups
    t : np.array
        The array that includes the values for parameterization variable, t.
    dim: int
        embedding dimension of the point cloud or group of points
    Returns
    -------
    P : 2D np.array
        Bezier curves (points for the curves)
    p : 2D np.array
        control points for each Bezier curves
    A : 2D np.array
        The matrix in the matrix form of fitting Bezier curves. See original paper for more details
    b : 2D np.array
        b in matrix form of fitting Bezier curves. See original paper for more details
    """
    A = np.zeros((4,4))
    b = np.zeros((4,dim))
    for i in range(4):
        for j in range(4):
            SUM1=0
            
            for k in range(len(group)):
                summ1 = ((1-t[k])**(6-i-j))*(t[k]**(i+j))
                SUM1 = SUM1+summ1
            A[i,j] = comb(3,i)*comb(3,j)*SUM1 
            
        for j in range(dim):
            SUM2=0
            for k in range(len(group)):
                if j <len(group[0]):
                    summ2 = ((1-t[k])**(3-i))*(t[k]**i)*(group[k,j])
                else:
                    summ2 = 0 
                SUM2 = SUM2+ summ2
            b[i,j] = comb(3,i)*SUM2
    
    #solve for the p matrix (includes control points)
    lu, piv = lu_factor(A)
    p= np.zeros((4,dim))
    for j in range(dim):
        p[:,j] = lu_solve((lu, piv), b[:,j])
    
    #compute the Bezier curves based on the found         
    P=np.zeros((len(group),dim))
    for i in range(len(group)):
        for j in range(dim):
            P[i,j] = ((1-t[i])**3)*p[0,j]+3*((1-t[i])**2)*t[i]*p[1,j]+3*(1-t[i])*(t[i]**2)*p[2,j]+(t[i]**3)*p[3,j]
    return P,p,A,b

def BC_coeffs(embedded_data,n_samples,spg,r,dim):
    """
    
    Parameters
    ----------
    embedded_data : 2D np.array
        Point cloud
    n_samples : int
        number of points in the point cloud
    spg : int
        samples per group (number of points in each group in point cloud)
    r : int
        Number of line segments for each group
    dim : int
        embedding dimension
    Returns
    -------
    param_coeffs : 2D np.array
        The coefficients for the lines generated using spg and r parameters.
    """       
    # number of points in point cloud
    n_samples = len(embedded_data) 
    # group number
    group_number = int(n_samples/spg)
    # parametrization variable
    t = np.linspace(0,1,spg)
    # Bezier curves
    Curves = np.zeros(shape=(group_number),dtype=object)
    
    # parameterization variable for generating line segments
    t_new = np.linspace(0,1,r+1)
    
    # generate arrays that will store points and Bezier curve coeffcients
    # and line segments
    all_segment_points = []
    segment_points = np.zeros((group_number,len(t_new)),dtype=object)
    param_coeffs = np.zeros((group_number*r,dim*2))        
    
    inc=0
    for i in range(group_number):
        # generate the group
        group = embedded_data[i*spg:(i+1)*spg,:]
        
        # fit Bezier Curves
        Bez_Curve = Bezier_Curve(group,t,dim)
        P = Bez_Curve[0]
        Curves[i]=P
        p = Bez_Curve[1]
        A = Bez_Curve[2]
        b = Bez_Curve[3]
        
        # End points of line segments
        segment_points= np.zeros((r+1,1))
        for dimension in range(0,dim):
            segments_end_pts_d=np.interp(t_new,t,Curves[i][:,dimension])
            segment_points = np.concatenate((np.reshape(segment_points,((r+1,dimension+1))),np.reshape(segments_end_pts_d,((r+1,1)))),axis=1)
        segment_points = segment_points[:,1:]
        
        # generate the lines and parametrize the lines
        for k in range(r):
            p1 = segment_points[k,:]
            p2 = segment_points[k+1,:]
            
            all_segment_points.append(p1)
            all_segment_points.append(p2)
            
            p1, p2 = Point(p1),Point(p2)
            l1 = Line(p1, p2)
            l1.plot_interval()
            dim = l1.ambient_dimension
            
            
            for d in range(dim):
                param= l1.arbitrary_point()[d]
                polynom = Poly(param)
                coefficients = polynom.coeffs()
                # exec('param%d= l1.arbitrary_point()[%d]' %(d+1,d))
                # exec('polynom = Poly(param%d)'%(d+1))
                # exec('coefficients = polynom.coeffs()')
                for l in range (2):
                    param_coeffs[i*r+k,2*d+l] = float(coefficients[l])
                    inc=inc+1 
            print('spg:{},r:{}----> In progress:%{}'.format(spg,r,inc/(group_number*r*2*dim)*100))
        
    # end points of each segment
    segment_ends = np.asarray(all_segment_points)
    segment_edns_unique = np.unique(segment_ends,axis=0)
    indexes = np.unique(segment_ends,axis=0, return_index=True)[1]
    segment_ends_unique=[segment_ends[index] for index in sorted(indexes)]
    unique_segment_ends = np.zeros((len(segment_ends_unique),dim))
    for i in range (len(segment_ends_unique)):
       unique_segment_ends[i,:]=segment_ends_unique[i]
    
    return param_coeffs          

        


def compDist_Segments(param_coeffs):
    """
    
    Parameters
    ----------
    param_coeffs : 2D np.array
        Coefficients for line segments generated using Bezier curves
    Returns
    -------
    distance_mat : 2D np.array
        Distance matrix between line segments generated using spg and r parameters
    """
    # generate the combinations between line segments
    A = np.linspace(0,len(param_coeffs)-1,len(param_coeffs)).astype(int)
    combination = list(combinations(A,2))
    
    # generate the pairwise distance matrix
    distance = np.zeros((len(param_coeffs),len(param_coeffs)))


    for kk in range(len(combination)):
        
        comb = combination[kk]
        coeff1 = param_coeffs[comb[0]]
        coeff2 = param_coeffs[comb[1]] 
        
        # distance function to minimize
        def func2d(init_guess):
            df_ds_sum = np.zeros(dim+1)
            df_dt_sum = np.zeros(dim+1)
            
            for d in range(dim):
                df_ds = -2*coeff1[2*d]*(coeff2[2*d]*init_guess[1]+coeff2[1+2*d]-coeff1[2*d]*init_guess[0]-coeff1[1+2*d])
                df_dt = 2*coeff2[2*d]*(coeff2[2*d]*init_guess[1]+coeff2[1+2*d]-coeff1[2*d]*init_guess[0]-coeff1[1+2*d])
                df_ds_sum[d+1] = df_ds_sum[d] + df_ds
                df_dt_sum[d+1] = df_dt_sum[d] + df_dt
            df = np.zeros(2)
            df[0]=df_ds_sum[-1] 
            df[1]=df_dt_sum[-1]
            
            dist_sum = np.zeros(dim+1)
            for d in range(dim):
                dist = (coeff1[2*d]*init_guess[0]+coeff1[1+2*d]-coeff2[2*d]*init_guess[1]-coeff2[1+2*d])**2
                dist_sum[d+1] = dist_sum[d] + dist
            
            f = dist_sum[-1]
            
            return f 
        
        bounds = [(0, 1), (0, 1)]
    
        result = shgo(func2d, bounds, n=30, sampling_method='sobol')    
        init_guess = result.x
    
        distance[comb[0],comb[1]]=np.sqrt(func2d(init_guess)) 
        
        if (kk % 100)==0:
            print('spg:{},r:{},combinations:{}----> In progress: %{}'.format(spg,r,len(combination),(kk+1)/len(combination)*100))
        
    distance_mat = distance+distance.T 
    
    return distance_mat
          
def perDiag_Bezier(distance_mat):
    
    perDiag = ripser(distance_mat, coeff=2,distance_matrix=True,maxdim=2)['dgms']
    
    return perDiag

def plot_perDiag(perDiag,perDiag_GP,dim):

    fig = plt.figure(figsize=(15,5))
    
    # plot point cloud if dim is less than 3
    if dim==3:        
        ax = fig.add_subplot(1, 3, 1,projection='3d')
        ax.scatter(embedded_data[:,0],embedded_data[:,1],embedded_data[:,2],s=3)
        ax.set_xlabel(r'$x_{1}$',fontsize=20)
        ax.set_ylabel(r'$x_{2}$',fontsize=20)
        ax.set_zlabel(r'$x_{3}$',fontsize=20)
    elif dim==2:
        ax = fig.add_subplot(1, 3, 1)
        ax.scatter(embedded_data[:,0],embedded_data[:,1],s=3)
        ax.set_xlabel(r'$x_{1}$',fontsize=20)
        ax.set_ylabel(r'$x_{2}$',fontsize=20)
    
    # plot pd
    if 2<=dim<=3:
        ax1 = fig.add_subplot(1, 3, 2)
        ax2 = fig.add_subplot(1, 3, 3)
    else:
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        
    markers_styles = ['o','v','d']
    
    for i in range(3):
        ax1.scatter(perDiag[i][:,0],perDiag[i][:,1],s=20,marker=markers_styles[i])
        ax2.scatter(perDiag_GP[i][:,0],perDiag_GP[i][:,1],s=20,marker=markers_styles[i])
        
        
    ax1.plot([0,max(perDiag[1][:,1])],[0,max(perDiag[1][:,1])],'--k')
    ax1.set_xlabel('Birth Time',fontsize = 18)
    ax1.set_ylabel('Death Time', fontsize = 18)   
    ax1.tick_params(labelsize=18)
    ax1.legend(['$H_{0}$','$H_{1}$','$H_{2}$'], fontsize = 15,loc='lower right')    
    
    ax2.plot([0,max(perDiag[1][:,1])],[0,max(perDiag[1][:,1])],'--k')
    ax2.set_xlabel('Birth Time',fontsize = 18)
    ax2.set_ylabel('Death Time', fontsize = 18)   
    ax2.tick_params(labelsize=18)
    ax2.legend(['$H_{0}$','$H_{1}$','$H_{2}$'], fontsize = 15,loc='lower right')     
    
    fig.subplots_adjust(wspace=0.5)
    
    return fig

          
if __name__ == '__main__':
    n_samples = 2000
    R=2
    r_sys=1
    spg=10
    r=2
    dim=3
    
    t_s =np.linspace(0,50*pi,n_samples)

    u = t_s
    v = np.sqrt(2)*t_s
    
    #define the 3 dimensional irrotational flow
    x_1 = R*np.cos(u)+r_sys*np.cos(u)*np.cos(v)
    x_2 = R*np.sin(u)+r_sys*np.sin(u)*np.cos(v)
    x_3 = r_sys*np.sin(v)

    x_1 = np.reshape(x_1,(len(x_1),1))
    x_2 = np.reshape(x_2,(len(x_2),1))
    x_3 = np.reshape(x_3,(len(x_3),1))

    embedded_data = np.concatenate((x_1,x_2,x_3),axis=1)
    
    # compute the coefficients for the fitted curves
    param_coeffs = BC_coeffs(embedded_data,n_samples,spg,r,dim)
    
    # compute the distance matrix between line segments
    distance_mat = compDist_Segments(param_coeffs)
    
    # compute persistence diagrams using the distance matrix
    perDiag = perDiag_Bezier(distance_mat)
    # compute persistence diagrams using the traditional approach
    perDiag_GP = ripser(embedded_data, coeff=2,n_perm=500,maxdim=2)['dgms']
    
    # plot the persistence diagram and dimension
    fig = plot_perDiag(perDiag,perDiag_GP,dim)