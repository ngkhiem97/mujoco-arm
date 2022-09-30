import numpy as np
from numpy.linalg import inv
from scipy.linalg import logm, norm, sqrtm
from pyquaternion import Quaternion
from cvxopt import matrix, solvers
import sys


class VelocityController:
    def __init__(self):
        self.ex,self.ey,self.ez,self.n,self.P,self.q,self.H,self.types,self.dq_bounds,self.q_bounds = self.robotParams()
        self.R_EE = np.array([[0,1,0],[0,0,1],[1,0,0]])
        self.er = 0.005
        self.ep = 0.0005
    
    def fwdkin_alljoints_gen3(self, q):
        P = self.P
        H = self.H

        # initialize utility variables
        R=np.eye(3)
        p=np.zeros((3,1))
        RR = np.zeros((3,3,self.n+1))
        pp = np.zeros((3,self.n+1))
        # forward kinematics
        for i in range(self.n):
            h_i = H[0:3,i]
            
            if self.types[0][i] == 0: #rev
                pi = P[0:3,i].reshape(3, 1)
                p = p+np.dot(R,pi)
                Ri = self.rot(h_i,q[i])
                R = np.dot(R,Ri)
            elif self.types[0][i] == 1: #pris
                pi = (P[:,i]+q[i]*h_i).reshape(3, 1)
                p = p+np.dot(R,pi)
            else: # default pris
                pi = (P[:,i]+q[i]*h_i).reshape(3, 1)
                p = p+np.dot(R,pi)
            
            pp[:,[i]] = p
            RR[:,:,i] = R
        
        # end effector T
        p=p+np.dot(R, P[0:3,self.n].reshape(3, 1))
        pp[:,[self.n]] = p
        RR[:,:,self.n] = np.dot(R, self.R_EE)
        
        return pp, RR
        
    def getJacobian_world_gen3(self, q):
        num_joints = self.n 
        H = self.H
        # Compute Forward Kinematics
        P_0_i = np.zeros((3,num_joints+1))
        R_0_i = np.zeros((3,3,num_joints+1))

        P_0_i,R_0_i=self.fwdkin_alljoints_gen3(q)
        
        P_0_T = P_0_i[:,num_joints]
        
        # Compute Jacobian
        J = np.zeros((6,num_joints))
        
        i = 0
        
        for i in range(num_joints):
            if self.types[0][i] == 0:
                J[:,i] = np.hstack((np.dot(R_0_i[:,:,i],H[:,i]), np.dot(self.hat(np.dot(R_0_i[:,:,i], H[:,i])), P_0_T - P_0_i[:,i])))
        
        return J
    
    def getJacobian_task_gen3(self, q):
        num_joints = self.n 
        H = self.H
        # Compute Forward Kinematics
        P_0_i = np.zeros((3,num_joints+1))
        R_0_i = np.zeros((3,3,num_joints+1))

        P_0_i,R_0_i=self.fwdkin_alljoints_gen3(q)
        
        P_0_T = P_0_i[:,num_joints]
        
        # Compute Jacobian
        J = np.zeros((6,num_joints))
        
        i = 0
        
        for i in range(num_joints):
            if self.types[0][i] == 0:
                J[:,i] = np.hstack((np.dot(R_0_i[:,:,i],H[:,i]), np.dot(self.hat(np.dot(R_0_i[:,:,i], H[:,i])), P_0_T - P_0_i[:,i])))
        
        J = np.dot(np.vstack((np.hstack((R_0_i[:,:,num_joints].T, np.zeros((3,3)))), np.hstack((np.zeros((3,3)), R_0_i[:,:,num_joints].T)))), J)
        
        return J
    
    def get_joint_vel_worldframe(self, twist_ee, q, vinit): 
        
        if(np.sum(np.absolute(twist_ee)) <= 0.0000001):
            return(np.zeros((1,self.n))[0])
            
        J = self.getJacobian_world_gen3(q)  
        pos_v = twist_ee[0:3].reshape(3,1)
        ang_w = twist_ee[3:6].reshape(3,1)        
        
        return(self.get_joint_vel(J, pos_v, ang_w, q, vinit))
    
    def get_joint_vel_taskframe(self, twist_ee, q, vinit): 
        if(np.sum(np.absolute(twist_ee)) <= 0.0000001):
            return(np.zeros((1,self.n))[0])
           
        J = self.getJacobian_task_gen3(q)        
        pos_v = twist_ee[0:3].reshape(3,1)
        ang_w = twist_ee[3:6].reshape(3,1)        
        
        return(self.get_joint_vel(J, pos_v, ang_w, q, vinit))
        
    def getqp_H(self, J, vr, vp):
        n = self.n
        er = self.er
        ep = self.ep
        
        H1 = np.dot(np.hstack((J,np.zeros((6,2)))).T,np.hstack((J,np.zeros((6,2)))))
        
        tmp = np.vstack((np.hstack((np.hstack((np.zeros((3,n)),vr)),np.zeros((3,1)))),np.hstack((np.hstack((np.zeros((3,n)),np.zeros((3,1)))),vp)))) 
        H2 = np.dot(tmp.T,tmp)

        H3 = -2*np.dot(np.hstack((J,np.zeros((6,2)))).T, tmp)
        H3 = (H3+H3.T)/2
        
        tmp = np.vstack((np.hstack((np.zeros((1,n))[0],np.sqrt(er),0)), np.hstack((np.zeros((1,n))[0],0,np.sqrt(ep)))))
        H4 = np.dot(tmp.T, tmp)

        H = 2*(H1+H2+H3+H4)

        return H
    
    def getqp_f(self):
        n = self.n
        er = self.er
        ep = self.ep
        f = -2*np.hstack((np.zeros((1,n))[0],er,ep)).T
        return f    
    
    def get_joint_vel(self, J, pos_v, ang_w, q, vinit):
        n= self.n

        # params for qp 
        H = self.getqp_H(J, ang_w, pos_v)        
        f = self.getqp_f()
        
        # bounds for qp
        bound = self.dq_bounds[0, :]

        LB = np.vstack((-0.1*bound.reshape(n, 1),0,0))
        UB = np.vstack((0.1*bound.reshape(n, 1),1,1))
        
        H = matrix(H, tc='d')
        f = matrix(f, tc='d')
        LB = matrix(LB, tc = 'd')
        UB = matrix(UB, tc = 'd')
        
        #add joint limits   
        q_lim_upper = self.q_bounds[:,1]
        q_lim_lower = self.q_bounds[:,0]
        
        k1_qlim_thr = 0.8 # when should the near-joint-limit slowing down kicks in.
        k2_qlim_thr = 0.9 # when should the near-joint-limit push-off kicks in.
        ita = 1-k2_qlim_thr # Level-2 budget threshold
        epsilon = k2_qlim_thr-k1_qlim_thr # Level-1 budget threshold
        
        ub_check = 1-q/q_lim_upper
        lb_check = 1-q/q_lim_lower
        
        #check threshold 1
        ub_ck_idx_1 = np.logical_and(ub_check <= ita+epsilon, ub_check >ita)
        lb_ck_idx_1 = np.logical_and(lb_check <= ita+epsilon, lb_check >ita)
        
        #check threshold 2
        ub_ck_idx_2 = np.logical_and(ub_check <= ita, ub_check >0)
        lb_ck_idx_2 = np.logical_and(lb_check <= ita, lb_check >0)
        
        #check negative situation
        ub_ck_idx_neg = ub_check <=0
        lb_ck_idx_neg = lb_check <=0
        
        #prepare A and b
        c = 0.9
        e = 0.5*bound.min()
        
        A_neg = np.zeros((1, n+2)) #QP requires A*x<=b
        b_neg = np.zeros((1, n+2)) #here we use A_neg*x>=b_neg

        A_neg[0][0:n] = 1
        b_neg[0][0:n] = -np.tan(c*np.pi/2)

        if np.sum(ub_ck_idx_neg) >= 1: #Infeasible start, upper bound, push to the negative direction
            A_neg[0][np.hstack((ub_ck_idx_neg, False, False))] = -1
            b_neg[0][np.hstack((ub_ck_idx_neg, False, False))] = e

        if np.sum(lb_ck_idx_neg)>=1: #Infeasible start, lower bound, push to the positive direction
            A_neg[0][np.hstack((lb_ck_idx_neg, False, False))] = 1
            b_neg[0][np.hstack((lb_ck_idx_neg, False, False))] = e
        
        if np.sum(ub_ck_idx_2)>=1: #level-2 upper bound, push to the negative direction
            A_neg[0][np.hstack((ub_ck_idx_2, False, False))] = -1
            for idx_tmp in range(n):
                if ub_ck_idx_2[idx_tmp] == True:
                    b_neg[0][idx_tmp] = e*(ita-ub_check[idx_tmp])/ita

        if np.sum(lb_ck_idx_2)>=1: #level-2 lower bound, push to the positive direction
            A_neg[0][np.hstack((lb_ck_idx_2, False, False))] = 1
            for idx_tmp in range(n):
                if lb_ck_idx_2[idx_tmp] == True:                    
                    b_neg[0][idx_tmp] = e*(ita-lb_check[idx_tmp])/ita

        if np.sum(ub_ck_idx_1)>=1: #level-1 upper bound, positive dir slow down
            A_neg[0][np.hstack((ub_ck_idx_1, False, False))] = 1
            for idx_tmp in range(n):
                if ub_ck_idx_1[idx_tmp] == True:
                    b_neg[0][idx_tmp] = -np.tan(c*np.pi*(ub_check[idx_tmp]-ita)/2/epsilon)

        if np.sum(lb_ck_idx_1)>=1: #level-1 lower bound, negative dir slow down
            A_neg[0][np.hstack((lb_ck_idx_1, False, False))] = -1
            for idx_tmp in range(n):
                if lb_ck_idx_1[idx_tmp] == True:
                    b_neg[0][idx_tmp] = -np.tan(c*np.pi*(lb_check[idx_tmp]-ita)/2/epsilon)
        
        
        """
        
         
        """   
        
        # inequality constraints        
        A_neg_tmp = np.zeros((n+2,n+2))
        np.fill_diagonal(A_neg_tmp,A_neg)
        A_neg = A_neg_tmp[0:n][:]
        b_neg = b_neg[0][0:n].T
        
        A = matrix([matrix(np.eye(n+2), tc='d'), matrix(-np.eye(n+2), tc='d'), matrix(-A_neg, tc='d')])           
        b = matrix([UB, -LB, matrix(-b_neg, tc='d')]) # Upper bound(joints + er + ep), lower bound(joints + er + ep), q_lim bounds(joints)
        
        # solve        
        solvers.options['show_progress'] = False
        
        init_guess = matrix(np.hstack((vinit)), tc='d')
        sol = solvers.qp(H,f,A,b,initvals=init_guess)
        dq_sln = sol['x']

        return dq_sln
    
    def robotParams(self):
        # basic parameters
        I3 = np.eye(3)
        ex, ey, ez = I3[:,0], I3[:,1], I3[:,2]        
        h1 = -ez
        h2 = ey
        h3 = -ez
        h4 = ey
        h5 = -ez
        h6 = ey
        h7 = -ez
        P = 0.01*np.array([[0,0,156.4], [0,-5.4,128.4], [0,-6.4,210.4], [0,-6.4,210.4], [0,-6.4,208.4], [0,0,105.9], [0,0,105.9],[0,0,61.5+155]]).T
        n = 7
        q = np.zeros((n, 1))
        H = np.array([h1, h2, h3, h4, h5, h6, h7]).T
        types = np.zeros((1, n))

        # bounds
        dq_lim = 0.8727 # velocity limit: 50 degrees/s
        dq_bounds = 0.9*dq_lim*np.ones((2, n))
        q_bounds = np.array([[-359.99,359.99],[-128.9,128.9],[-359.99,359.99],[-147.8,147.8],[-359.99,359.99],[-120.3,120.3],[-359.99,359.99]])*np.pi/180
        
        return ex,ey,ez,n,P,q,H,types,dq_bounds,q_bounds
    
    def rot(self, h, q):
        h=h/norm(h)
        R = np.eye(3) + np.sin(q)*self.hat(h) + (1 - np.cos(q))*np.dot(self.hat(h), self.hat(h))
        return R
        
    def hat(self, h):
        h_hat = np.array([[0, -h[2], h[1]], [h[2], 0, -h[0]], [-h[1], h[0], 0]])
        return h_hat
        
# quaternion multiplication
def quatmultiply(q1, q0):
    w0, x0, y0, z0 = q0[0][0], q0[0][1], q0[0][2], q0[0][3]
    w1, x1, y1, z1 = q1[0][0], q1[0][1], q1[0][2], q1[0][3]
    
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64).reshape(1, 4)

# convert a unit quaternion to angle/axis representation
def quat2axang(q):

    s = norm(q[0][1:4])
    if s >= 10*np.finfo(np.float32).eps:
        vector = q[0][1:4]/s
        theta = 2*np.arctan2(s,q[0][0])
    else:
        vector = np.array([0,0,1])
        theta = 0
    axang = np.hstack((vector,theta))
    
    return axang