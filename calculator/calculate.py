'''
此版为测试版 最新版
'''
import numpy as np
# from scipy.integrate import quad
import matplotlib.pyplot as plt
import math
import numpy as np
import imageio.v2 as imageio #用于生成gif 不需要可注释掉，同时is_save= False; 若安装的是imageio则`import imageio`
import os
from scipy.integrate import quad
from .wrapper import time_counter

class Calculator():
    '''
    用于计算指定翼型的气动特性\n
    v: 远场来流速度\n
    alpha: 迎角\n
    sep_loc: 分段点坐标\n
    name: 翼型名称，输出时使用\n
    method: 积分方式选择:0-分段积分,1-积分法\n
    '''
    def __init__(self, v: float = 1, alpha: float = 0, sep_loc: any = None, 
                 camber_func_derivative: any =  None,**kwargs) -> None: 
        if 'name' in kwargs.keys(): #翼型名称
            assert type(kwargs['name']) == str, 'name must be str'
            self.name = 'Naca:'+kwargs['name']+' ' 
        else:
            self.name = 'Naca: not given '
        
        if 'method' in kwargs.keys(): #积分方式选择
            assert type(kwargs['method']) == int, 'method must be int'
            assert kwargs['method'] in [0,1], 'method must be 0 or 1'
            self.method = kwargs['method']
        else:
            self.method = 1

        if camber_func_derivative is not None: #中弧线函数导数
            self.camber_func = camber_func_derivative
        else:
            self.camber_func = None
            print('warning: camber_func is None')

        #分界点坐标
        if sep_loc is None: #测试用，圆柱
            self.sep_loc = np.array([[math.cos(math.pi/100*i), math.sin(math.pi/100*i)] for i in range(0,200)])*0.3
        else:
            self.sep_loc = sep_loc
        
        self.number = np.size(self.sep_loc,0)   #分段数（点数）
        assert self.number%2 == 0, 'the number of sep_loc must be even'
        
        self.ctrl_loc = np.zeros((np.size(self.sep_loc,0), 2)) #控制点坐标
        # self.b = np.zeros((np.size(self.sep_loc,0), 2)) #斜率截距
        self.norm = np.zeros((np.size(self.sep_loc,0), 2)) #法向量
        self.length = np.zeros((np.size(self.sep_loc,0), 1)) #分段长度
        
        self.v = v #远场来流速度
        self.alpha = alpha #迎角
        self.v_x = None #x方向速度
        self.v_y = None #y方向速度
        
        self.p = None
        self.cl = None
        self.cm = 0 #力矩系数
        self.cl2 = np.zeros((1,2)) # 用于压力积分法计算的cl,cp
        self.cl3 = 0 #薄翼理论cl
        self.cm3 = 0 #薄翼理论cm

        self.ux = None #x方向速度矩阵
        self.uy = None  #y方向速度矩阵
        
        self.A = np.zeros((np.size(self.sep_loc,0), np.size(self.sep_loc,0))) #系数矩阵
        self.lambda_c = np.zeros((self.number+1, 1)) #lambda矩阵
        self.v_array = None #速度系数矩阵

        if 'del_points' in kwargs.keys(): #除去后缘点数，为测试用接口，默认不去除(不处理)
            assert type(kwargs['del_points']) == int, 'del_points must be int'
            assert kwargs['del_points'] < self.number//2, 'del_points must < number//2'
            self.del_points = kwargs['del_points']
        else:
            self.del_points = 0
        print('______Calculator start init______') 
        print('---naca-:',self.name,'\n---alpha:',self.alpha,'\n---v----:',self.v)
        self.__init()
        self.__calculate_lambda()
        self.__calculate_cp()
        self.__calculate_cl_cm()

        self.calculate_cl2() #压力积分法计算cl,cp
        self.thin_airfoil_theory() #薄翼理论计算cl,cm
        print('______Calculator finish init______')
    
    def __init(self):
        for i in range(np.size(self.sep_loc,0)):
            self.length[i] = np.linalg.norm(self.sep_loc[i]-self.sep_loc[i-1]) 
            self.ctrl_loc[i] = (self.sep_loc[i]+self.sep_loc[i-1])/2
            self.norm[i] = np.array([self.sep_loc[i,1]-self.sep_loc[i-1,1], self.sep_loc[i-1,0]-self.sep_loc[i,0]])/self.length[i]
            # self.kb[i,0] = (self.sep_loc[i,1]-self.sep_loc[i-1,1])/(self.sep_loc[i,0]-self.sep_loc[i-1,0]) #斜率
            # self.kb[i,1] = self.sep_loc[i,1]-self.kb[i,0]*self.sep_loc[i,0] #截距
        self.v_x = self.v*math.cos(np.radians(self.alpha))  #θ->rad
        self.v_y = self.v*math.sin(np.radians(self.alpha))

    @time_counter
    def __calculate_lambda(self):
        '''
        计算系数矩阵A\n
        计算lambda及c
        '''
        for i in range(self.number):
            for j in range(self.number):

                # 该段为核心代码，采用分段积分
                if self.method == 0:
                    num = 50
                    dx = (self.sep_loc[j,0]-self.sep_loc[j-1,0])/num
                    dy = (self.sep_loc[j,1]-self.sep_loc[j-1,1])/num
                    for x,y in zip(np.linspace(self.sep_loc[j-1,0], self.sep_loc[j,0], num),\
                                    np.linspace(self.sep_loc[j-1,1], self.sep_loc[j,1], num)):
                        self.A[i,j] += 0.5*math.log((x-self.ctrl_loc[i,0])**2 + (y-self.ctrl_loc[i,1])**2)*(dx**2+dy**2)**0.5

                # 该段为核心代码，采用积分法
                elif self.method == 1:
                    norm1 = np.array([0,1])
                    norm2 = self.norm[j]
                    angle = math.acos(np.dot(norm1,norm2)/(np.linalg.norm(norm1)*np.linalg.norm(norm2)))
                    angle = angle if norm2[0] > 0 else -angle
                    rotat_matrix = np.array([[math.cos(angle), -math.sin(angle)],
                                            [math.sin(angle), math.cos(angle)]])
                    sep_loc1 = np.dot(rotat_matrix, self.sep_loc[j-1])
                    sep_loc2 = np.dot(rotat_matrix, self.sep_loc[j])
                    ctrl_loc = np.dot(rotat_matrix, self.ctrl_loc[i])
                    distance = lambda x: np.linalg.norm(np.array([x,sep_loc1[1]])-ctrl_loc)
                    if i==j:
                        self.A[i,j] = quad(lambda x: math.log(distance(x)), sep_loc2[0], ctrl_loc[0])[0]+\
                                        quad(lambda x: math.log(distance(x)), ctrl_loc[0], sep_loc1[0])[0]
                    else:
                        self.A[i,j] = quad(lambda x: math.log(distance(x)), sep_loc2[0], sep_loc1[0])[0]
        # print('finish calculate A:\n', self.A)
        # 目前定义后缘为计数起点
        func = lambda i: self.v_x*self.ctrl_loc[i,1] - self.v_y*self.ctrl_loc[i,0]
        self.v_array = np.array([func(i) for i in range(self.number)])\
                        .reshape(self.number,1)

        A_temp = np.hstack((self.A/(math.pi*2), np.ones((self.number,1))))
        temp_v_array = np.vstack((-self.v_array, np.array([[0]])))
        
        #后缘点库塔条件
        append_line = np.zeros((1, self.number+1))
        append_line[0,-2] = 1
        append_line[0,0] = 1 
        
        A_temp = np.vstack((A_temp, append_line))
        # print(A_temp)

        self.lambda_c = np.linalg.solve(A_temp, temp_v_array)
        # print('finish calculate lambda&c:\n', self.lambda_c)
        print('finish calculate lambda&c')
    
    def __calculate_cp(self):
        '''
        计算压力系数
        '''
        self.p = np.zeros((2,self.number//2-self.del_points))
        self.p[0] = np.array([1-(self.lambda_c[i,0]/self.v)**2 for i in range(self.del_points,self.number//2)])
        self.p[1] = np.array([1-(self.lambda_c[i,0]/self.v)**2 for i in range(self.number//2, self.number-self.del_points)])

        # print('finish calculate cp:\n', self.cp)
        print('finish calculate cp')

        return self.p
    
    def __calculate_cl_cm(self):
        '''
        计算cl\n
        计算力矩系数cm
        '''
        gamma = sum([self.lambda_c[i,0]*self.length[i][0] for i in range(self.number)]) # 环量
        self.cl = 2*gamma/self.v
        print('finish calculate cl:', self.cl)

        self.cm = sum([self.lambda_c[i,0]*self.sep_loc[i,0]*self.length[i][0] for i in range(self.number)]) #力矩系数
        self.cm *= -2/self.v
        print('finish calculate cm:', self.cm)

        return self.cl, self.cm
    
    def calculate_cl2(self):
        '''
        积分法计算cl,cp
        '''
        for i in range(self.number//2):   #注意cp的顺序
            self.cl2[0,0] -= self.p[0,i]*self.length[self.number//2-i-1][0]*self.norm[self.number//2-i-1][0]
            self.cl2[0,0] -= self.p[1,i]*self.length[self.number//2+i][0]*self.norm[self.number//2+i][0]
            self.cl2[0,1] -= self.p[0,i]*self.length[self.number//2-i-1][0]*self.norm[self.number//2-i-1][1]
            self.cl2[0,1] -= self.p[1,i]*self.length[self.number//2+i][0]*self.norm[self.number//2+i][1]
        print('积分法 cl:',self.cl2[0,1],'cd:',self.cl2[0,0])
        
    def render_cp(self) -> None:
        '''
        显示Cp图
        '''
        fig, ax = plt.subplots()
        ax.set(xlabel='x', ylabel='Cp',
            title='Cp (alpha='+str(self.alpha)+'°)\n'+self.name)

        x1= self.ctrl_loc[self.del_points:self.number//2,0].flatten()
        x2= self.ctrl_loc[self.number//2:self.number-self.del_points,0].flatten()
        y1 = self.p[0].flatten()
        y2 = self.p[1].flatten()

        # print(x1)
        # print(y1)
        ax.scatter(x1,y1, s=5, c= 'tomato') #上表面
        ax.scatter(x2,y2, s=5, c= 'cornflowerblue') #下表面
        # ax.plot(self.ctrl_loc[:,0].flatten(),self.ctrl_loc[:,1].flatten(), 'black', alpha=0.5) #显示翼型
        ax.plot(x1,y1, 'tomato')
        ax.plot(x2,y2, 'cornflowerblue')
        ax.set_xlim(-0.01,1.01)
        ax.set_ylim(max(max(y1),max(y2))+0.05,min(min(y1),min(y2))-0.05)
        ax.grid()
        plt.show(block=True)
        plt.close()
    
    def thin_airfoil_theory(self):
        '''
        用于薄翼理论
        '''
        c = 1
        x = lambda x : c/2*(1-math.cos(x))
        f = self.camber_func
        rad = np.radians(self.alpha)

        a0 = rad - 1/math.pi*quad(lambda a: f(x(a)), 0, math.pi)[0]
        a1 = 2/math.pi*quad(lambda a: f(x(a))*math.cos(a), 0, math.pi)[0]
        a2 = 2/math.pi*quad(lambda a: f(x(a))*math.cos(2*a), 0, math.pi)[0]
        self.cl3 = math.pi*(2*a0+a1)
        self.cm3 = -(self.cl3/4+math.pi*(a1-a2)/4)

        print('薄翼  cl:', self.cl3, 'cm:', self.cm3)
        return self.cl3, self.cm3

    @time_counter 
    def render_streamline(self,number_x: int = 100,number_y: int = 100) -> None:
        '''
        显示流线图\n
        number_x, number_y为计算点数
        '''
        self.ux = np.zeros((number_y,number_x))
        self.uy = np.zeros((number_y,number_x))
        x,y = np.meshgrid(np.linspace(-0.2,1.2,number_x),np.linspace(-0.5,0.5,number_y)) #计算点
        for i in range(number_y):
            for j in range(number_x):
                loc = np.array([x[i,j],y[i,j]])
                for k in range(self.number): 
                    loc_ctrl = self.ctrl_loc[k]
                    loc_r = loc_ctrl-loc
                    r = np.linalg.norm(loc_r)
                    self.ux[i,j] += -self.lambda_c[k,0]*self.length[k][0]*loc_r[1]/(2*math.pi*r**2)
                    self.uy[i,j] += self.lambda_c[k,0]*self.length[k][0]*loc_r[0]/(2*math.pi*r**2)
                self.ux[i,j] += self.v_x
                self.uy[i,j] += self.v_y

        map = (self.ux**2+self.uy**2)**0.5  #速度映射-10~10
        mpa_min = np.min(map)
        mpa_max = np.max(map)
        map = ((map-mpa_min)/(mpa_max-mpa_min)-0.5)*10

        fig, ax = plt.subplots()
        ax.set(xlabel='x', ylabel='y',
            title='Streamline (alpha='+str(self.alpha)+'°)\n'+self.name)
        ax.fill(self.ctrl_loc[:,0],self.ctrl_loc[:,1],'teal', zorder= 3)

        ax.contourf(x,y,map, cmap='coolwarm',alpha= 0.5,zorder= 0)
        strm = ax.streamplot(x,y,self.ux,self.uy, density=2,color= map,cmap='autumn', linewidth= 0.5, zorder= 1)
        # fig.colorbar(strm.lines, zorder= 3)
        ax.axis('equal')
        print('finish render streamline')
        # plt.clim(-1,1)
        plt.show()
        plt.close()

    @time_counter
    def render_pots(self, number_x: int = 50, number_y: int = 10, 
                    frame: int = 100, step: float = 0.01, 
                    is_save: bool = True, is_show: bool = False,
                    x_lim: any = [-1,0], y_lim: any = [-0.5,0.5]) -> None:
        '''
        显示流点图\n
        number_x, number_y为计算点数, \n
        frame为帧数, step为步长, \n
        is_save为是否保存gif\n
        is_show为是否显示动态图\n
        x_lim, y_lim为生成点的起始位置
        '''
        if is_save:
                index = 0
                if not os.path.exists('./render_result'):
                    os.mkdir('./render_result')
                while True:
                    if os.path.exists('./render_result/'+str(index)):
                        index += 1
                    else:
                        os.mkdir('./render_result/'+str(index))
                        break

        ux = np.zeros((number_y,number_x))
        uy = np.zeros((number_y,number_x))
        count = 0
        fig, ax = plt.subplots()
        c = np.array([i%10 for i in range(number_x)]*number_y).reshape(number_y,number_x) #颜色映射

        for _ in range(frame):
            if count == 0:
                alpha_rad = np.radians(self.alpha) #旋转矩阵
                rotation_matrix = np.array([[np.cos(alpha_rad), -np.sin(alpha_rad)],
                                            [np.sin(alpha_rad), np.cos(alpha_rad)]])
                x_loc, y_loc = np.meshgrid(np.linspace(x_lim[0], x_lim[1], number_x), 
                                           np.linspace(y_lim[0], y_lim[1], number_y))
                rotated_meshgrid = np.dot(rotation_matrix, np.array([x_loc.flatten(), y_loc.flatten()]))
                x_loc = rotated_meshgrid[0].reshape(number_y, number_x)
                y_loc = rotated_meshgrid[1].reshape(number_y, number_x)
            else:
                for i in range(number_y):
                    for j in range(number_x):
                        loc = np.array([x_loc[i,j],y_loc[i,j]])
                        for k in range(self.number): 
                            loc_ctrl = self.ctrl_loc[k]
                            loc_r = loc_ctrl-loc
                            r = np.linalg.norm(loc_r)
                            ux[i,j] += -self.lambda_c[k,0]*self.length[k][0]*loc_r[1]/(2*math.pi*r**2)
                            uy[i,j] += self.lambda_c[k,0]*self.length[k][0]*loc_r[0]/(2*math.pi*r**2)
                        ux[i,j] += self.v_x
                        uy[i,j] += self.v_y
                x_loc += ux*step
                y_loc += uy*step
                ux = np.zeros((number_y,number_x))
                uy = np.zeros((number_y,number_x))
            if is_show:
                plt.ion()
            plt.cla()
            ax.set(xlabel='x', ylabel='y',
                title='Streamline (alpha='+str(self.alpha)+'°)\n'+self.name+' frame:'+str(count))
            ax.fill(self.ctrl_loc[:,0],self.ctrl_loc[:,1],'teal',zorder= 5)
            ax.scatter(x_loc,y_loc,s=5,c=c,marker='o',cmap='rainbow',zorder= 0)
            ax.axis('equal')
            ax.set_xlim(-0.2,1.2)
            ax.set_ylim(-0.5,0.5)
            ax.grid()
            count += 1
            if is_save:
                plt.savefig('./render_result/'+str(index)+'/'+str(count)+'.png')
            if not is_show:
                plt.close

        if is_save: #生成gif
            buff = []
            print('gif loc:', './render_result/'+str(index)+'/result.gif')
            for i in range(1,frame+1):
                try:
                    buff.append(imageio.imread('./render_result/'+str(index)+'/'+str(i)+'.png'))
                    os.remove('./render_result/'+str(index)+'/'+str(i)+'.png')
                except:
                    print('warning: no such file:','./render_result/'+str(index)+'/'+str(i)+'.png')
            imageio.mimsave('./render_result/'+str(index)+'/result.gif', buff, 'GIF', duration=0.1)
            print('finish save gif:','./render_result/'+str(index)+'/result.gif')
        
        plt.close()
        print('finish render pots')

if __name__ == '__main__':
    pass
    # na = Naca5(naca= '23012',number= 100)
    # na = Naca00(naca= '0021',number= 200)
    # na.render()

    # t = Calculater(v=1, alpha=5, sep_loc= na.points, name = na.name)
    # t.render_cp() #显示Cp图
    # t.render_streamline() #显示流线图
    # t.render_pots(number_x=50,number_y=30,x_lim=[-2,0],frame=100,is_show= False) #显示流点图

    # 下面用于计算cl-alpha曲线 °
    # x = []
    # y = []
    # for i in range(-5,11):
    #     t = Calculater(v=1, alpha=i, sep_loc= na.points)
    #     x.append(i)
    #     y.append(t.cl)
    # fig, ax = plt.subplots()
    # ax.set(xlabel='Alpha', ylabel='Cl',
    #         title='Cl-Alpha(°)')
    # ax.plot(x,y)
    # ax.grid()
    # print(np.polyfit(x,y,1))
    # plt.show()

    # 下面用于计算cl-alpha曲线 rad
    # x = []
    # y = []
    # cl2 = [] #压力积分法计算cl
    # cm = [] #cm
    # for i in range(-5,11):
    #     t = Calculater(v=1, alpha=i, sep_loc= na.points)
    #     x.append(i/180*math.pi)
    #     y.append(t.cl)
    #     cl2.append(-t.cl2[0,1])
    #     cm.append(t.cm)
    # fig, ax = plt.subplots()
    # ax.set(xlabel='Alpha', ylabel='Cl',
    #         title='Cl-Alpha(rad)')
    # ax.plot(x,y)
    # ax.plot(x,cl2)
    # ax.plot(x,cm)
    # ax.grid()
    # ax.legend(['normal','integral'])
    # print(np.polyfit(x,y,1))
    # plt.show()