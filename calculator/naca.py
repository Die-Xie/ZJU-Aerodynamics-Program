import numpy as np
import matplotlib.pyplot as plt
import math
from .wrapper import time_counter

class Naca_root():
    '''
    翼型的基类 
    '''
    def __init__(self, naca: str, number: int):
        self.name = naca
        self.number = number
        self._points = None
        self.up_points = None
        self.down_points = None
        self.camber_points = None
        self.yc = None #中弧线函数
        self.dyc = None #中弧线导数函数

    def render(self) -> None:
        '''
        显示翼型图
        '''
        fig, ax = plt.subplots()
        ax.set(xlabel='x', ylabel='y',
            title='NACA '+self.name)
        
        ax.plot(self.up_points[:,0],self.up_points[:,1], 'tomato')
        ax.plot(self.down_points[:,0],self.down_points[:,1], 'cornflowerblue')
        ax.plot(self.camber_points[:,0],self.camber_points[:,1], 'black')
        ax.grid()
        ax.axis('equal')
        plt.show()
        plt.close()

    @property
    def points(self):
        '''
        获取翼型点
        '''
        return self._points
    
    @points.setter
    def points(self,*args,**kwargs):
        raise ValueError('不允许修改points')
    
    @property
    def camber_func(self):
        '''
        获取翼型中弧线上 y=f(x) 函数
        '''
        assert self.yc is not None, '请先初始化翼型'
        return self.yc 
    
    @property
    def camber_func_derivative(self):
        '''
        获取翼型中弧线上 y=f(x) 的导数函数
        '''
        assert self.dyc is not None, '请先初始化翼型'
        return self.dyc

class Naca0(Naca_root):
    '''
    对称翼型 
    '''
    def __init__(self, naca: str = '0012', number: int = 200):
        if len(naca) != 4:
            raise ValueError('输入翼型错误: 仅支持4位数翼型')
        elif int(naca[0]) != 0:
            raise ValueError('输入翼型错误: 仅支持对称翼型,有弯度翼型请使用Naca4()')
        super().__init__(naca, number)
        self.t = int(naca[2:4])/100 #最大厚度(%)

        self.k1 = 0
        self.m = 0
        self.k2k1 = 0 #reflexed 翼型
        self.camber_points = np.array([[i, 0] for i in np.linspace(0,1,self.number)]).reshape(self.number,2)
        self.__init()
        self.yc = lambda x: 0
        self.dyc = lambda x: 0

    @time_counter
    def __init(self):
        y = lambda x: self.t/0.2*(0.2969*x**0.5-0.126*x-0.3516*x**2+0.2843*x**3-0.1036*x**4)
        self.up_points = np.array([[i,y(i)] for i in np.linspace(0,1,self.number//2)]).reshape(self.number//2,2)
        self.down_points = np.array([[i,-y(i)] for i in np.linspace(0,1,self.number//2)]).reshape(self.number//2,2)
        # self._points = np.vstack((self.down_points[::-1], self.up_points))
        self._points = np.vstack((np.delete(self.up_points.copy(), -1, axis=0)[::-1],
                                    np.delete(self.down_points.copy(), 0, axis=0)))

class Naca4(Naca_root):
    '''
    获取NACA翼型数据,仅支持4位数翼型
    '''
    def __init__(self, naca: str, number: int):
        if len(naca) != 4:
            raise ValueError('输入翼型错误: 仅支持4位数翼型')
        elif int(naca[0]) == 0:
            raise ValueError('输入翼型错误: 仅支持有弯度翼型,对称翼型请使用Naca00()')

        super().__init__(naca, number)
        self.m = int(naca[0])/100 #最大弯度
        self.p = int(naca[1])/10 #最大弯度位置(小数)
        self.t = int(naca[2:])/100 #机翼最大厚度占弦长
        self.__init()

    @time_counter
    def __init(self):
        # 中弧线
        self.yc = lambda x: self.m/self.p**2*(2*self.p*x-x**2) if x<=self.p\
                            else self.m/(1-self.p)**2*((1-2*self.p)+2*self.p*x-x**2)
        self.dyc = lambda x: self.m/self.p**2*(2*self.p-2*x) if x<=self.p\
                            else self.m/(1-self.p)**2*(2*self.p-2*x)

        # 上下表面
        self.yt = lambda x: 5*self.t*(0.2969*x**0.5-0.126*x-0.3516*x**2+0.2843*x**3-0.1036*x**4)
        self.x_up = lambda x: x-self.yt(x)*math.sin(math.atan(self.dyc(x)))
        self.x_down = lambda x: x+self.yt(x)*math.sin(math.atan(self.dyc(x)))
        self.y_up = lambda x: self.yc(x)+self.yt(x)*math.cos(math.atan(self.dyc(x)))
        self.y_down = lambda x: self.yc(x)-self.yt(x)*math.cos(math.atan(self.dyc(x)))

        # 获取翼型上的点
        # 上表面需为逆序 注意去除[0,0]点
        number = self.number+1
        # x = np.linspace(0, 1, number)
        temp_x = np.linspace(math.pi,0, number) #加密机翼前后缘
        x = np.array([0.5*math.cos(i)+0.5 for i in temp_x])
        x_up_loc = np.array([self.x_up(i) for i in x[::-1]]).reshape(number,1)
        y_up_loc = np.array([self.y_up(i) for i in x[::-1]]).reshape(number,1)
        
        x_down_loc = np.array([self.x_down(i) for i in x]).reshape(number,1)
        y_down_loc = np.array([self.y_down(i) for i in x]).reshape(number,1)

        self.up_points = np.hstack((x_up_loc, y_up_loc))

        self.down_points = np.hstack((x_down_loc, y_down_loc))
        self._points = np.vstack((np.delete(self.up_points.copy(), -1, axis=0),
                                    np.delete(self.down_points.copy(), 0, axis=0)))
        
        # 中弧线
        self.camber_points = np.array([[i, self.yc(i)] for i in x]).reshape(number,2)
        print('NaCa finish init:',self.name)

class Naca5(Naca_root):
    '''
    获取NACA翼型数据,仅支持5位数翼型
    默认翼型为NACA 23015\n
    目前支持: \n\t23018 23012 23015 23021 23025 23027 23030
            \n\t23112 23115 23118 23121 23124 23127 23130 等几乎所有5位翼型
    '''
    def __init__(self, naca: str = '23015', number: int = 200):
        if len(naca) != 5:
            raise ValueError('输入翼型错误: 仅支持5位数翼型')
        super().__init__(naca, number)
        self.x = int(naca[0])
        self.y = int(naca[1])
        self.w = int(naca[2])
        self.zz = int(naca[3:])

        self.cl = self.x*3/20 #设计升力系数
        self.p = self.y/20 #最大弯度位置(小数)
        self.t = self.zz/100 #最大厚度(%)

        self.k1 = 0
        self.m = 0
        self.k2k1 = 0 #reflexed 翼型
        
        self.__init()

    @time_counter
    def __init(self):
        # normal 翼型
        dic_nor = {0.15:(15.957, 0.2025), 
                    0.05:(361.4, 0.058), 
                    0.10:(51.642, 0.126), 
                    0.20:(6.643, 0.290), 
                    0.25:(3.230, 0.391), 
                    0.30:(1.861, 0.461),
                    0.35:(1.165, 0.506)}
        dic_ref = {0.10:(51.990, 0.1300, 0.000764),
                    0.15:(15.793, 0.2170, 0.00677),
                    0.20:(6.520, 0.3180, 0.0303),
                    0.25:(3.191, 0.4410, 0.1355)}
        if self.w == 0:
            if self.p in dic_nor.keys():
                self.k1, self.m = dic_nor[self.p]
            else:
                raise ValueError('暂时不支持:'+str(self.p))
        
            # 中弧线
            self.yc = lambda x: self.k1/6*(x**3-3*self.m*x**2+self.m**2*(3-self.m)*x) if x<=self.m\
                                else self.k1*self.m**3/6*(1-x)
            self.dyc = lambda x: self.k1/6*(3*x**2-6*self.m*x+self.m**2*(3-self.m)) if x<=self.m\
                                else -self.k1*self.m**3/6
        
        # reflexed 翼型
        elif self.w == 1:
            if self.p in dic_ref.keys():
                self.k1, self.m, self.k2k1 = dic_ref[self.p]
            else:
                raise ValueError('暂时不支持:'+str(self.p))
            
            # 中弧线
            self.yc = lambda x: self.k1/6*((x-self.m)**3-self.k2k1*(1-self.m)**3*x-self.m**3*x+self.m**3) if x<=self.m\
                                else self.k1/6*(self.k2k1*(x-self.m)**3-self.k2k1*(1-self.m)**3*x-self.m**3*x+self.m**3)
            self.dyc = lambda x: self.k1/6*(3*(x-self.m)**2-self.k2k1*(1-self.m)**3-self.m**3) if x<=self.m\
                                else self.k1/6*(3*self.k2k1*(x-self.m)**2-self.k2k1*(1-self.m)**3-self.m**3)
        else:
            raise ValueError('输入翼型错误:w ='+str(self.w))

        # 上下表面
        self.yt = lambda x: 5*self.t*(0.2969*x**0.5-0.126*x-0.3516*x**2+0.2843*x**3-0.1036*x**4)
        self.x_up = lambda x: x-self.yt(x)*math.sin(math.atan(self.dyc(x)))
        self.x_down = lambda x: x+self.yt(x)*math.sin(math.atan(self.dyc(x)))
        self.y_up = lambda x: self.yc(x)+self.yt(x)*math.cos(math.atan(self.dyc(x)))
        self.y_down = lambda x: self.yc(x)-self.yt(x)*math.cos(math.atan(self.dyc(x)))

        # 获取翼型上的点
        # 上表面需为逆序 注意去除[0,0]点
        number = self.number+1
        # x = np.linspace(0, 1, number)
        temp_x = np.linspace(math.pi,0, number) #加密机翼前后缘
        x = np.array([0.5*math.cos(i)+0.5 for i in temp_x])
        # print(x)

        x_up_loc = np.array([self.x_up(i) for i in x[::-1]]).reshape(number,1)
        y_up_loc = np.array([self.y_up(i) for i in x[::-1]]).reshape(number,1)
        
        x_down_loc = np.array([self.x_down(i) for i in x]).reshape(number,1)
        y_down_loc = np.array([self.y_down(i) for i in x]).reshape(number,1)

        self.up_points = np.hstack((x_up_loc, y_up_loc))

        self.down_points = np.hstack((x_down_loc, y_down_loc))
        self._points = np.vstack((np.delete(self.up_points.copy(), -1, axis=0),
                                    np.delete(self.down_points.copy(), 0, axis=0)))

        # 中弧线
        self.camber_points = np.array([[i, self.yc(i)] for i in x]).reshape(number,2)
        print('NaCa finish init:',self.name)
