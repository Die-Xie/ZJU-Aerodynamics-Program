'''
该文档是一个使用的例子,用于展示如何使用设计的api
'''
# 从package中导入类，可使用如下方法
from calculator import Calculator, Naca0, Naca4, Naca5

import matplotlib.pyplot as plt
import numpy as np
import math

if __name__ == "__main__":

    # 创建一个Naca对象，参数为翼型名称和点数
    # na = Naca0('0012', 100)   #对称翼型
    # # na = Naca4('4410', 100)   #naca4位翼型
    na = Naca5('23021', 100)    #naca5位翼型
    # 将翼型渲染出来
    # na.render()

    # 创建一个计算器对象，参数为速度，攻角，翼型点集，翼型名称，计算方法等
    t = Calculator(v = 1, alpha = -5, sep_loc=na.points, name = na.name, method = 1,camber_func_derivative= na.dyc) #直接使用积分
    # t = Calculator(v = 1, alpha = 0, sep_loc=na.points, name = na.name, method = 0) #用分段积分代替积分（默认分段数50）       

    # 计算并渲染cp分布
    t.render_cp()

    # 计算并渲染速度分布,参数为x,y方向的模拟点数点数
    # t.render_streamline(number_x= 100, number_y= 100)

    # 计算并渲染流线gif，参数为x,y方向的模拟点数点数，帧数，步长，是否保存，是否在运算过程中显示等
    # t.render_pots(number_x= 100, number_y= 30, frame= 100)

    '''
    下面是一个更为详细使用例
    '''
    # 下面为一个实际使用的例子,用于计算cl-alpha曲线（弧度制）
    # 下面用于计算cl-alpha曲线 rad
    # x = []
    # y = []
    # cl2 = [] #压力积分法计算cl
    # cm = [] #cm
    # for i in range(-5,11):
    #     t = Calculator(v=1, alpha=i, sep_loc= na.points)
    #     x.append(i/180*math.pi)
    #     y.append(t.cl)
    #     cl2.append(t.cl2[0,1])
    #     cm.append(t.cm)
    # fig, ax = plt.subplots()
    # ax.set(xlabel='Alpha', ylabel='Cl',
    #         title='Cl-Alpha(rad)')
    # ax.plot(x,y)
    # ax.plot(x,cl2)
    # ax.plot(x,cm)
    # ax.grid()
    # ax.legend(['环量求解','积分法','力矩'])
    # print(np.polyfit(x,y,1)) #拟合
    # plt.show()