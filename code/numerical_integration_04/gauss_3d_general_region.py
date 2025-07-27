# -*- coding: UTF-8 -*-
import sys
sys.path.append(r'/home/ld/numerical-analysis-python/NumericalCalculationMethod')
"""
@file_name: gauss_3d_general_region.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
import math


class Gauss3DGeneralIntegration:
    """
    高斯勒让德计算一般区域的三重积分
    """

    def __init__(self, int_fun, a, b, c_x, d_x, alpha_xy, beta_xy, zeros_num=np.array([10, 10, 10])):
        self.ax, self.bx = a, b  # x积分区间
        self.ay, self.by = c_x, d_x  # y积分区间，关于x的函数
        self.az, self.bz = alpha_xy, beta_xy  # z积分区间，关于x,y的函数
        self.int_fun = int_fun  # 被积函数
        self.z_p, self.y_n, self.x_m = zeros_num[2], zeros_num[1], zeros_num[0]  # 零点数
        self.int_value = None  # 最终积分值

    def cal_3d_int(self):
        """
        核心算法：一般区域的三重数值积分
        :return:
        """
        A_k_x, zero_points_x = self._cal_Ak_zeros_(self.x_m)  # 获取插值系数和高斯零点
        A_k_y, zero_points_y = self._cal_Ak_zeros_(self.y_n)  # 获取插值系数和高斯零点
        A_k_z, zero_points_z = self._cal_Ak_zeros_(self.z_p)  # 获取插值系数和高斯零点
        h1, h2 = (self.bx - self.ax) / 2, (self.bx + self.ax) / 2  # 用于区间转换为[-1, 1]
        self.int_value = 0.0
        for i in range(self.x_m):  # 针对积分变量x
            int_x = 0.0
            zp_x = h1 * zero_points_x[i] + h2  # 区间转换后的零点
            c1, d1 = self.ay(zp_x), self.by(zp_x)  # 变量y的积分上下限的函数值
            k1, k2 = (d1 - c1) / 2, (d1 + c1) / 2  # 用于区间转换为[-1, 1]
            for j in range(self.y_n):
                zp_y = k1 * zero_points_y[j] + k2  # 区间转换后的零点
                beta_1, alpha_1 = self.bz(zp_x, zp_y), self.az(zp_x, zp_y)  # 零点函数值
                l1, l2 = (beta_1 - alpha_1) / 2, (beta_1 + alpha_1) / 2  # 用于区间转换为[-1, 1]
                zp_z = l1 * zero_points_z + l2  # 区间转换后的零点，一维数组
                f_val = self.int_fun(zp_x, zp_y, zp_z)  # 零点函数值，一维数组
                int_y = np.dot(f_val, A_k_z)  # 积分值累加
                int_x += A_k_y[j] * l1 * int_y
            self.int_value += A_k_x[i] * k1 * int_x  # 积分值累加
        self.int_value *= h1
        return self.int_value

    def _cal_Ak_zeros_(self, n):
        """
        求解勒让德的高斯零点和Ak系数，n为零点数
        :return:
        """
        t = sympy.Symbol("t")
        # 勒让德多项式构造
        p_n = (t ** 2 - 1) ** n / math.factorial(n) / 2 ** n
        diff_p_n = sympy.diff(p_n, t, n)  # n阶导数
        # 求解多项式的全部零点
        zero_points = np.asarray(sympy.solve(diff_p_n, t), dtype=np.float)
        Ak_poly = sympy.lambdify(t, 2 / (1 - t ** 2) / (sympy.diff(diff_p_n, t, 1)) ** 2)
        A_k = Ak_poly(zero_points)  # 求解Ak系数
        return A_k, zero_points
