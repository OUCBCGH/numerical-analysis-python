# # -*- coding: UTF-8 -*-
# import sys
# sys.path.append(r'/home/ld/numerical-analysis-python/NumericalCalculationMethod')
"""
@file_name: util_font.py
@time: 2023-02-14
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
# import matplotlib.pyplot as plt
# import matplotlib as mpl

# # 设置数学模式下的字体格式和中文显示
# #  "font.weight": "bold", "axes.labelweight": "bold"
# rc = {"font.family": "serif", "mathtext.fontset": "cm"}
# plt.rcParams.update(rc)
# mpl.rcParams["font.family"] = "FangSong"  # 中文显示
# plt.rcParams["axes.unicode_minus"] = False  # 解决坐标轴负数的负号显示问题


import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager

# 指定中文字体路径（确保路径正确）
font_path = "/usr/share/fonts/winfonts/simfang.ttf"
zh_font = font_manager.FontProperties(fname=font_path)

# 设置字体和数学字体
mpl.rcParams["font.family"] = zh_font.get_name()
mpl.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "cm"