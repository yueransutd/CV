# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 14:32:20 2017

@author: jessicasutd
"""

import xlwt
workbook=xlwt.Workbook(encoding="utf-8")
sheet1=workbook.add_sheet("Sheet1")

sheet1.write(0,0,"hhhh")
sheet1.col(0).width=7000
workbook.save("pythonxl.xls")
print("workbook created")