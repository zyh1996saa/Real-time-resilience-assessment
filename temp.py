# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 12:22:51 2022

@author: NINGMEI
"""

xxcase=copy.deepcopy(ori_case118)
xxcase['branch'] = np.delete(xxcase['branch'],[7,2,10,5,1,13,3,4],axis=0)

xxcase['bus'] = np.delete(xxcase['bus'],[7,2,10,5,1,13,3,4],axis=0)
xxres = runpf.runpf(xxcase)

xxcase2=copy.deepcopy(ori_case118)
xxcase2['branch'] = np.delete(xxcase2['branch'],[2,5],axis=0)

xxcase2['bus'][5,8] = 0
xxcase2['bus'][2,10] = 2
xxcase2['bus'][4,10] = 2
xxcase2['bus'][5,10] = 2
xxcase2['bus'][2,6] = 2
xxcase2['bus'][4,6] = 2
xxcase2['bus'][5,6] = 2
xxcase2['bus'][5,1] = 3
xxres2 = runpf.runpf(xxcase2)


yyy1 = copy.deepcopy(yyy)
