'''
Created on Oct 11, 2012

@author: andy
'''

class Test(object):
    
    def __init__(self):
        self.state = 1
        
    def test(self, s = None):
        s = self.state if s is None else s
        print(s)
