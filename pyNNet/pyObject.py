# -*- coding: utf-8 -*-
"""
+--------------------------------------------+
| FileName: pyObject.py                       |
+--------------------------------------------+
| Created on Wed Jul  4 10:37:59 2018        |
+--------------------------------------------+
| @author: Benito Fernandez                  |
+--------------------------------------------+
"""
from enum import Enum

'''---------------------------------------------------------------------------
Direction of signal in Processing Element (derived class)
    - forward:  [  fowardOperatorName,   fowardOperatorName_]
    - backward: [backwardOperatorName, backwardOperatorName_]

    Note: The underscore signifies the gradient of the
          operator,
          function,
          variable (gradient with respect to the specific variable)
'''

class Direction(Enum):
     FWD = 0
     BWD = 1

'''---------------------------------------------------------------------------
Side of Processing Element
    - IN:  [ inputOperatorName,  inputOperatorName_]
    - OUT: [outputOperatorName, outputOperatorName_]
'''

class Side(Enum):
     IN  = 0
     OUT = 1

xDebug = True

# Base pyGNNetCore object

class pyObject(object):

    __uid = 0        # This id will be maintained by the class and
                     # should not be edited by user.
    '''
    - Parameters:
        - _network: Network Id the object belongs to
        - _type:    Type of object (overloaded by derived classes).
        - _name:    String identifier of object
    '''
    def __init__(self, _type=None, _name=None, _creator=None):

#        super(pyObject, self).__init__()
        self._id = id(self)
        self.id = pyObject.__uid
        pyObject.__uid += 1
        self._type = self.__class__.__name__
        __name = "unnamed_"+str(self._type)
        # assign type (it's class - derived, not root)
        if _type is not None:
            self._type =  _type
        # assign name
        if _name is None:
            self._name = __name
        else:
            self._name =  _name
        # assign creator
        self._parent = self._creator = _creator
        # assign ancestry
        if 'self.ancestors' in locals():
            # self.ancestors exists.
            self.ancestors.append(self._parent)
        else:
            self.ancestors = []
        if xDebug and _creator is None: print("Creating a py{}({}) [{}].".format(self._type,self._id,self.id))

    def __repr__(self):
        s = "I am a {Type}[{Id}] object, named '{name}'"
        return s.format(Type = self.__class__.__name__,
                        Id   = self.id,
                        name = self._name)

    def __str__(self):
        return "[{type}]".format(type=self.__class__.__name__)

    def get_uid(self):
        return self.id

    def get_name(self):
        return self._name

    def set_name(self, newName):
        self._name = newName

    def get_objectsCreated(self):
        return self.__uid

    def setParent(self, parent):
        self._parent = parent

    def getParent(self):
        return self._parent

    def getAncestors(self):
        return self.ancestors

    def myClassName(self):
        return self.__class__.__name__

    def printMe(self):
        print(30*'-')
        print(self)
        print('_type   :', self._type,
              '\n_creator:', self._creator,
              '\n_name   :', self._name,
              '\n_id     :', self.id)

    def test():
        po = pyObject()
        po.printMe()

        poNamed = pyObject(_name='poNamed')
        poNamed.printMe()

        poFull = pyObject(_type='[givenType]', _name='poFull')
        poFull.printMe()

'''----------------------------------------------------------------------
Example of pyObject
'''
def printMe(po):
    print(30*'-')
    print(po)
    print('_type   :'  , po._type,
          '\n_name   :', po._name,
          '\n_creator:', po._creator,
          '\n_id     :'  , po._id)

#<------------------------------ main() function for testing if run alone
def testpyObject():
    po = pyObject()
    printMe(po)

    poNamed = pyObject(_name='poNamed')
    printMe(poNamed)

    poFull = pyObject(_type='givenType', _name='poFull')
    poFull.printMe()

if __name__ == "__main__":
#    testpyObject()
    pyObject.test()

# --- END OF FILE ---
