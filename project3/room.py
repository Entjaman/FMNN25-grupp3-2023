
import numpy as np
import scipy.linalg as sci


#Notes and questions. We don't caluclate the u values on the borders, when calculating the size
#of our 2D matrix do we subtract for the i=0,j=0 and i=last, j=last points, which are on 
#the boundaries and not inside the room. 

#i think I have switched the indecies compared to the lecture, with i being y value, and j being x value

class Room():
    def __init__(self, room_size, dx):
        """
        room_size           : vector x[0] - size of room in x dimension x[1] size of room in y dimension
        dx                  : step size in the Cartesian Grid
        c1                  : arbitrary constant between (0,1)
        c2                  : arbitrary constant for the curvature condition between (c1,1)
        
        Implemented as per slide 21 in the course lecture
        """
        self.room_size = room_size
        self.dx = dx
        self.x_size = int(room_size[0]/dx)
        self.y_size = int(room_size[1]/dx)
        self.v_size = int(self.x_size*self.y_size) #If I should remove the boundarys could just subtract here
        self.b = np.zeros((self.v_size,1))
        
        #Create the a matrix. 
        
        a_r = np.zeros(self.v_size)
        a_c = np.zeros(self.v_size)
        a_r[0] = -4
        a_c[0] = -4
        a_r[1]= a_r[self.x_size] =1
        a_c[1] = a_c[self.x_size] = 1
        a =  sci.toeplitz(a_c,a_r)
        #removes the a values that should come from boundary walls instead. 
        for i in range(self.y_size-1):
            a[(1+i)*(self.x_size)-1][(1+i)*(self.x_size)]=0
            a[(1+i)*(self.x_size)][(1+i)*(self.x_size)-1]=0
        self. a = 1/(dx**2)*a
#        for i in range(self.v_size):
#            self.a[i,i]=-4
#            if()
#            self.a[i,i+x_size] = 1
#                

    #wall can be a string that is top, bottom,left,or right
    def add_dirichlt_condition(self,value,wall,start):
        for i in range(len(value)):
            if(wall=='bottom'):
                index = start+i
                assert(index<self.x_size)
            elif(wall=='top'):
                index = self.x_size*(self.y_size-1)+start+i
                assert(index<self.v_size)
            elif(wall=='left'):
                index = (i+start)*self.x_size
                assert(index<=self.x_size*(self.y_size-1))
            elif(wall=='right'):
                index = (i+start+1)*self.x_size-1
                assert(index<self.v_size)
            self.b[index][0] = self.b[index][0]-value[i]/(self.dx**2)
    
    def add_neumann_wall(self,wall):
        if(wall=='right'):
            new_x_size = self.x_size +1
            for i in range(len(self.right)):
                index = (i+1)*self.x_size-1
                self.b[index][0] = self.b[index][0]+self.right[i]/(self.dx**2)
            b_reshape = np.reshape(self.b,(self.y_size,self.x_size))
            self.b_add_column = np.c_[b_reshape, np.zeros((self.x_size,1))]
            self.b_add_column[0][new_x_size-1] = -self.bottom[self.x_size-1]/(self.dx**2)
            self.b_add_column[self.y_size-1][new_x_size-1] = -self.top[self.x_size-1]/(self.dx**2)
            self.bottom = np.append(self.bottom,self.bottom[self.x_size-1])
            self.top = np.append(self.top,self.top[self.x_size-1])
            self.x_size = new_x_size
            #I'll just recreate the A matrix,seems easier. 
            self.v_size = int(self.x_size*self.y_size)
            a_r = np.zeros(self.v_size)
            a_c = np.zeros(self.v_size)
            a_r[0] = -4
            a_c[0] = -4
            a_r[1]= a_r[self.x_size] =1
            a_c[1] = a_c[self.x_size] = 1
            a =  sci.toeplitz(a_c,a_r)
            #removes the a values that should come from boundary walls instead. 
            for i in range(self.y_size-1):
                a[(1+i)*(self.x_size)-1][(1+i)*(self.x_size)]=0
                a[(1+i)*(self.x_size)][(1+i)*(self.x_size)-1]=0
            self. a = 1/(self.dx**2)*a
            self.b = np.reshape(self.b_add_column,(self.x_size*self.y_size))
        elif(wall=='left'):
            new_x_size = self.x_size +1
        return None
    
    def add_neuman_condition(self,u,value,wall,start):
        for i in range(len(value)):
            if(wall=='right'):
                x = 0
                
        
        return None
        
    
    #creates our initial walls
    def create_walls(self, wall_bottom, wall_top, wall_right, wall_left):
        
        self.bottom = np.full((self.x_size),wall_bottom)
        self.top = np.full((self.x_size),wall_top)
        self.right = np.full((self.y_size),wall_right)
        self.left = np.full((self.y_size),wall_left)
        self.add_dirichlt_condition(self.bottom, 'bottom', 0)
        self.add_dirichlt_condition(self.top, 'top', 0)
        self.add_dirichlt_condition(self.right, 'right', 0)
        self.add_dirichlt_condition(self.left, 'left', 0)
        

    
    
    def solve(self):
        return sci.solve(self.a,self.b)


#def main():
   
room_two = Room(np.array([1,2]),1/20)
room_two.create_walls(5,40,15,15)
u_two = room_two.solve()

room_one = Room(np.array([1,1]),1/3)
room_three = Room(np.array([1,1]),1/3)
room_one.create_walls(15,15,15,40)
u_one = room_one.solve()

room_three.create_walls(15,15,40,15)
u_three = room_three.solve()

#adds the changing boundary due to the neumann condition.
u_one = np.reshape(u_one,(room_one.y_size,room_one.x_size))
u_one = np.c_[u_one,np.full((room_one.x_size,1),15)]
u_three = np.reshape(u_three,(room_three.y_size,room_three.x_size))
u_three = np.c_[np.full((room_three.x_size,1),15),u_three]

room_one.add_neumann_wall('right')


#if __name__ == "__main__":
#    main()
    
    