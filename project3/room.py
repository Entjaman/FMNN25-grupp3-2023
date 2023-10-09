
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
        
        self.u_new = None
        self.u_current = None
        
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
    
    def update_dirichlt_condition(self,wall,start,new_value):
        s = int(start/self.dx)
        for i in range(len(new_value)):
            if(wall=='right'):
                index = (i+s+1)*self.x_size-1
                self.b[index][0] = self.b[index][0]+self.right[i+s]/(self.dx**2)
                self.right[i+s] = new_value[i]
            elif(wall=='left'):
                index = (i+s)*self.x_size
                self.b[index][0] = self.b[index][0]+self.left[i+s]/(self.dx**2)
                self.left[i+s] = new_value[i]
        self.add_dirichlt_condition(new_value, wall, start)
        return None
    
    #wall can be a string that is top, bottom,left,or right
    def add_dirichlt_condition(self,value,wall,start):
        start = int(start/self.dx)
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
    
    def update_neuman_condition(self,wall,start,new_value):
        
        return None
            
    def add_neuman_condition(self,u,value,wall):
        for i in range(len(value)):
            if(wall=='right'):
                index = (i+1)*self.x_size-1
            elif(wall=='left'):
                index = i*self.x_size
            
            self.b[index][0] = self.b[index][0]-(value[i]-u[i])/(self.dx**2)
            self.a[index][index] = self.a[index][index]/(self.dx**2)
        
        return None

    
    #a neumann wall has to be a full wall
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
            #update the u_new to contain the previous boundary values
            self.u_new = np.reshape(self.u_new,(self.y_size,self.x_size))
            self.u_new = np.c_[self.u_new,self.right]
            self.x_size = new_x_size
            #I'll just recreate the A matrix,seems easier. 
            self.v_size = int(self.x_size*self.y_size)
            self.u_new = np.reshape(self.u_new,((self.v_size,1)))
            a_r = np.zeros(self.v_size)
            a_c = np.zeros(self.v_size)
            a_r[0] = -4
            a_c[0] = -4
            a_r[1]= a_r[self.x_size] =1
            a_c[1] = a_c[self.x_size] = 1
            a =  sci.toeplitz(a_c,a_r)
            #removes the a values that should come from boundary walls instead & changes the -4 to -3 for the neuman condition boundary
            for i in range(self.y_size-1):
                a[(1+i)*(self.x_size)-1][(1+i)*(self.x_size)]=0
                a[(1+i)*(self.x_size)][(1+i)*(self.x_size)-1]=0
            for i in range(self.y_size):
                a[(1+i)*(self.x_size)-1][(1+i)*(self.x_size)-1]= -3
            self. a = 1/(self.dx**2)*a
            self.b = np.reshape(self.b_add_column,(self.x_size*self.y_size))
            
        elif(wall=='left'):
            new_x_size = self.x_size +1
            for i in range(len(self.left)):
                index = (i)*self.x_size
                self.b[index][0] = self.b[index][0]+self.left[i]/(self.dx**2)
            b_reshape = np.reshape(self.b,(self.y_size,self.x_size))
            self.b_add_column = np.c_[np.zeros((self.x_size,1)),b_reshape]
            self.b_add_column[0][0] = -self.bottom[self.x_size-1]/(self.dx**2)
            self.b_add_column[self.y_size-1][0] = -self.top[self.x_size-1]/(self.dx**2)
            self.bottom = np.append(self.bottom,self.bottom[self.x_size-1])
            self.top = np.append(self.top,self.top[self.x_size-1])
            #update the u_new to contain the previous boundary values
            self.u_new = np.reshape(self.u_new,(self.y_size,self.x_size))
            self.u_new = np.c_[self.left,self.u_new]
            self.x_size = new_x_size
            self.v_size = int(self.x_size*self.y_size)
            self.u_new = np.reshape(self.u_new,((self.v_size,1)))
            a_r = np.zeros(self.v_size)
            a_c = np.zeros(self.v_size)
            a_r[0] = -4
            a_c[0] = -4
            a_r[1]= a_r[self.x_size] =1
            a_c[1] = a_c[self.x_size] = 1
            a =  sci.toeplitz(a_c,a_r)
            #removes the a values that should come from boundary walls instead & changes the -4 to -3 for the neuman condition boundary
            for i in range(self.y_size-1):
                a[(1+i)*(self.x_size)-1][(1+i)*(self.x_size)]=0
                a[(1+i)*(self.x_size)][(1+i)*(self.x_size)-1]=0
            for i in range(self.y_size):
                a[(i)*(self.x_size)][(i)*(self.x_size)] = -3
            self. a = 1/(self.dx**2)*a
            self.b = np.reshape(self.b_add_column,(self.x_size*self.y_size))

            
    
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
        
    
    #can add the realaxation demand here. 
    def update(self):
        self.u_current = self.u_new
        self.u_new = None
    
    def solve(self):
        self.u_new = sci.solve(self.a,self.b)
        return self.u_new
    
    def get_boundary_values(self,k,length,start,wall):
        start = int(start/self.dx)
        length = int(length/(self.dx))
        u_bound = np.array([])
        if(k=='new'):
            u=self.u_new
        elif(k=='old'):
            u = self.u_current
        
        for i in range(length):
            if(wall=='right'):
                index = (i+start+1)*self.x_size-1
            elif(wall=='left'):
                index = (i+start)*self.x_size
            u_bound = np.append(u_bound,u[index])
        
        return u_bound

#def main():

#initialization
room_two = Room(np.array([1,2]),1/3)
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
room_three.add_neumann_wall('left')
room_one.update()
room_two.update()
room_three.update()
#start the iteration

u_one_old = room_one.get_boundary_values('old',1,0,'right')
u_three_old = room_three.get_boundary_values('old',1,0,'left')
room_two.update_dirichlt_condition('right', 1, u_three_old)
room_two.update_dirichlt_condition('left', 0, u_one_old)
room_two.solve()

u_two_new_left = room_two.get_boundary_values('new', 1, 0, 'left')
u_two_new_right = room_two.get_boundary_values('new', 1, 1, 'right')
#room_one.

#if __name__ == "__main__":
#    main()
    
    