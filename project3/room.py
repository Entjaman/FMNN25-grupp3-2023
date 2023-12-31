
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
        dirichlt_value = np.array([])
        for i in range(len(new_value)):
            if(wall=='right'):
                index = (i+s+1)*self.x_size-1
                self.b[index][0] = self.b[index][0]-self.right[i+s]
                self.right[i+s] = -new_value[i]/(self.dx**2)
            elif(wall=='left'):
                index = (i+s)*self.x_size
                self.b[index][0] = self.b[index][0]-self.left[i+s]
                self.left[i+s] = -new_value[i]/(self.dx**2)
            dirichlt_value = np.append(dirichlt_value,-new_value[i]/(self.dx**2))
        self.add_dirichlt_condition(dirichlt_value, wall, start)
    
    
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
            self.b[index][0] = self.b[index][0]+value[i]
    
    
    def update_neuman_condition(self,wall,new_value):
        for i in range(len(new_value)):
            if(wall=='right'):
                index = (i+1)*self.x_size-1
                self.b[index][0] = self.b[index][0]-self.right[i]
            elif(wall=='left'):
                index = (i)*self.x_size
                self.b[index][0] = self.b[index][0]-self.left[i]
        self.add_neuman_condition(new_value,wall)

            
    def add_neuman_condition(self,value,wall):
        wall_update = np.array([])
        for i in range(len(value)):
            if(wall=='right'):
                index = (i+1)*self.x_size-1
            elif(wall=='left'):
                index = i*self.x_size
            neuman_value = -(value[i]-self.u_current[index])/(self.dx**2)
            wall_update = np.append(wall_update,neuman_value)
            self.b[index][0] = self.b[index][0]+neuman_value
        
        if(wall=='right'):
            self.right = wall_update
        elif(wall=='left'):
            self.left = wall_update
        return None

    
    #a neumann wall has to be a full wall
    def add_neumann_wall(self,wall,value):
        if(wall=='right'):
            new_x_size = self.x_size +1
            for i in range(len(self.right)):
                index = (i+1)*self.x_size-1
                self.b[index][0] = self.b[index][0]-self.right[i]
            b_reshape = np.reshape(self.b,(self.y_size,self.x_size))
            self.b_add_column = np.c_[b_reshape, np.zeros((self.x_size,1))]
            self.b_add_column[0][new_x_size-1] = +self.bottom[self.x_size-1]
            self.b_add_column[self.y_size-1][new_x_size-1] = +self.top[self.x_size-1]
            self.bottom = np.append(self.bottom,self.bottom[self.x_size-1])
            self.top = np.append(self.top,self.top[self.x_size-1])
            #update the u_new to contain the previous boundary values
            self.u_current = np.reshape(self.u_current,(self.y_size,self.x_size))
            self.u_current = np.c_[self.u_current,-self.right*self.dx**2]
            self.x_size = new_x_size
            #I'll just recreate the A matrix,seems easier. 
            self.v_size = int(self.x_size*self.y_size)
            self.u_current = np.reshape(self.u_current,((self.v_size,1)))
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
            self.b = np.reshape(self.b_add_column,(self.x_size*self.y_size,1))
            
            
        elif(wall=='left'):
            new_x_size = self.x_size +1
            for i in range(len(self.left)):
                index = (i)*self.x_size
                self.b[index][0] = self.b[index][0]-self.left[i]
            b_reshape = np.reshape(self.b,(self.y_size,self.x_size))
            self.b_add_column = np.c_[np.zeros((self.x_size,1)),b_reshape]
            self.b_add_column[0][0] = +self.bottom[self.x_size-1]
            self.b_add_column[self.y_size-1][0] = +self.top[self.x_size-1]
            self.bottom = np.append(self.bottom,self.bottom[self.x_size-1])
            self.top = np.append(self.top,self.top[self.x_size-1])
            #update the u_new to contain the previous boundary values
            self.u_current = np.reshape(self.u_current,(self.y_size,self.x_size))
            self.u_current = np.c_[-self.left*self.dx**2,self.u_current]
            self.x_size = new_x_size
            self.v_size = int(self.x_size*self.y_size)
            self.u_current = np.reshape(self.u_current,((self.v_size,1)))
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
            self.b = np.reshape(self.b_add_column,(self.x_size*self.y_size,1))
        self.add_neuman_condition(value,wall)

            
    
    #creates our initial walls
    def create_walls(self, wall_bottom, wall_top, wall_right, wall_left):
        
        self.bottom = -np.full((self.x_size),wall_bottom)/(self.dx**2)
        self.top = -np.full((self.x_size),wall_top)/(self.dx**2)
        self.right = -np.full((self.y_size),wall_right)/(self.dx**2)
        self.left = -np.full((self.y_size),wall_left)/(self.dx**2)
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
    
    def relax(self,omega):
        #for i in range(len(u_new)):
        self.u_new= omega*self.u_new+(1-omega)*self.u_current
        self.update()

#def main():

#initialization

dx = 1/20


room_two = Room(np.array([1,2]),dx)
room_two.create_walls(5,40,15,15)
u_two = room_two.solve()

room_one = Room(np.array([1,1]),dx)
room_three = Room(np.array([1,1]),dx)
room_one.create_walls(15,15,15,40)
u_one = room_one.solve()

room_three.create_walls(15,15,40,15)
u_three = room_three.solve()

#adds the changing boundary due to the neumann condition.
u_one = np.reshape(u_one,(room_one.y_size,room_one.x_size))
u_one = np.c_[u_one,np.full((room_one.x_size,1),15)]
u_three = np.reshape(u_three,(room_three.y_size,room_three.x_size))
u_three = np.c_[np.full((room_three.x_size,1),15),u_three]

room_one.update()
room_two.update()
room_three.update()

u_two_new_left = room_two.get_boundary_values('old', 1, 0, 'left')
u_two_new_right = room_two.get_boundary_values('old', 1, 1, 'right')
room_one.add_neumann_wall('right',u_two_new_left)
room_three.add_neumann_wall('left',u_two_new_right)


u_two_new_right2 = room_two.get_boundary_values('old', 0.5, 0.5, 'right')
room_four = Room(np.array([0.5,0.5]),dx) 
room_four.create_walls(40,15,15,15) # bottom, top, right, left
room_four.solve()
room_four.update()
room_four.add_neumann_wall('left',u_two_new_right2)




#start the iteration
for i in range(10):
    u_one_old = room_one.get_boundary_values('old',1,0,'right')
    u_three_old = room_three.get_boundary_values('old',1,0,'left')
    
    u_four_old = room_four.get_boundary_values('old',0.5,0,'left')
    room_two.update_dirichlt_condition('right', 0.5, u_four_old)
    
    room_two.update_dirichlt_condition('right', 1, u_three_old)
    room_two.update_dirichlt_condition('left', 0, u_one_old)
    room_two.solve()
    
    u_two_new_left = room_two.get_boundary_values('new', 1, 0, 'left')
    u_two_new_right = room_two.get_boundary_values('new', 1, 1, 'right')
    room_one.update_neuman_condition('right', u_two_new_left)
    room_three.update_neuman_condition('left', u_two_new_right)
    room_one.solve()
    room_three.solve()
    
    bounds_r4 = room_two.get_boundary_values('old', 0.5, 0.5, 'right')
    room_four.update_neuman_condition('left', bounds_r4)
    room_four.solve()
    room_four.relax(0.8)
    
    #do the realaxation. 
    room_one.relax(0.8)
    room_two.relax(0.8)
    room_three.relax(0.8)

#if __name__ == "__main__":
#    main()
    
    