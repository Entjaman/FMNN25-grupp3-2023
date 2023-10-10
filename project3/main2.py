from mpi4py import MPI
import numpy as np
from scipy.linalg import solve 
from scipy.sparse import csr_matrix
from room import Room
from matplotlib import pyplot as plt

# Initialize MPI
comm = MPI.Comm.Clone( MPI.COMM_WORLD )
rank = comm.Get_rank()
size = comm.Get_size()

dx = 1/20

# Define room-specific variables, boundary conditions, and parameters
## ROOM 2
room_two = Room(np.array([1,2]),dx) # size 1x2
room_two.create_walls(5,40,15,15)
u_two = room_two.solve()
room_two.update()
u_two_new_left = room_two.get_boundary_values('old', 1, 0, 'left')
u_two_new_right = room_two.get_boundary_values('old', 1, 1, 'right')
u_two_new_right2 = room_two.get_boundary_values('old', 0.5, 0.5, 'right')

## ROOM 1
room_one = Room(np.array([1,1]),dx) # size 1x1
room_one.create_walls(15,15,15,40)
u_one = room_one.solve()
room_one.update()
room_one.add_neumann_wall('right',u_two_new_left)

## ROOM 3
room_three = Room(np.array([1,1]),dx) # size 1x1
room_three.create_walls(15,15,40,15)
u_three = room_three.solve()
room_three.update()
room_three.add_neumann_wall('left',u_two_new_right)

## ROOM 4
room_four = Room(np.array([0.5,0.5]),dx) 
room_four.create_walls(40,15,15,15) # bottom, top, right, left
u_four = room_four.solve()
room_four.update()
room_four.add_neumann_wall('left',u_two_new_right2)



# rank 0 is room 2
# rank 1 is room 1
# rank 2 is room 3
# rank 3 is room 4

omega = 0.8
iterations_count = 10

# Main iterative loop
for iter in range(iterations_count):
    print(iter)
    # Step 1: On Room 2, obtain v_k_r for room 1, 4 and 3 and send new  bound to Room 1 and Room 3 and Room 4
    if rank == 0: # TODO maybe modify rank nbr after order and not by room order
        print("Rank is 0")
        if(iter==0): #first iteration
            bounds_r1 = room_two.get_boundary_values('old', 1, 0, 'left')
            bounds_r3 = room_two.get_boundary_values('old', 1, 1, 'right')
            bounds_r4 = room_two.get_boundary_values('old', 0.5, 0.5, 'right')

            comm.send(bounds_r1, dest = 1)
            comm.send(bounds_r3, dest = 2)
            comm.send(bounds_r4, dest = 3)
        else:
            u_one_old = comm.recv(source = 1) # boundary for room 1
            u_three_old = comm.recv(source = 2) # boundary for room 3
            u_four_old = comm.recv(source = 3) # boundary for room 4
            ## update boundaries for room1, 4 and 3 with b1, b4 and b3
            room_two.update_dirichlt_condition('right', 1, u_three_old)
            room_two.update_dirichlt_condition('left', 0, u_one_old)
            room_two.update_dirichlt_condition('right', 0.5, u_four_old)
            ## solve room 2 with new boundaries
            room_two.solve()

            bounds_r1 = room_two.get_boundary_values('new', 1, 0, 'left')
            bounds_r3 = room_two.get_boundary_values('new', 1, 1, 'right')
            bounds_r4 = room_two.get_boundary_values('new', 0.5, 0.5, 'right')

            comm.send(bounds_r1, dest = 1)
            comm.send(bounds_r3, dest = 2)
            comm.send(bounds_r4, dest = 3)
            room_two.relax(omega)

    # Step 2: On Room 1, receive v_k_r from Room 2 and solve left system
    if rank == 1:
        print("Rank is 1")
        bounds_r1 = comm.recv(source = 0)
       
       ## TODO update boundaries for room 1
        room_one.update_neuman_condition('right', bounds_r1)
        room_one.solve()
        room_one.relax(omega)
        u1_kp1 = room_one.get_boundary_values('old',1,0,'right')

        comm.send(u1_kp1, dest = 0) ## send to room 2

    if rank == 2:
        print("Rank is 2")
        bounds_r3 = comm.recv(source = 0)
       
        ##  update boundaries for room 3
        room_three.update_neuman_condition('left', bounds_r3)
        room_three.solve()
        room_three.relax(omega)
        u3_kp1 = room_three.get_boundary_values('old',1,0,'left')

        comm.send(u3_kp1, dest = 0)

    if rank == 3:
        print("Rank is 3")
        bounds_r4 = comm.recv(source = 0)
       
        ##  update boundaries for room 3
        room_four.update_neuman_condition('left', bounds_r4)
        room_four.solve()
        room_four.relax(omega)
        u4_kp1 = room_four.get_boundary_values('old',0.5,0.5,'right')

        comm.send(u4_kp1, dest = 0)


    if(iter == iterations_count-1):
        if rank == 0:
            # comm.send(u_two, dest=3, tag=2)
            u_2 = np.flipud(np.reshape(room_two.u_current,(6,3)))
            print('matrix two', u_2)
            plt.imshow(u_2, cmap='hot', interpolation='nearest')
            plt.show()
        if rank == 1:
            u_1 = np.flipud(np.reshape(room_one.u_current,(3,4)))
            print('matrix one', u_1)
            plt.imshow(u_1, cmap='hot', interpolation='nearest')
            plt.show()
            # comm.send(u_one, dest=3, tag=1)
        if rank == 2:
            u_3 = np.flipud(np.reshape(room_three.u_current,(3,4)))
            print('matrix three', u_3)
            plt.imshow(u_3, cmap='hot', interpolation='nearest')
            plt.show()
            # comm.send(u_three, dest=3, tag=3)
        if rank == 3:
            u_4 = np.flipud(np.reshape(room_four.u_current,(3,4)))
            print('matrix three', u_4)
            plt.imshow(u_4, cmap='hot', interpolation='nearest')
            plt.show()
            # comm.send(u_three, dest=3, tag=3)

    # ## TODO add if rank == 3 --> plot ...
    # if rank == 3:
    #     u_one = comm.recv(source = 1, tag=1)
    #     u_two = comm.recv(source=0, tag=2)
    #     u_three = comm.recv(source = 2, tag=3)


    
    # comm.Barrier()

# Finalize MPI
MPI.Finalize()