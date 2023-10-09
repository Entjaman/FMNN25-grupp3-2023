from mpi4py import MPI
import numpy as np
from scipy.linalg import solve 
from scipy.sparse import csr_matrix
from room import Room

# Initialize MPI
comm = MPI.Comm.Clone( MPI.COMM_WORLD )
rank = comm.Get_rank()
size = comm.Get_size()

dx = 1/3 

# Define room-specific variables, boundary conditions, and parameters
## ROOM 2
room_two = Room(np.array([1,2]),dx) # size 1x2
room_two.create_walls(5,40,15,15)
u_two = room_two.solve()

## ROOM 1
room_one = Room(np.array([1,1]),dx) # size 1x1
room_one.create_walls(15,15,15,40)
u_one = room_one.solve()
u1_km1 = u_one

## ROOM 3
room_three = Room(np.array([1,1]),dx) # size 1x1
room_three.create_walls(15,15,40,15)
u_three = room_three.solve()
u3_km1 = u_three


# rank 0 is room 2
# rank 1 is room 1
# rank 2 is room 3

omega = 0.8
iterations_count = 10

# Main iterative loop
for iter in range(iterations_count):
    # Step 1: On Room 2, obtain v_k_r for room 1 and 3 and send new  bound to Room 1 and Room 3
    if rank == 0: # TODO maybe modify rank nbr after order and not by room order
        print("Rank is 0")
        if(iter==0): #first iteration
            ## TODO intitial solver from Room for room 2
            break
        else:
            b1 = comm.recv(source = 1) # boundary for room 1
            b3 = comm.recv(source = 2) # boundary for room 3
            ## TODO update boundaries for room1 and 3 with b1 and b3
            ## TODO solve room 2 with new boundaries

        ## TODO calculated_bound1 = ... calc new boundary for room 1 from room 2
        ## TODO calculated_bound3 = ... calc new boundary for room 3 from room 2

        ## TODO comm.send(calculated_bound1, dest = 1)
        ## TODO comm.send(calculated_bound3, dest = 2)

    # Step 2: On Room 1, receive v_k_r from Room 2 and solve left system
    if rank == 1:
        print("Rank is 1")
        bounds_r1 = comm.recv(source = 0)
       
       ## TODO update boundaries for room 1
        u1_kp1 = room_one.solve()
        u1_kp1 = omega*u1_kp1 +(1-omega)*u1_km1
        u1_km1 = u1_kp1
        comm.send(u1_kp1, dest = 0) ## send to room 2

    if rank == 2:
        print("Rank is 2")
        bounds_r3 = comm.recv(source = 0)
       
        ## TODO update boundaries for room 3
        u3_kp1 = room_three.solve()
        u3_kp1 = omega*u3_kp1 + (1-omega)*u3_km1
        u3_km1 = u3_kp1
        comm.send(u3_kp1, dest = 0)

    if(iter == iterations_count-1):
        if rank == 0:
            comm.send(u_two, dest=3, tag=2)
        if rank == 1:
            comm.send(u_one, dest=3, tag=1)
        if rank == 2:
            comm.send(u_three, dest=3, tag=3)

    ## TODO add if rank == 3 --> plot ...
    if rank == 3:
        u_one = comm.recv(source = 1, tag=1)
        u_two = comm.recv(source=0, tag=2)
        u_three = comm.recv(source = 2, tag=3)


    
    # comm.Barrier()

# Finalize MPI
MPI.Finalize()