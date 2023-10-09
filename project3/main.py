from mpi4py import MPI
import numpy as np
from scipy.linalg import solve 
from scipy.sparse import csr_matrix

# Initialize MPI
comm = MPI.Comm.Clone( MPI.COMM_WORLD )
rank = comm.Get_rank()
size = comm.Get_size()

# Define room-specific variables, boundary conditions, and parameters
# Define initial temperature values for omega1 and omega2 based on rank

# Define room-specific variables, boundary conditions, and parameters
# Define initial temperature values for each room based on rank
## TODO m
if rank == 0:  # Room 1
    u_k = np.ones((2, 1)) * 15.0
elif rank == 1:  # Room 2
    u_k = np.ones((2, 2)) * 15.0
elif rank == 2:  # Room 3
    u_k = np.ones((2, 1)) * 15.0

# Define relaxation parameter (w)
omega = 0.8
dx = 1/20

# row indices, column indices and values
rowptr = np.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 3])
colind = np.array([0, 1, 0, 1, 2, 1, 2, 3, 2, 3])
values = np.array([-2.,1.,1.,-2.,1.,1.,-2.,1.,1.,-2.])
# create sparse from COO format
A = csr_matrix( (values, (rowptr, colind)) )

# setup right hand side
b = np.array([1.,2.,2.,1.])

# Main iterative loop
for iteration in range(10):
    # Step 1: On Room 2, obtain v_k_r and send to Room 1
    if rank == 1:
        v_k_r = np.copy(u_k[:, -1])  # Extract the right boundary values of Room 2
        print("P[",rank,"] sent data =",v_k_r)
        comm.send(v_k_r, dest=0)

    # Step 2: On Room 1, receive v_k_r from Room 2 and solve left system
    if rank == 0:
        # Receive v_k_r from Room 2
        v_k_r = comm.recv(source=1)

        # Solve the left system with v_k_r as Dirichlet condition
        left_matrix = np.zeros((2, 2))  # Define the left matrix based on your problem
        right_vector = np.zeros((2, 1))  # Define the right vector based on your problem

        # Fill in the left_matrix and right_vector based on your problem

        # Solve the left system using scipy.linalg.solve
        u_k[:, 0] = solve(A, b).flatten()

        # Calculate the Neumann condition at r
        n_c = (u_k[1, 0] - 2 * u_k[0, 0] + v_k_r[0]) / (dx**2)  # 2nd order central differences

        # Send the computed Neumann condition to Room 2
        print("P[",rank,"] sent data =",n_c)
        comm.send(n_c, dest=1)

    # Step 3: On Room 2, receive data from Room 1 and solve the right system
    if rank == 1:
        n_c = comm.recv(source=0)  # Receive Neumann condition from Room 1
        u_k[:, -1] = n_c  # Set the right boundary values of Room 2 using Neumann condition

        # Solve the right system with Neumann condition to obtain u_k+1_r
        u_k[:, -1] = solve(A, n_c).flatten()

    # Step 4: Relaxation
    u_k = omega * u_k + (1 - omega) * u_k  # Apply relaxation

    # Synchronize processes
    comm.Barrier()

# Finalize MPI
MPI.Finalize()