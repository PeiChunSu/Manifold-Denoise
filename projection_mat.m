function P=projection_mat(n)
A=colbasis(magic(n));
P=A*inv(A'*A)*A';
end