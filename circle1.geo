// Gmsh project created on Fri Sep 23 15:52:20 2022
SetFactory("OpenCASCADE");
//+
Circle(1) = {0, 0, 0, 1, 0, 2*Pi};
//+
Curve Loop(1) = {1};
//+
Curve Loop(2) = {1};
//+
Plane Surface(1) = {2};
